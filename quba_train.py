from panqec.codes import surface_2d
from panqec.error_models import PauliErrorModel
from panqec.decoders import MatchingDecoder, BeliefPropagationOSDDecoder
import sys
import torch
import random
import hashlib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datetime import timedelta

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import os

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from functions import QuBA, collate, compute_accuracy, logical_error_rate, \
    construct_tanner_graph_edges, generate_syndrome_error_volume, adapt_trainset, ler_loss,bb_code,save_model, load_model, copbb_code

import argparse

from codes_q import *


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=1800))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    return local_rank, rank, torch.device(f'cuda:{local_rank}')


def sync_tensor(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / dist.get_world_size()

def md5_hash_array(arr: np.ndarray) -> str:
    arr_bytes = arr.tobytes()
    return hashlib.md5(arr_bytes).hexdigest()


parser = argparse.ArgumentParser(description='Distributed GNN Decoder Training for QEC')

parser.add_argument('--c_dist', type=int, default=6, help='Code distance')
parser.add_argument('--len_train_set', type=int, default=3000, help='Length of training set')
parser.add_argument('--len_test_set', type=int, default=300, help='Length of test set')
parser.add_argument('--n_iters', type=int, default=30)


parser.add_argument('--n_node_features', type=int, default=32)
parser.add_argument('--n_edge_features', type=int, default=32)
parser.add_argument('--msg_net_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=16)
#parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')



#fixed 0.0005 0.00016
parser.add_argument('--max_train_err_rate', type=float, default=0.15, help='Max training error rate')
parser.add_argument('--test_err_rate', type=float, default=0.05, help='Test error rate')

parser.add_argument('--epochs', type=int, default=120)



args = parser.parse_args()


# Override constants with parsed values
c_dist = args.c_dist
len_train_set = args.len_train_set
len_test_set = args.len_test_set
n_iters = args.n_iters

n_node_features = args.n_node_features
n_edge_features = args.n_edge_features
msg_net_size = args.msg_net_size
batch_size = args.batch_size


max_train_err_rate = args.max_train_err_rate
test_err_rate = args.test_err_rate

epochs = args.epochs

#batch_size = args.batch_size

set_seed(42 + int(os.environ["LOCAL_RANK"]))
local_rank, rank, device = setup_ddp()
use_amp = True

"""
    Parameters
"""
#d = 18
error_model_name = "DP"
if(error_model_name == "X"):
    error_model = [1, 0, 0]
elif (error_model_name == "Z"):
    error_model = [0, 0, 1]
elif (error_model_name == "XZ"):
    error_model = [0.5, 0, 0.5]
elif (error_model_name == "DP"):
    error_model = [1/3, 1/3, 1/3]

size = 2 * c_dist ** 2 - 1
n_node_inputs = 4
n_node_outputs = 4
#n_iters = 65
#n_node_features = 32
#n_edge_features = 32
#len_test_set = 5000
#test_err_rate = 0.10

#len_train_set = len_test_set * 20
#len_train_set = 100000
#max_train_err_rate = 0.15

lr = 0.0005
weight_decay = 0.0001
msg_net_dropout_p = 0.1
gru_dropout_p = 0.1

heads = 4

#d5_X_itr_nf_ef_maxtdata_maxe_testdata_ter_lr_wd_

# fname = f"trained_models/d{d}_X_{n_iters}_{n_node_features}_{n_edge_features}_{len_train_set}_{max_train_err_rate}_{len_test_set}_{test_err_rate}_{lr}_{weight_decay}_"

"""
    Create the bb code
"""
# code = surface_2d.RotatedPlanar2DCode(dist)
# [[72,12,6]]
# code, A_list, B_list = create_bivariate_bicycle_codes(6, 6, [3], [1,2], [1,2], [3])
# d = 6
# [[144,12,12]]
# code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1,2], [1,2], [3])
# d = 12
# [[288,12,18]]
# code, A_list, B_list = create_bivariate_bicycle_codes(12, 12, [3], [2,7], [1,2], [3])
#batch_size = 16

if rank == 0:
    print("Error Model: ",error_model_name)

    print("\n===== Hyperparameters =====")
    print(f"  n_node_inputs={n_node_inputs}, n_node_outputs={n_node_outputs},\n"
        f"  n_node_features={n_node_features}, n_edge_features={n_edge_features},\n"
        f"  lr={lr}, weight_decay={weight_decay},\n"
        f"  msg_net_size={msg_net_size}, msg_net_dropout_p={msg_net_dropout_p}, gru_dropout_p={gru_dropout_p}")
    print(f"  batch_size={batch_size}")
    print("loss = ler loss + sloss ")
    
    #print(f"Parsed args: c_dist={c_dist}, len_train_set={len_train_set}, max_train_err_rate={max_train_err_rate}, len_test_set={len_test_set}, test_err_rate={test_err_rate}, heads={heads}")
    print("===== Parsed Args =====")
    print(f"  c_dist={c_dist}, len_train_set={len_train_set}, max_train_err_rate={max_train_err_rate},\n"
        f"  len_test_set={len_test_set}, test_err_rate={test_err_rate}, heads={heads},\n"
        f"  n_iters={n_iters}, epochs={epochs}")
    
#d = 18
#dist = d
# dist = 1

#code = bb_code(c_dist)
code = copbb_code(c_dist)

if rank == 0:
    print(f"train: {c_dist}, code name: {code.name}_d{c_dist}")   

gnn = QuBA(dist=c_dist, n_node_inputs=n_node_inputs, n_node_outputs=n_node_outputs, n_iters=n_iters,
                 n_node_features=n_node_features, n_edge_features=n_edge_features,
                 msg_net_size=msg_net_size, msg_net_dropout_p=msg_net_dropout_p, gru_dropout_p=gru_dropout_p, heads=heads)
gnn.to(device)
#gnn.to(local_rank)
gnn = torch.nn.parallel.DistributedDataParallel(gnn, device_ids=[local_rank])

src, tgt = construct_tanner_graph_edges(code)
src_tensor = torch.LongTensor(src)
tgt_tensor = torch.LongTensor(tgt)
QuBA.construct_tanner_graph_edges = (src_tensor, tgt_tensor)

hxperp = torch.FloatTensor(code.hx_perp).to(device)
hzperp = torch.FloatTensor(code.hz_perp).to(device)
QuBA.hxperp = hxperp
QuBA.hzperp = hzperp

#QuBA.device = torch.device(f"cuda:{local_rank}")
QuBA.device = device

#total_params = sum(param.numel() for param in gnn.parameters())

#fnameload = f"trained_models/BB_n288_k12_d18_from_d18_DP_45_50_50_40000_0.2_2000_0.1_0.0001_0.0001_128_0.05_0.05_gnn.pth 0.041_0.043_0.0435 100"
# fnameload = f"trained_models/BB_n72_k12_d6_from_d6_DP_30_50_50_20000_0.15_1000_0.06_0.0001_0.001_128_0.05_0.05_gnn.pth 0.041_0.015_0.041 27"
# fnameload = f"trained_models/BB_n72_k12_d6_DP_30_50_50_20000_0.15_1000_0.06_0.0001_0.001_128_0.05_0.05_gnn.pth 0.035_0.026_0.04 110"


fnamenew = f"bnn_testing/copbb/{code.name}_d{c_dist}_{error_model_name}_{n_iters}_{n_node_features}_{n_edge_features}_" \
           f"{len_train_set}_{max_train_err_rate}_{len_test_set}_{test_err_rate}_{lr}_" \
           f"{weight_decay}_{msg_net_size}_{msg_net_dropout_p}_{gru_dropout_p}_"
optimizer = optim.AdamW(gnn.parameters(), lr=lr, weight_decay=weight_decay)

#exploration_samples = 10**7
#lr_reduce_epoch_step = exploration_samples // len_train_set
max_training_data = 10**8
end_training_epoch = max_training_data // len_train_set

scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.5)

""" automatic mixed precision """
scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp)

criterion = nn.CrossEntropyLoss()
losses = []
test_losses = []
frac = []
repoch = []
le_rates = np.zeros((epochs,5), dtype='float')
train_time = []
start_time = time.time()


# Generate the training set
train_data = generate_syndrome_error_volume(code, error_model, p=max_train_err_rate, batch_size=len_train_set)
#print("Training set size:", len(train_data))

#print("Train MD5:\n", md5_hash_array(train_data))
#os.makedirs("data/traindata", exist_ok=True)
#np.save("data/traindata/d18_bb_train_set2.npy", train_data)

trainset = adapt_trainset(train_data, code, num_classes=n_node_inputs)
train_sampler = DistributedSampler(trainset)
trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate, pin_memory=True,
    num_workers=4)

#Load the validation data
#val_data = np.load("data/traindata/d18_bb_val_set.npy")

# Generate the validation set
val_data = generate_syndrome_error_volume(code, error_model=error_model, p=test_err_rate, batch_size=len_test_set, for_training=False)
#print("Test set size:", len(val_data))

#print("Val MD5:\n", md5_hash_array(val_data))
#os.makedirs("data/traindata", exist_ok=True)
#np.save("data/traindata/d18_bb_val_set2.npy", val_data)

testset = adapt_trainset(val_data, code, num_classes=n_node_inputs, for_training=False)
test_sampler = DistributedSampler(testset)
testloader = DataLoader(testset, batch_size=512, sampler=test_sampler, collate_fn=collate, pin_memory=True,
    num_workers=4)

for r in range(dist.get_world_size()):
    dist.barrier()
    if rank == r:
        print(f"[Rank {rank}] Train MD5: {md5_hash_array(train_data)}")
        print(f"[Rank {rank}] Val MD5:   {md5_hash_array(val_data)}")
        sys.stdout.flush()
    dist.barrier()


if rank == 0:
    print("epoch, lr, wd, LER_X, LER_Z, LER_tot, test loss, bgnn_tot_train_loss, gnn_train_loss, kl, beta, beta*kl, train time")

        
        
size = 2 * code.N
error_index = code.N
# size = QuBA.dist ** 2 + error_index
min_test_err_rate = test_err_rate
min_ler_tot = test_err_rate
fname_traning_data=""
# if max_training_data:# <= 10**8:
#     fname_traning_data = f"training_data/d{dist}_X_{10**8}_{max_train_err_rate}.npy"
#     if os.path.isfile(fname_traning_data):
#         print("Training data loaded from file")
#     else:
#         raise Exception("Training data file doesn't exists!")
# else:
#     raise Exception("Required training data too large!!!")

# training_data = np.load(fname_traning_data, mmap_mode="r")


kl_annealing_epochs = 10  # warm-up period
kl_scale = 1e-6
#lambda_l2 = 1e-5


patience = 15
no_improve_epochs = 0
prev_best_ler = float('inf')
#prev_ler_vals = []

stop_training = torch.tensor([0], device=device)  # 0: continue, 1: stop

for epoch in range(epochs):
    train_sampler.set_epoch(epoch)
    test_sampler.set_epoch(epoch)
    
    #beta = min(1.0, epoch / kl_annealing_epochs)
    beta = min(1.0, epoch / kl_annealing_epochs) * kl_scale
    
    #print(f"Epoch {epoch}, beta = {beta:.3f}")
    
    gnn.train()
    if epoch == end_training_epoch:
        break
    # print(epoch)
    # trainset = np.copy(training_data[len_train_set*epoch:len_train_set*(epoch+1),:])
    # trainset = adapt_trainset(trainset,code,num_classes=n_node_inputs)
    # trainset = adapt_trainset(generate_syndrome_error_volume(code, error_model, p=max_train_err_rate, batch_size=len_train_set),
    #                           code, num_classes=n_node_inputs)
    # trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate, shuffle=False)
    epoch_loss = []
    for i, (inputs, targets, src_ids, dst_ids) in enumerate(trainloader):# loop over minibatches-Iterates over batches produced by your trainloader
        inputs, targets = inputs.to(device), targets.to(device)
        src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
        
        
        # This assertion fails:
        #assert src_ids.max() < inputs.size(1)
        
        # Add these checks before the outputs = model18(...) call
        #print(f"Input shapes - inputs: {inputs.shape}, src_ids: {src_ids.shape}, dst_ids: {dst_ids.shape}")
        #print(f"Max src_ids: {src_ids.max()}, Max dst_ids: {dst_ids.max()}")
        #print(f"Input tensor size(1): {inputs.size(1)}")
        
        #print("inputs.shape", inputs.shape)
        #print("src_ids.max()", src_ids.max())

        #assert src_ids.max() < inputs.size(1), f"src_ids.max={src_ids.max()} is out of bounds for inputs.size(1)={inputs.size(1)}"
        #assert dst_ids.max() < inputs.size(1), f"dst_ids.max={dst_ids.max()} is out of bounds for inputs.size(1)={inputs.size(1)}"
    
    
        #optimizer.zero_grad()
        loss = 0

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):#automatic mixed precision training-do forward and loss computation in mixed precision
            #assert src_ids.max() < inputs.size(1), f"src_ids.max={src_ids.max()} is out of bounds for inputs.size(1)={inputs.size(1)}"
            #assert dst_ids.max() < inputs.size(1), f"dst_ids.max={dst_ids.max()} is out of bounds for inputs.size(1)={inputs.size(1)}"
            
            outputs = gnn(inputs, src_ids, dst_ids)#Run the GNN Forward Pass-The GNN performs message passing over several iterations and returns(outputs = [out_0, out_1, ..., out_T])
                #where each out_j is a prediction tensor at iteration step j.
            #print(f"[DEBUG] outputs.shape = {outputs.shape}")  
            for j, out in enumerate(outputs): ## process the output out from each iteration step j (outputs[j]: [B, C]-j is the step of msg-passing iteration)
                #print(f"[DEBUG] out.shape = {out.shape}")            # should be [num_nodes, 4]
                #print(f"[DEBUG] targets.shape = {targets.shape}")    # should be [batch_size, num_nodes] or [num_nodes]
                #print(f"[DEBUG] error_index = {error_index}")

                eloss = criterion(out.view(-1, size, n_node_inputs)[:, error_index:].reshape(-1, n_node_inputs),
                                  targets.view(-1, size)[:, error_index:].flatten())
                sloss = criterion(out.view(-1, size, n_node_inputs)[:, :error_index].reshape(-1, n_node_inputs),
                                  targets.view(-1, size)[:, :error_index].flatten())
                ler_los = ler_loss(out, targets, code)
                #loss += ler_loss(out, targets, code) + sloss + eloss#ler_loss(out, targets, code) +
                loss += ler_los + 0.5 * (sloss + eloss)
            loss /= outputs.shape[0]# the average loss across all message-passing iterations
            
            # Add KL divergence term
            kl = gnn.module.kl_divergence()
            total_loss = loss + beta * kl  # use annealed or fixed beta
            '''
            if i == 0 and rank == 0:
                print(f"[Epoch {epoch}, Step {i}] eloss={eloss.item():.4f}, sloss={sloss.item():.4f}, "
                  f"ler={ler_los.item():.4f}, kl={kl.item():.4f}, beta={beta:.4f}, total_loss={total_loss.item():.4f}")
            '''
            '''
            # Optional: Add L2 regularization manually
            l2_reg = torch.tensor(0., device=device)
            for param in gnn.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param, p=2)**2

            total_loss += lambda_l2 * l2_reg
            '''
            
            # print(loss)
            if use_amp:
                scaler.scale(total_loss).backward()
                #Gradient Clipping - To avoid exploding gradients during training with mixed precision (AMP)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                # print(loss)
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        epoch_loss.append(total_loss.detach())
    epoch_loss = torch.mean(torch.tensor(epoch_loss)).item()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        test_loss = compute_accuracy(gnn.module, testloader, code)
        lerx, lerz, ler_tot = logical_error_rate(gnn.module, testloader, code)

        # Wrap each in tensors and sync
        #fraction_solved = sync_tensor(torch.tensor(fraction_solved, device=device)).item()
        test_loss = sync_tensor(torch.tensor(test_loss, device=device)).item()
        lerx = sync_tensor(torch.tensor(lerx, device=device)).item()
        lerz = sync_tensor(torch.tensor(lerz, device=device)).item()
        ler_tot = sync_tensor(torch.tensor(ler_tot, device=device)).item()


    scheduler.step() # update lr
    #print(optimizer.param_groups[0]['lr'],optimizer.param_groups[0]["weight_decay"])

    # repoch.append(epoch)
    # losses.append(epoch_loss)
    # test_losses.append(test_loss)
    # frac.append(fraction_solved)
    le_rates[epoch, 0] = lerx
    le_rates[epoch, 1] = lerz
    le_rates[epoch, 2] = ler_tot
    le_rates[epoch, 3] = test_loss
    le_rates[epoch, 4] = epoch_loss
    curr_time = time.time() - start_time
    # train_time.append(curr_time)
    
    
    if rank == 0:
        print(epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[0]["weight_decay"], lerx, lerz, ler_tot, test_loss, epoch_loss, loss.item(), kl.item(), beta, beta * kl.item(), curr_time) 
        # fraction_solved,
        #print(f"[Epoch {epoch}, Step {i}] eloss={eloss.item():.4f}, sloss={sloss.item():.4f}, ler_loss={ler_loss_value.item():.4f}, kl={kl.item():.4f}, total_loss={total_loss.item():.4f}, beta={beta:.4f}")

        if ler_tot < min_test_err_rate:
            min_test_err_rate = ler_tot
            min_ler = ler_tot
            save_model(gnn.module, fnamenew + f'gnn.pth {lerx}_{lerz}_{ler_tot} {epoch}', confirm=False)

        if ler_tot < min_ler:
            min_ler_tot = ler_tot
            save_model(gnn.module, fnamenew + f'gnn.pth {lerx}_{lerz}_{ler_tot} {epoch}', confirm=False)

        if epoch % 10==0:
            np.save(fnamenew+f'training_lers_and_losses',le_rates)
            save_model(gnn.module, fnamenew + f'gnn.pth {lerx}_{lerz}_{ler_tot} {epoch}', confirm=False)
            torch.cuda.empty_cache()

        if ler_tot == 0:
            min_ler_tot = ler_tot
            save_model(gnn.module, fnamenew + f'gnn.pth {lerx}_{lerz}_{ler_tot} {epoch}', confirm=False)
            stop_training.fill_(1)
            print(f"[Early Stopping] ler_tot=0. Stopping at epoch {epoch}.")
            
        # Track best LER improvement
        if ler_tot + 1e-6 < prev_best_ler:
            prev_best_ler = ler_tot
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
        # Check patience for best LER improvement
        if no_improve_epochs >= patience:
            print(f"[Early Stopping] No improvement in LER for {patience} epochs. Stopping at epoch {epoch}.")
            stop_training.fill_(1)
        
    dist.broadcast(stop_training, src=0)
    if stop_training.item() == 1:
        break

dist.destroy_process_group()

'''
        # Track repeated LER values
        prev_ler_vals.append(round(ler_tot, 5))
        if prev_ler_vals.count(round(ler_tot, 5)) >= 20:
            print(f"[Early Stopping] LER {ler_tot} repeated 20 times. Stopping at epoch {epoch}.")
            stop_training.fill_(1)
'''