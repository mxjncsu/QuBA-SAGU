import torch
import random

import os
import math
import time
import copy

import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import hashlib
import numpy as np
import torch.nn as nn

from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta

from functions import QuBA, collate, compute_accuracy, logical_error_rate, \
    construct_tanner_graph_edges, generate_syndrome_error_volume, adapt_trainset, ler_loss, bb_code, save_model, load_model
    
# === Seed and DDP Setup ===
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_ddp():
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=1800))
    ##dist.init_process_group(backend="nccl")
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

def average_model_weights_weighted(models, weights):
    """Average weights of DDP-wrapped models using specified weights.

    Args:
        models: list of (dist, code, DDP-wrapped model)
        weights: list of floats (same length), should sum to 1

    Returns:
        avg_model: a new model with averaged weights (wrapped in DDP)
    """
    assert len(models) == len(weights), "Number of models and weights must match"
    assert abs(sum(weights) - 1.0) < 1e-5, "Weights must sum to 1"
    
    avg_model = copy.deepcopy(models[0][2])  # DDP-wrapped model
    avg_state_dict = avg_model.module.state_dict()  # Get the inner model's state_dict

    model_state_dicts = [m.module.state_dict() for (_, _, m) in models]

    # Weighted average of parameters
    for key in avg_state_dict:
        avg_state_dict[key] = sum(w * model_state_dicts[i][key] for i, w in enumerate(weights))

    # Load back to the avg_model's .module
    avg_model.module.load_state_dict(avg_state_dict)
    return avg_model


def init_model(code, c_dist, n_node_features, n_edge_features, msg_net_size, n_iters):
    model = QuBA(
        dist=c_dist,
        n_node_inputs=n_node_inputs,
        n_node_outputs=n_node_outputs,
        n_iters=n_iters,
        n_node_features=n_node_features,
        n_edge_features=n_edge_features,
        msg_net_size=msg_net_size,
        msg_net_dropout_p=msg_net_dropout_p,
        gru_dropout_p=gru_dropout_p,
        heads=heads
    ).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    src, tgt = construct_tanner_graph_edges(code)
    src_tensor = torch.LongTensor(src)
    tgt_tensor = torch.LongTensor(tgt)
    QuBA.construct_tanner_graph_edges = (src_tensor, tgt_tensor)
    
    QuBA.hxperp = torch.FloatTensor(code.hx_perp).to(device)
    QuBA.hzperp = torch.FloatTensor(code.hz_perp).to(device)
    QuBA.device = device
    
    return model

def train_domain_model(model, code, c_dist, trainloader, criterion, optimizer, scaler, epoch):
#def train_domain_model(model, code, criterion, optimizer, scaler, beta, use_amp, device):
#def train_domain_model(model, code, c_dist, lr, epoch):
    model.train()
    error_index = code.N
    size = 2 * code.N
    beta = min(1.0, epoch / kl_annealing_epochs) * kl_scale
    
    
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
    
    '''
    sub_train_data = generate_syndrome_error_volume(code, error_model, p=max_train_err_rate, batch_size=sub_len_train_set)
    sub_trainset = adapt_trainset(sub_train_data, code, num_classes=n_node_inputs)
    sub_train_sampler = DistributedSampler(sub_trainset)
    sub_train_sampler.set_epoch(epoch)
    sub_trainloader = DataLoader(sub_trainset, batch_size=batch_size, sampler=sub_train_sampler,
                                 collate_fn=collate, pin_memory=True, num_workers=4)
    '''

    #optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


    epoch_loss = []
    for _, (inputs, targets, src_ids, dst_ids) in enumerate(trainloader):
        src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        
        iter_loss = 0.0
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs, src_ids, dst_ids)

            for j, out in enumerate(outputs): ## process the output out from each iteration step j (outputs[j]: [B, C]-j is the step of msg-passing iteration)
                eloss = criterion(out.view(-1, size, n_node_inputs)[:, error_index:].reshape(-1, n_node_inputs),
                                  targets.view(-1, size)[:, error_index:].flatten())
                sloss = criterion(out.view(-1, size, n_node_inputs)[:, :error_index].reshape(-1, n_node_inputs),
                                  targets.view(-1, size)[:, :error_index].flatten())
                ler_los = ler_loss(out, targets, code)
                #loss += ler_loss(out, targets, code) + sloss + eloss#ler_loss(out, targets, code) +
                iter_loss += ler_los + 0.5 * (sloss + eloss)

            iter_loss /= outputs.shape[0]
            kl = model.module.kl_divergence()
            batch_loss = iter_loss + beta * kl

            # Backpropagation
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()

        #total_loss += batch_loss.item()
        epoch_loss.append(batch_loss.detach())
    epoch_loss = torch.mean(torch.tensor(epoch_loss)).item()
    return epoch_loss, optimizer, model
#    return total_loss / len(trainloader), optimizer, scheduler, model

def train_epoch_warmup(model, code, trainloader, criterion, optimizer, scaler, use_amp, device):
#def train_epoch_warmup(model, code, trainloader, criterion, beta, lr, use_amp, device):
    model.train()
    error_index = code.N
    size = 2 * code.N
    beta = min(1.0, epoch / kl_annealing_epochs) * kl_scale
    #total_loss = 0.0
    
    #src_ids, dst_ids = model.module.construct_tanner_graph_edges  # Get Tanner graph from model
    #QuBA.hxperp = torch.FloatTensor(code.hx_perp).to(device)
    #QuBA.hzperp = torch.FloatTensor(code.hz_perp).to(device)
    #QuBA.device = device
    
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
    
    #optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    epoch_loss = []
    for _, (inputs, targets, src_ids, dst_ids) in enumerate(trainloader):
        src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
        inputs, targets = inputs.to(device), targets.to(device)
        
        iter_loss = 0.0
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs, src_ids, dst_ids)

            for j, out in enumerate(outputs): ## process the output out from each iteration step j (outputs[j]: [B, C]-j is the step of msg-passing iteration)
                eloss = criterion(out.view(-1, size, n_node_inputs)[:, error_index:].reshape(-1, n_node_inputs),
                                  targets.view(-1, size)[:, error_index:].flatten())
                sloss = criterion(out.view(-1, size, n_node_inputs)[:, :error_index].reshape(-1, n_node_inputs),
                                  targets.view(-1, size)[:, :error_index].flatten())
                ler_los = ler_loss(out, targets, code)
                #loss += ler_loss(out, targets, code) + sloss + eloss#ler_loss(out, targets, code) +
                iter_loss += ler_los + 0.5 * (sloss + eloss)

            iter_loss /= outputs.shape[0]
            kl = model.module.kl_divergence()
            batch_loss = iter_loss + beta * kl

            # Backpropagation
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()

        #total_loss += batch_loss.item()
        epoch_loss.append(batch_loss.detach())
    epoch_loss = torch.mean(torch.tensor(epoch_loss)).item()
    return epoch_loss, optimizer, model
#    return total_loss / len(trainloader), optimizer, scheduler, model


def evaluate_and_save_model(
    epoch, model, code, testloader, test_sampler,
    optimizer, scheduler,
    le_rates, start_time,
    train_loss,
    fnamenew, device, rank,
    stop_training, prev_best_ler, no_improve_epochs, min_test_err_rate, min_ler_tot, patience
):
        
    model.eval()
    model.to(device)
    #no_improve_epochs = 0
    test_sampler.set_epoch(epoch)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        
        #test_sampler.set_epoch(epoch)
        test_loss = compute_accuracy(model.module, testloader, code)
        lerx, lerz, ler_tot = logical_error_rate(model.module, testloader, code)

        # Sync across ranks
        test_loss = sync_tensor(torch.tensor(test_loss, device=device)).item()
        lerx = sync_tensor(torch.tensor(lerx, device=device)).item()
        lerz = sync_tensor(torch.tensor(lerz, device=device)).item()
        ler_tot = sync_tensor(torch.tensor(ler_tot, device=device)).item()

    # Update learning rate
    scheduler.step()

    # Logging LERs and loss
    le_rates[epoch] = [lerx, lerz, ler_tot, train_loss, test_loss]
    curr_time = time.time() - start_time

    if rank == 0:
        print(epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[0]["weight_decay"],
              lerx, lerz, ler_tot, train_loss, test_loss, curr_time)

        # Save best model based on total LER
        if ler_tot < min_test_err_rate:
            min_test_err_rate = ler_tot
            min_ler_tot = ler_tot
            save_model(model.module, fnamenew + f'gnn.pth {lerx}_{lerz}_{ler_tot} {epoch}', confirm=False)

        # Save best model based on LER_Z
        if ler_tot < min_ler_tot:
            min_ler_tot = ler_tot
            save_model(model.module, fnamenew + f'gnn.pth {lerx}_{lerz}_{ler_tot} {epoch}', confirm=False)

        # Save periodically
        if epoch % 10 == 0:
            np.save(fnamenew + 'training_lers_and_losses.npy', le_rates)
            #save_model(model.module, fnamenew + f'gnn.pth {lerx}_{lerz}_{ler_tot} {epoch}', confirm=False)
            torch.cuda.empty_cache()

        # Early stopping condition
        if ler_tot == 0:
            min_ler_tot = ler_tot
            save_model(model.module, fnamenew + f'gnn.pth {lerx}_{lerz}_{ler_tot} {epoch}', confirm=False)
            stop_training.fill_(1)
            print(f"[Early Stopping] ler_tot=0. Stopping at epoch {epoch}.")

        # Track best LER
        if ler_tot + 1e-6 < prev_best_ler:
            prev_best_ler = ler_tot
            no_improve_epochs = 0
        else:
            #no_improve_epochs = getattr(evaluate_and_save_model, "no_improve_epochs", 0) + 1
            #evaluate_and_save_model.no_improve_epochs = no_improve_epochs
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"[Early Stopping] No improvement in LER for {patience} epochs. Stopping at epoch {epoch}.")
            stop_training.fill_(1)

    # Sync stop signal across ranks
    dist.broadcast(stop_training, src=0)
    
    return stop_training.item() == 1
    #return min_test_err_rate, min_ler_tot, prev_best_ler, stop_training.item() == 1


def evaluate_model_only(model, code, testloader, test_sampler, device, rank, use_amp, epoch, optimizer, scheduler, train_loss, start_time, prefix=""):
    model.eval()
    model.to(device)
    test_sampler.set_epoch(epoch)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        test_loss = compute_accuracy(model.module, testloader, code)
        lerx, lerz, ler_tot = logical_error_rate(model.module, testloader, code)

        test_loss = sync_tensor(torch.tensor(test_loss, device=device)).item()
        lerx = sync_tensor(torch.tensor(lerx, device=device)).item()
        lerz = sync_tensor(torch.tensor(lerz, device=device)).item()
        ler_tot = sync_tensor(torch.tensor(ler_tot, device=device)).item()

    # Update learning rate schedule
    scheduler.step()
    curr_time = time.time() - start_time

    if rank == 0:
        print(epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[0]["weight_decay"],
            lerx, lerz, ler_tot, train_loss, test_loss, curr_time)

    return lerx, lerz, ler_tot, test_loss



# === Setup ===
set_seed(42 + int(os.environ['LOCAL_RANK']))
local_rank, rank, device = setup_ddp()

# === Hyperparameters ===
n_node_inputs = 4
n_node_outputs = 4
n_iters = 35
n_node_features = 64
n_edge_features = 32
msg_net_size = 128
msg_net_dropout_p = 0.1
gru_dropout_p = 0.1
heads = 4

lr_max = 0.0005
weight_decay = 0.0001
batch_size = 16

#start
s_train_data_start = 24000
s_len_test_set_start = 1200

#final
s_train_data_final = 24000
s_len_test_set_final = 1200


#data_generate
max_train_err_rate = 0.15
test_err_rate = 0.05  

#dart loop parameters
E_total = 90
E_prime = 20
E_m = 50
lambda_step = 10
ema_factor = 0.5

use_amp = True

criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=True)

# === Error Model ===
error_model_name = "DP"
if(error_model_name == "X"):
    error_model = [1, 0, 0]
elif (error_model_name == "Z"):
    error_model = [0, 0, 1]
elif (error_model_name == "XZ"):
    error_model = [0.5, 0, 0.5]
elif (error_model_name == "DP"):
    error_model = [1/3, 1/3, 1/3]

# === domain distance ===
distances = [10, 12, 18] #50000 80000, 100000
#weights_equal = [1/3, 1/3, 1/3]  # Must sum to 1.0
weights = [0.1, 0.2, 0.7]  # Must sum to 1.0

# === Data Preparation ===
#sarting model
c_dist_start = 6
code_start = bb_code(c_dist_start)
model_start = init_model(code_start, c_dist_start, n_node_features=64, n_edge_features=32, msg_net_size=128, n_iters=35)

#training data(total)
train_data_start = generate_syndrome_error_volume(code_start, error_model, p=max_train_err_rate, batch_size=s_train_data_start)
trainset_start = adapt_trainset(train_data_start, code_start, num_classes=n_node_inputs)
train_sampler_start = DistributedSampler(trainset_start)
trainloader_start = DataLoader(trainset_start, batch_size=batch_size, sampler=train_sampler_start,
                            collate_fn=collate, pin_memory=True, num_workers=4)

#testing data
test_data_start = generate_syndrome_error_volume(code_start, error_model, p=test_err_rate, batch_size=s_len_test_set_start, for_training=False)
testset_start = adapt_trainset(test_data_start, code_start, num_classes=n_node_inputs, for_training=False)
test_sampler_start = DistributedSampler(testset_start)
testloader_start = DataLoader(testset_start, batch_size=512, sampler=test_sampler_start, 
                              collate_fn=collate, pin_memory=True, num_workers=4)




#domain
model_configs = [
    {"c_dist": 10, "n_node_features": 64, "n_edge_features": 32, "msg_net_size": 128, "n_iters": 40, "sub_len_train_set": 6000, "test_len": 300},
    {"c_dist": 12, "n_node_features": 64, "n_edge_features": 32, "msg_net_size": 128, "n_iters": 50, "sub_len_train_set": 8000, "test_len": 400},
    {"c_dist": 18, "n_node_features": 64, "n_edge_features": 32, "msg_net_size": 128, "n_iters": 65, "sub_len_train_set": 10000, "test_len": 500}
]

models = []
domain_datasets = {}
for config in model_configs:
    c_dist = config["c_dist"]
    code = bb_code(c_dist)
    model = init_model(
        code=code,
        c_dist=c_dist,
        n_node_features=config["n_node_features"],
        n_edge_features=config["n_edge_features"],
        msg_net_size=config["msg_net_size"],
        n_iters=config["n_iters"]
    )
    models.append((c_dist, code, model))
    
    # Training
    sub_len = config["sub_len_train_set"]
    train_data = generate_syndrome_error_volume(code, error_model, p=max_train_err_rate, batch_size=sub_len)
    trainset = adapt_trainset(train_data, code, num_classes=n_node_inputs)
    train_sampler = DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate, pin_memory=True, num_workers=4)

    # Testing
    test_data = generate_syndrome_error_volume(code, error_model, p=test_err_rate, batch_size=config["test_len"], for_training=False)
    testset = adapt_trainset(test_data, code, num_classes=n_node_inputs, for_training=False)
    test_sampler = DistributedSampler(testset)
    testloader = DataLoader(testset, batch_size=512, sampler=test_sampler, collate_fn=collate, pin_memory=True, num_workers=4)

    domain_datasets[c_dist] = {
        "code": code,
        "trainloader": trainloader,
        "train_sampler": train_sampler,
        "testloader": testloader,
        "test_sampler": test_sampler,
    }
#sub_len_train_set = s_train_data_start // len(distances)



#final_model
c_dist_final = 18
code_final = bb_code(c_dist_final)
model_final = init_model(code_final, c_dist_final, n_node_features=64, n_edge_features=32, msg_net_size=128, n_iters=35)

#final-model: total training data
train_data_final = generate_syndrome_error_volume(code_final, error_model, p=max_train_err_rate, batch_size=s_train_data_final)
trainset_final = adapt_trainset(train_data_final, code_final, num_classes=n_node_inputs)
train_sampler_final = DistributedSampler(trainset_final)
trainloader_final = DataLoader(trainset_final, batch_size=batch_size, sampler=train_sampler_final,
                            collate_fn=collate, pin_memory=True, num_workers=4)
#final-model: testing data
test_data = generate_syndrome_error_volume(code_final, error_model, p=test_err_rate, batch_size=s_len_test_set_final, for_training=False)
testset = adapt_trainset(test_data, code_final, num_classes=n_node_inputs, for_training=False)
test_sampler = DistributedSampler(testset)
testloader = DataLoader(testset, batch_size=512, sampler=test_sampler, collate_fn=collate, pin_memory=True, num_workers=4)



fnamenew = f"bnn_testing/dart_{error_model_name}_{n_iters}_{n_node_features}_{n_edge_features}_" \
           f"{s_train_data_start}_{max_train_err_rate}_{s_len_test_set_start}_{test_err_rate}_{lr_max}_" \
           f"{weight_decay}_{msg_net_size}_{msg_net_dropout_p}_{gru_dropout_p}_"



min_test_err_rate = test_err_rate
min_ler_tot = test_err_rate

start_time = time.time()
le_rates = np.zeros((E_total,5), dtype='float')


kl_annealing_epochs = 10  # warm-up period
kl_scale = 1e-5
patience = 20


if rank == 0:
    print("===== Training Configuration Summary =====")
    print(f"  Error Model         : {error_model_name}")
    print(f"  Initial Train Dist  : {c_dist_start} (code: {code_start.name})")
    print(f"  Final Trai&Tst Dist : {c_dist_final} (code: {code_final.name})")
    print(f"  Domain Distances    : {distances}")
    print(f"  weights             : {weights}")
    print()
    print("===== Starting Model =====")
    print(f"  n_node_inputs       : {n_node_inputs}")
    print(f"  n_node_outputs      : {n_node_outputs}")
    print(f"  n_node_features     : {n_node_features}")
    print(f"  n_edge_features     : {n_edge_features}")
    print(f"  msg_net_size        : {msg_net_size}")
    print(f"  msg_net_dropout_p   : {msg_net_dropout_p}")
    print(f"  gru_dropout_p       : {gru_dropout_p}")
    print(f"  n_iters             : {n_iters}")
    print(f"  heads               : {heads}")
    print()
    print("===== Optimization =====")
    print(f"  max_lr              : {lr_max}")
    print(f"  weight_decay        : {weight_decay}")
    print(f"  batch_size          : {batch_size}")
    print()
    print("===== Training Setup =====")
    print(f"  Total Epochs        : {E_total}")
    print(f"  Warmup Epochs (E')  : {E_prime}")
    print(f"  Middle Epochs (E_m) : {E_m}")
    print(f"  Lambda Step         : {lambda_step}")
    print(f"  Ema_decay           : {ema_factor}")
    print(f"  Early Stop Patience : {patience}")
    print()
    print("===== Dataset Info =====")
    print(f"  Train Set Size      : {s_train_data_start}")
    print(f"  Test Set Size       : {s_len_test_set_start}")
    print(f"  f_Train Set Size    : {s_train_data_final}")
    print(f"  f_Test Set Size     : {s_len_test_set_final}")
    print(f"  Train Error Rate    : {max_train_err_rate}")
    print(f"  Test Error Rate     : {test_err_rate}")
    print("==========================================\n")


if rank == 0:
    print("epoch, lr, wd, LER_X, LER_Z, LER_tot, train loss, test loss, train time")


no_improve_epochs = 0
prev_best_ler = float('inf')
stop_training = torch.tensor([0], device=device)  # 0: continue, 1: stop


#start
optimizer_start = optim.AdamW(model_start.parameters(), lr=lr_max, weight_decay=weight_decay)
scheduler_start = optim.lr_scheduler.StepLR(optimizer_start, step_size=int(E_prime * 2 / 3), gamma=0.5)#E_prime *2/3 #40
scaler_start = torch.cuda.amp.GradScaler(enabled=True)

#domain
optimizers = {}
schedulers = {}
scalers = {}

for c_dist, code, model in models:
    # Initialize per-model optimizer/scheduler/scaler
    optimizer = optim.AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int((E_m-E_prime)* 2/3), gamma=0.5)#(E_m-E_prime)* 2/3
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Store in dictionaries keyed by c_dist (or model ID)
    optimizers[c_dist] = optimizer
    schedulers[c_dist] = scheduler
    scalers[c_dist] = scaler


lr_final = 0.0001
#final
optimizer_final = optim.AdamW(model_final.parameters(), lr=lr_final, weight_decay=weight_decay)
scheduler_final = optim.lr_scheduler.StepLR(optimizer_final, step_size=10, gamma=0.5)#(E_total-E_m)*2/3
scaler_final = torch.cuda.amp.GradScaler(enabled=True)
#lr=0.0005
#lr_final=0.0005

for epoch in range(E_total):
    #beta = min(1.0, epoch / kl_annealing_epochs) * kl_scale
    #lr = 0.5 * lr_max * (1 + math.cos((epoch / E_total) * math.pi))
    #lr = lr_max

    #for param_group in optimizer.param_groups:
    #    param_group["lr"] = lr

    train_sampler_start.set_epoch(epoch)
    train_sampler_final.set_epoch(epoch)
    
  
    # === Warm-up Phase (ERM on individual domain D_i) ===
    if epoch < E_prime:
        if rank == 0 and epoch == 0:
            print("***[Epoch <= E_prime Phase]***")
        #avg_loss, optimizer, scheduler, scaler, model_start = train_epoch_warmup(
        train_loss, optimizer, model_start = train_epoch_warmup(
            model=model_start,
            code=code_start,
            trainloader=trainloader_start,
            criterion=criterion,
            optimizer=optimizer_start,
            #scheduler=scheduler_start,
            scaler=scaler_start,
            #beta=beta,
            #lr=lr_final,
            use_amp=use_amp,
            device=device
        )
        #scheduler_start.step()
        
        evaluate_model_only(
            model=model_start, 
            code=code_start, 
            testloader=testloader_start, 
            test_sampler=test_sampler_start, 
            device=device, 
            rank=rank, 
            use_amp=use_amp, 
            epoch=epoch, 
            optimizer=optimizer, 
            scheduler=scheduler_start, 
            train_loss=train_loss, 
            start_time=start_time, 
            prefix="Start"
        )
                
        #if rank == 0:
        #    print("[epoch <= E_prime] Epoch:", epoch, "Loss:", avg_loss)
        
        if epoch == E_prime-1:
            if rank == 0:
                print("***[Epoch = E_prime Phase]*** Epoch:", epoch)
            
            for c_dist, code, model in models:
                model.load_state_dict(model_start.state_dict())
    
    elif E_prime <= epoch < E_m:
        if rank == 0 and epoch == E_prime:
            print("***[E_prime < epoch <= E_m Phase]*** Epoch:", epoch)
            
        #accu_domain_loss = 0.0
        for c_dist, code, model in models:
            optimizer = optimizers[c_dist]
            scheduler = schedulers[c_dist]
            scaler = scalers[c_dist]
            
            train_info = domain_datasets[c_dist]
            trainloader = train_info["trainloader"]
            train_sampler = train_info["train_sampler"]
            
            train_sampler.set_epoch(epoch)
            
            testloader = train_info["testloader"]
            test_sampler = train_info["test_sampler"]
            
            train_loss, optimizer, model = train_domain_model(
                model=model,
                code=code,
                c_dist=c_dist,
                trainloader=trainloader,
                criterion=criterion,
                #train_sampler=train_sampler,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch
            )
            _, _, ler_tot, _ = evaluate_model_only(
                model=model,
                code=code,
                testloader=testloader,
                test_sampler=test_sampler,
                device=device,
                rank=rank,
                use_amp=use_amp,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=train_loss,
                start_time=start_time,
                prefix=f"Domain-{c_dist}"
            )
        if rank == 0:    
            print("=" * 2)
            
        #20 50(10)   #10 40(4)
        if (epoch + 1 - E_prime) % lambda_step == 0 or epoch == E_m-1:
            if epoch == E_m-1:
                if rank == 0:
                    print("****Epoch == E_m Phase]**** Epoch:", epoch)

            avg_model = average_model_weights_weighted(models, weights)

            
            if (epoch + 1 - E_prime) % lambda_step == 0:
                if rank == 0:
                    print("****[(epoch + 1 - E_prime)\lambda_step Phase]**** Epoch:", epoch)
                for c_dist, _, model in models:
                        model.load_state_dict(avg_model.state_dict())
      
    # === After Warm-up: Copy averaged model to all domain models and to model_start ===
    else:
        model_final.load_state_dict(avg_model.state_dict())
        model_final.to(device)
        #avg_model.to(device)

        train_loss, optimizer, model_train = train_epoch_warmup(
            model=model_final,
            code=code_final,
            trainloader=trainloader_final,
            criterion=criterion,
            optimizer=optimizer_final,
            #scheduler=scheduler_final,
            scaler=scaler_final,
            #beta=beta,
            #lr=lr_final,
            use_amp=use_amp,
            device=device
        )

        # Evaluate model_start at transition epoch
        stop_flag = False
        #min_test_err_rate, min_ler_tot, prev_best_ler, stop_flag = evaluate_and_save_model(
        stop_flag = evaluate_and_save_model(
            epoch=epoch,
            model=model_train,
            code=code_final,
            testloader=testloader,
            test_sampler=test_sampler,
            optimizer=optimizer,
            scheduler=scheduler_final,
            le_rates=le_rates,
            start_time=start_time,
            train_loss=train_loss,
            fnamenew=fnamenew,
            device=device,
            rank=rank,
            stop_training=stop_training,
            prev_best_ler=prev_best_ler,
            no_improve_epochs=no_improve_epochs,
            min_test_err_rate=min_test_err_rate,
            min_ler_tot=min_ler_tot,
            patience=patience
        )
        if stop_flag:
            break
           
      
dist.destroy_process_group()