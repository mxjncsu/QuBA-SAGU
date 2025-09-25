"""
You need a virtual environment (clean one)
pip uninstall panqec
pip uninstall ldpc

Afterwards

git clone -b public_osd https://github.com/alexandrupaler/ldpc.git
pip install -e ldpc

pip install panqec
"""

import ldpc
import os
from panqec.codes import surface_2d
from panqec.error_models import PauliErrorModel
from panqec.decoders import MatchingDecoder, BeliefPropagationOSDDecoder
from ldpc import BpOsdDecoder, BpDecoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import sympy
import pickle
import hashlib

from functions import QuBA, collate,compute_accuracy, logical_error_rate, \
    construct_tanner_graph_edges, generate_syndrome_error_volume, adapt_trainset,ler_loss,bb_code,load_model,copbb_code

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
    
def init_log_probs_of_decoder(decoder, my_log_probs):
    #print("old ", decoder.log_prob_ratios)

    for i in range(len(decoder.log_prob_ratios)):
        decoder.set_log_prob(i, my_log_probs[i])

    #print("new ", decoder.log_prob_ratios)
def md5_hash_array(arr: np.ndarray) -> str:
    arr_bytes = arr.tobytes()
    return hashlib.md5(arr_bytes).hexdigest()    

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    use_amp = True  #to use automatic mixed precision
    amp_data_type = torch.float16
else:
    device = torch.device('cpu')
    use_amp = False
    '''float16 is not supported for cpu use bfloat16 instead'''
    amp_data_type = torch.bfloat16

set_seed(42)


error_model_name = "DP"
if(error_model_name == "X"):
    error_model = [1, 0, 0]
elif (error_model_name == "Z"):
    error_model = [0, 0, 1]
elif (error_model_name == "XZ"):
    error_model = [0.5, 0, 0.5]
elif (error_model_name == "DP"):
    error_model = [1/3, 1/3, 1/3]

# size = 2 * d ** 2 - 1
n_node_inputs = 4
n_node_outputs = 4
n_iters= 100
n_node_features=64
n_edge_features=32

msg_net_size = 128
msg_net_dropout_p = 0.1
gru_dropout_p = 0.1

#enable_osd = False
enable_osd=True

d = dist = 16

print("n_iters: ", n_iters, "n_node_outputs: ", n_node_outputs, "n_node_features: ", n_node_features,"n_edge_features: ", n_edge_features ,"msg_net_size:", msg_net_size, "enable_osd",enable_osd)



#fname_ours_d6    = "bnn_testing/ours_final/BB_n72_k12_d6_DP_35_64_32_20000_0.15_2000_0.05_0.0005_0.0001_128_0.1_0.1_"
#fname_ours_d10   = "bnn_testing/ours_final/BB_n90_k8_d10_DP_40_64_32_50000_0.15_3000_0.05_0.0005_0.0001_256_0.1_0.1_"
#fname_ours_d12   = "bnn_testing/ours_final/BB_n144_k12_d12_DP_50_32_32_80000_0.15_4000_0.05_0.0005_0.0001_256_0.1_0.1_"
#fname_ours_d18   = "bnn_testing/ours_final/BB_n288_k12_d18_DP_65_64_32_100000_0.15_5000_0.05_0.0005_0.0001_128_0.1_0.1_"
#fname_ours_d24   = "bnn_testing/BB_n360_k12_d24_DP_80_64_32_150000_0.15_6000_0.05_0.0005_0.0001_128_0.1_0.1_"
#fname_ours_d34   = "bnn_testing/ours_final/BB_n756_k16_d34_DP_50_64_32_50000_0.15_3000_0.05_0.0005_0.0001_128_0.1_0.1_"

#fname_ours_d6_copBB   = "bnn_testing/ours_final/copBB_n30_k4_d6_DP_40_32_32_30000_0.15_1000_0.05_0.0005_0.0001_256_0.1_0.1_"
#fname_ours_d16_copBB  = "bnn_testing/copbb/copBB_n154_k6_d16_DP_60_64_32_100000_0.15_5000_0.05_0.0005_0.0001_128_0.1_0.1_"


#fname_ours_d18_dart  = "bnn_testing/dart_DP_35_64_32_24000_0.15_1200_0.05_0.0005_0.0001_128_0.1_0.1_"

#fname_ours_d6_sagu   = "bnn_testing/ours_final/BB_n72_k12_d6_DP_35_64_32_24000_0.15_1200_0.05_0.0005_0.0001_128_0.1_0.1_"
#fname_ours_d12_sagu   = "bnn_testing/ours_final/BB_n144_k12_d12_DP_50_64_32_24000_0.15_1200_0.05_0.0005_0.0001_128_0.1_0.1_"
fname_ours_d18_sagu   = "bnn_testing/ours_final/BB_n288_k12_d18_DP_50_64_32_24000_0.15_1200_0.05_0.0001_0.0001_128_0.1_0.1_"
#fname_ours_d34_sagu   = "bnn_testing/ours_final/BB_n756_k16_d34_DP_50_64_32_24000_0.15_1200_0.05_0.0005_0.0001_128_0.1_0.1_"




#code = bb_code(dist)
code = copbb_code(dist)
print('trained', 16, '\t test', dist, "\tcode name :",code.name)
# code = surface_2d.RotatedPlanar2DCode(dist)
gnn = QuBA(dist=dist, n_node_inputs=n_node_inputs, n_node_outputs=n_node_outputs, n_iters=n_iters,
                 n_node_features=n_node_features, n_edge_features=n_edge_features,
                 msg_net_size=msg_net_size, msg_net_dropout_p=msg_net_dropout_p, gru_dropout_p=gru_dropout_p)
gnn.to(device)

src, tgt = construct_tanner_graph_edges(code)
src_tensor = torch.LongTensor(src)
tgt_tensor = torch.LongTensor(tgt)
QuBA.construct_tanner_graph_edges = (src_tensor, tgt_tensor)

hxperp = torch.FloatTensor(code.hx_perp).to(device)
hzperp = torch.FloatTensor(code.hz_perp).to(device)
QuBA.hxperp = hxperp
QuBA.hzperp = hzperp

QuBA.device = device


#load_model(gnn, fname_ours_d6  + "gnn.pth 0.01999000459909439_0.005497251637279987_0.01999000459909439 43", device)
#load_model(gnn, fname_ours_d10 + "gnn.pth 0.004000000189989805_0.005333333276212215_0.005666667129844427 29", device)
#load_model(gnn, fname_ours_d12 + "gnn.pth 0.004997501149773598_0.005497251637279987_0.005747126415371895 18", device)
#load_model(gnn, fname_ours_d18 + "gnn.pth 0.0_0.0_0.0 12", device)
#load_model(gnn, fname_ours_d24 + "gnn.pth 0.0_0.0_0.0 13", device)
#load_model(gnn, fname_ours_d34 + "gnn.pth 0.0_0.0_0.0 28", device)


#load_model(gnn, fname_ours_d6_copBB  + "gnn.pth 0.0359281450510025_0.045908182859420776_0.049900200217962265 30", device)
#load_model(gnn, fname_ours_d16_copBB + "gnn.pth 0.0007998400251381099_0.0009998000459745526_0.0009998000459745526 36", device)

#load_model(gnn, fname_ours_d18_dart + "gnn.pth 0.0_0.0_0.0 52", device)


#load_model(gnn, fname_ours_d6_sagu  + "gnn.pth 0.02500000223517418_0.022499999031424522_0.02666666731238365 22", device)
#load_model(gnn, fname_ours_d12_sagu  + "gnn.pth 0.006666666828095913_0.007499999832361937_0.007499999832361937 23", device)
load_model(gnn, fname_ours_d18_sagu  + "gnn.pth 0.00416666641831398_0.007500000298023224_0.007500000298023224 85", device)
#load_model(gnn, fname_ours_d34_sagu  + "gnn.pth 0.0_0.0_0.0 65", device)



#0.0005 0.00016
#err_rates = np.array([0.001, 0.0007, 0.0004, 0.0002])
# err_rates = np.array([0.14,0.06,0.05,0.04,0.03])
#err_rates = np.array([0.2,0.18,0.16,0.14,0.12,0.1,0.08,0.06])
# err_rates = np.array([0.2,0.18,0.16,0.14])
# err_rates = np.array([0.12,0.1,0.08,0.06])
# err_rates = np.array([0.06])
#err_rates = np.array([0.0001, 0.0004, 0.0006, 0.0008, 0.0009, 0.001])
err_rates = np.array([0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2])
#err_rates = np.array([0.04,0.06,0.08,0.1,0.12])
#err_rates = np.array([0.08,0.12,0.16,0.2])
# err_rates = np.array([0.2,0.18,0.16,0.14,0.12,0.1,0.08,0.06,0.05,0.04])
nruns=len(err_rates)

le_rates = np.zeros((nruns,8),dtype='float')
len_test_set_org = 10**3
print(f"test set size = {len_test_set_org}")
#print("i \tp \tlerx_mean \tlerz_mean \tlertot_mean \tlerx_std \tlerz_std \tlertot_std \tL_tot \ttime_taken")
#print("i \terr_rate \tlerx_mean±2σ \tlerz_mean±2σ \tlertot_mean±2σ \tlerx_95CI \tlerz_95CI \tlertot_95CI \ttest_loss \ttime_taken")
print("i \terr_rate \tlertot_mean±2σ")

#print(i, err_rate, lerx_mean, lerz_mean, lertot_mean, lerx_std, lerz_std, lertot_std, test_loss, np.round(t2-t1,2))

n_mc_samples = 30

gnn.eval()
with torch.no_grad():
    for i in range(nruns):
        st = time.time()
        err_rate = err_rates[i]
        len_test_set = int(len_test_set_org)

        syndrome_error_volume = generate_syndrome_error_volume(code, error_model, p=err_rate, batch_size=len_test_set, for_training=False)
        print(f"Test MD5: {md5_hash_array(syndrome_error_volume)}")
    
        testset = adapt_trainset(syndrome_error_volume, code, num_classes=n_node_inputs, for_training=False)
        testloader = DataLoader(testset, batch_size=128, collate_fn=collate, shuffle=False)
        
        t1 = time.time()
        # print("data generation",t1 - st)
        x_decoder = BpOsdDecoder(code.hz, error_rate=float(err_rate), max_iter=100, schedule = 'serial', bp_method="ms", ms_scaling_factor=0.5, osd_method="osd0")
        z_decoder = BpOsdDecoder(code.hx, error_rate=float(err_rate), max_iter=100, schedule = 'serial', bp_method="ms", ms_scaling_factor=0.5, osd_method="osd0")
        osd_decoder = (x_decoder,z_decoder)
        
        with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
            lerx_list = []
            lerz_list = []
            lertot_list = []

            for _ in range(n_mc_samples):
                lerx_i, lerz_i, ler_tot_i = logical_error_rate(
                    gnn, testloader, code,
                    osd_decoder, enable_osd=enable_osd,
                    n_iters=gnn.n_iters
                )
                lerx_list.append(lerx_i)
                lerz_list.append(lerz_i)
                lertot_list.append(ler_tot_i)

            # Convert to numpy for easy stats
            lerx_array = np.array(lerx_list)
            lerz_array = np.array(lerz_list)
            lertot_array = np.array(lertot_list)

            # Mean and standard deviation
            lerx_mean, lerx_std = lerx_array.mean(), lerx_array.std()
            lerz_mean, lerz_std = lerz_array.mean(), lerz_array.std()
            lertot_mean, lertot_std = lertot_array.mean(), lertot_array.std()
            
            
            # ----- 1. Spread of distribution (±2σ) -----
            lerx_lo_spread = lerx_mean - 2 * lerx_std
            lerx_hi_spread = lerx_mean + 2 * lerx_std

            lerz_lo_spread = lerz_mean - 2 * lerz_std
            lerz_hi_spread = lerz_mean + 2 * lerz_std

            lertot_lo_spread = lertot_mean - 2 * lertot_std
            lertot_hi_spread = lertot_mean + 2 * lertot_std

            # ----- 2. 95% CI of mean (±1.96 * SE) -----
            ci_factor = 1.96 / np.sqrt(n_mc_samples)

            lerx_lo_ci = lerx_mean - ci_factor * lerx_std
            lerx_hi_ci = lerx_mean + ci_factor * lerx_std

            lerz_lo_ci = lerz_mean - ci_factor * lerz_std
            lerz_hi_ci = lerz_mean + ci_factor * lerz_std

            lertot_lo_ci = lertot_mean - ci_factor * lertot_std
            lertot_hi_ci = lertot_mean + ci_factor * lertot_std

        t2 = time.time()
        test_loss = 0  # compute_accuracy(gnn, testloader, code)
        # t4= time.time()
        # print("test loss",t4-t3)
        le_rates[i, 0] = err_rate
        le_rates[i, 1] = lerx_mean
        le_rates[i, 2] = lerz_mean
        le_rates[i, 3] = lertot_lo_spread
        le_rates[i, 4] = lertot_hi_spread
        le_rates[i, 5] = lertot_lo_ci
        le_rates[i, 6] = lertot_hi_ci

        #le_rates[i, 3] = lertot_mean
        #le_rates[i, 5] = lerx_std
        #le_rates[i, 6] = lerz_std
        #le_rates[i, 7] = lertot_std
        le_rates[i, 7] = t2 - t1
        print(i, err_rate, lerx_mean, lerz_mean, lertot_mean, lerx_std, lerz_std, lertot_std, test_loss, np.round(t2-t1,2))
        #print(i, err_rate, lerx_mean, lerz_mean, lertot_mean, lerx_std, lerz_std, lertot_std, test_loss, np.round(t2-t1,2))

        #print(f"{i} \t{err_rate:.5f} \t"
                #f"{lertot_mean:.5f}±{2 * lertot_std:.5f} \t"
                #)
