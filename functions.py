from panqec.codes import surface_2d
from panqec.error_models import PauliErrorModel
from panqec.decoders import BeliefPropagationOSDDecoder, MatchingDecoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import sympy
import os
from codes_q import *
from torch_scatter import scatter_softmax, scatter_add, scatter_max

#Some codes forked from https://github.com/arshpreetmaan/astra/tree/main
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Prior mean = 0, prior_std=1.0
        self.prior_std = prior_std
        
        # Posterior - weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-6)) #log(variance) of posterior
        #self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).fill_(-5))  # Lower std

        # Posterior - bias
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features).fill_(-6))

    def forward(self, input):
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)

        weight_eps = torch.randn_like(weight_std)
        bias_eps = torch.randn_like(bias_std)

        weight = self.weight_mu + weight_std * weight_eps
        bias = self.bias_mu + bias_std * bias_eps

        return torch.nn.functional.linear(input, weight, bias)

    def kl_divergence(self):
        kl = 0.5 * torch.sum(# weights
            torch.exp(self.weight_logvar) / self.prior_std**2 +
            (self.weight_mu**2) / self.prior_std**2 -
            1 - self.weight_logvar
        )
        kl += 0.5 * torch.sum(# biases
            torch.exp(self.bias_logvar) / self.prior_std**2 +
            (self.bias_mu**2) / self.prior_std**2 -
            1 - self.bias_logvar
        )
        return kl


class QuBA(nn.Module):
    dist = None
    construct_tanner_graph_edges = None
    device = None
    hxperp = None
    hzperp = None

    def __init__(self, dist=3, n_iters=7, n_node_features=10, n_node_inputs=9, n_edge_features=11,
                 n_node_outputs=9, msg_net_size=96, msg_net_dropout_p=0.0, gru_dropout_p=0.0, heads=4):
        super(QuBA, self).__init__()

        QuBA.dist = dist
        
        self.n_iters = n_iters
        self.n_node_features = n_node_features
        self.n_node_inputs = n_node_inputs
        self.n_edge_features = n_edge_features
        self.n_node_outputs = n_node_outputs
        self.heads = heads
        self.head_dim = n_edge_features // heads
        self.total_edge_features = self.heads * self.head_dim
        self.initial_node_embedding = nn.Parameter(torch.randn(1, n_node_features))
        self.final_digits = BayesianLinear(self.n_node_features, self.n_node_outputs)

        # Q/K projections for attention
        self.q_proj = BayesianLinear(n_node_features, self.total_edge_features)
        self.k_proj = BayesianLinear(n_node_features, self.total_edge_features)

        self.q_norm = nn.BatchNorm1d(self.total_edge_features)
        self.k_norm = nn.BatchNorm1d(self.total_edge_features)

        # Learnable temperature parameter for attention scaling
        self.temperature = nn.Parameter(torch.tensor(2.0))
        self.dropout = nn.Dropout(msg_net_dropout_p)
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Message Network
        self.msg_net = nn.Sequential(
            BayesianLinear(2 * n_node_features, msg_net_size),
            nn.BatchNorm1d(msg_net_size),
            nn.ReLU(),
            nn.Dropout(msg_net_dropout_p),
            BayesianLinear(msg_net_size, msg_net_size),
            nn.BatchNorm1d(msg_net_size),
            nn.ReLU(),
            nn.Dropout(msg_net_dropout_p),
            BayesianLinear(msg_net_size, msg_net_size),
            nn.BatchNorm1d(msg_net_size),
            nn.ReLU(),
            nn.Dropout(msg_net_dropout_p),
            BayesianLinear(msg_net_size, self.total_edge_features)
        )

        self.msg_norm = nn.BatchNorm1d(self.total_edge_features)

        # LSTM
        self.lstm = nn.LSTM(input_size=self.total_edge_features + n_node_inputs, hidden_size=n_node_features)

        self.gru_drop = nn.Dropout(gru_dropout_p)

    def forward(self, node_inputs, src_ids, dst_ids):
        device = node_inputs.device
        node_states = self.initial_node_embedding.expand(node_inputs.shape[0], -1)
        outputs_tensor = torch.zeros(self.n_iters, node_inputs.shape[0], self.n_node_outputs, device=device)

        for i in range(self.n_iters):
            # Project Q and K, apply normalization
            Q = self.q_norm(self.q_proj(node_states[src_ids])).view(-1, self.heads, self.head_dim)
            K = self.k_norm(self.k_proj(node_states[dst_ids])).view(-1, self.heads, self.head_dim)

            # Scaled dot-product attention
            temp = torch.clamp(self.temperature, min=0.5, max=5.0)
            attn_score = (Q * K).sum(dim=-1) / temp
            attn_score = self.leaky_relu(attn_score)

            # Normalize attention scores
            attn_score = attn_score - scatter_max(attn_score, dst_ids, dim=0)[0][dst_ids]
            attn_weight = scatter_softmax(attn_score, dst_ids, dim=0)
            attn_weight = self.dropout(attn_weight)

            # Concatenate node states and compute message
            msg_in = torch.cat((node_states[src_ids], node_states[dst_ids]), dim=1)
            raw_msg = self.msg_net(msg_in).view(-1, self.heads, self.head_dim)
            messages = (attn_weight.unsqueeze(-1) * raw_msg).view(-1, self.total_edge_features)
            messages = self.msg_norm(messages) 

            # Aggregate messages at destination nodes
            agg_msg = torch.zeros(node_inputs.shape[0], self.total_edge_features, device=device, dtype=messages.dtype)
            agg_msg.index_add_(dim=0, index=dst_ids, source=messages)

            gru_inputs = torch.cat((agg_msg, node_inputs), dim=1)
            
            
            # Update hidden state
            output, (node_states_new, _) = self.lstm(gru_inputs.view(1, node_inputs.shape[0], -1),
                                                        (node_states.view(1, node_inputs.shape[0], -1),
                                                        torch.zeros_like(node_states.view(1, node_inputs.shape[0], -1))))
            # Residual Connection + Dropout
            node_states = self.gru_drop(node_states_new.squeeze(0)) + node_states

            # Project to output
            outputs_tensor[i] = self.final_digits(node_states)
            node_states = self.gru_drop(node_states) 

        return outputs_tensor

    #variational Bayesian inference
    def kl_divergence(self):
        kl = self.final_digits.kl_divergence()
        kl += self.q_proj.kl_divergence()
        kl += self.k_proj.kl_divergence()
        for layer in self.msg_net:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl_divergence()
        return kl
    
def collate(list_of_samples):
    """Merges a list of samples to form a mini-batch.

    Args:
      list_of_samples is a list of tuples (inputs, targets),
          inputs of shape (n_nodes, n_node_inputs): Inputs to each node in the graph. Inputs are one-hot coded digits.
          A missing digit is encoded with all zeros. n_nodes= nodes in the tanner graph
          targets of shape (n_nodes): A LongTensor of targets (correct digits of tanner graph).

    Returns:
      inputs of shape (batch_size*n_nodes, n_node_inputs): Inputs to each node in the graph. Inputs are one-hot coded digits
        for syndromes/errors. A missing digit is encoded with all zeros.
      targets of shape (batch_size*n_nodes): A LongTensor of targets (correct digits of tanner graph).
      src_ids of shape (batch_size*nodes in the tanner graph): LongTensor of source node ids for each edge in the large graph.
      dst_ids of shape (batch_size*nodes in the tanner graph): LongTensor of destination node ids for each edge in the large graph.
    """
    # YOUR CODE HERE
    (inp, target) = list_of_samples[0]
    if QuBA.construct_tanner_graph_edges is None:
        raise Exception
    og_src_ids, og_tgt_ids = QuBA.construct_tanner_graph_edges

    all_inputs = inp.clone().detach()
    all_targets = target.clone().detach()
    all_src_ids = og_src_ids.clone().detach()
    all_dst_ids = og_tgt_ids.clone().detach()

    if QuBA.dist is None:
        raise Exception
    # add = 2 * (QuBA.dist) ** 2
    add = 2 * (QuBA.hxperp.shape[1])


    for (inp, target) in list_of_samples[1:]:
        og_src_ids = torch.add(og_src_ids, add)
        og_tgt_ids = torch.add(og_tgt_ids, add)
        all_inputs = torch.cat((all_inputs, inp))
        all_targets = torch.cat((all_targets, target))
        all_src_ids = torch.cat((all_src_ids, og_src_ids))
        all_dst_ids = torch.cat((all_dst_ids, og_tgt_ids))

    return all_inputs, torch.LongTensor(all_targets), torch.LongTensor(all_src_ids), torch.LongTensor(all_dst_ids)



def plot_code(code):
    qcord = code.qubit_coordinates
    scord = code.stabilizer_coordinates
    x1, y1 = zip(*qcord)
    d = plt.scatter(x1, y1, color="k")
    ztype = []
    xtype = []
    for i in scord:
        if code.stabilizer_type(i) == "vertex":
            ztype.append(i)
        else:
            xtype.append(i)

    x2, y2 = zip(*ztype)
    z = plt.scatter(x2, y2, color="g")
    x3, y3 = zip(*xtype)
    x = plt.scatter(x3, y3, color="r")
    plt.legend((d, z, x), ("data", "Z type", "X type"))
    # plt.savefig("d5_panqec.png")
    # plt.show()


def construct_tanner_graph_edges(code):
    # graph=[(i, j) for i, j in zip(*code.stabilizer_matrix.nonzero())]
    # src_ids, dst_ids = code.stabilizer_matrix.nonzero()  # syndrome,data qubit
    s = np.zeros((code.hx.shape[0]*2,code.hx.shape[1]*2),dtype='int64')
    s[code.N // 2:, :code.N] = code.hx
    s[:code.N // 2, code.N:] = code.hz
    src_ids, dst_ids = s.nonzero()  # syndrome,data qubit
    # z first = detect x , using hz
    # src_idsx, dst_idsx = code.hx.nonzero()
    # src_idsz, dst_idsz = code.hz.nonzero()
    l = int(len(dst_ids) / 2)
    # dst_ids = dst_ids - 1
    dst_ids[l:] = dst_ids[l:] + code.N

    # for only Z stab = detect X
    # dst_ids = dst_ids[:l]
    # dst_ids = np.append(dst_ids, src_ids[l:])

    temp = src_ids
    src_ids = np.append(src_ids, dst_ids)
    dst_ids = np.append(dst_ids, temp)

    # G = nx.Graph()
    G = nx.DiGraph()
    for (s, t) in zip(src_ids, dst_ids):
        G.add_edge(s, t)

    # color_map = ['red' if node < code.N else 'green' for node in G]
    # nx.draw(G, node_color=color_map, with_labels=True)
    # # plt.savefig("trained_models/d3_panqec.png")
    # plt.show()
    return src_ids, dst_ids


def generate_syndrome_error_volume(code, error_model, p, batch_size, for_training=True):
    d = code.D
    size = 2 * code.N
    syndrome_error_volume = np.zeros((batch_size, size), dtype='uint8')
    starttime = time.time()
    # bpdec = decoder(code, error_model, error_rate=0.1, osd_order=0)
    # decoder = MatchingDecoder
    # mwpm = decoder(code, error_model, error_rate=p)
    if not for_training:
        px, py, pz = p * error_model[0] * np.ones(code.N), p * error_model[1] * np.ones(code.N), p * error_model[2] * np.ones(code.N)
        # px, py, pz = p / 3 * np.ones(code.N), p / 3 * np.ones(code.N), p / 3 * np.ones(code.N)
        noise = np.random.uniform(0, 1, (batch_size, code.N))
        err_z = np.logical_and(noise > px, noise < (px + py + pz))
        syndrome_x = (err_z @ code.hx.T) % 2  # [num_shots, N_half]
        err_x = noise < (px + py)
        syndrome_z = (err_x @ code.hz.T) % 2  # [num_shots, N_half]

        # error = np.zeros((batch_size, size), dtype='uint8')
        # for i in range(batch_size):
        #     error[i] = error_model.generate(code, p)
        # syndrome = code.measure_syndrome(error).T

        errorxz= (err_x + 2*err_z) #% 3
        # syndromexz = syndrome
        syndromexz = np.append(syndrome_z, 2*syndrome_x, axis=1)
        syndrome_error_volume = np.append(syndromexz, errorxz, axis=1)

        # for i in range(batch_size):
        #     error = error_model.generate(code, p)
        #     syndrome = code.measure_syndrome(error)
        #     syndrome_error_volume[i] = np.append(syndrome, error[:d ** 2])
        #     # # pred_error = bpdec.decode(syndrome)
        #     # syndrome_error = list(syndrome)
        #     # syndrome_error.extend(list(error[:d ** 2]))
        #     # # syndrome_error.extend(list(pred_error[:d ** 2]))
        #     # syndrome_error_volume[i, :] = syndrome_error

    if for_training:
        # syndrome_error_volume = np.zeros((batch_size, size), dtype=int)
        rng = np.random.default_rng(1)
        # error = np.zeros((batch_size, size), dtype='uint8')
        # syndrome = error
        # noise = np.random.uniform(0, 1, (batch_size, code.N))
        # err_z = err_x = noise
        err_z = np.zeros((batch_size, code.N),dtype='uint8')
        err_x = np.zeros((batch_size, code.N),dtype='uint8')
        for i in range(batch_size):
            pr = p * rng.random()
            px, py, pz = pr * error_model[0] * np.ones(code.N), pr * error_model[1] * np.ones(code.N), pr * error_model[
                2] * np.ones(code.N)

            # px, py, pz = pr / 3 * np.ones(code.N), pr / 3 * np.ones(code.N), pr / 3 * np.ones(code.N)
            noise = np.random.uniform(0, 1, (code.N))
            err_z[i] = np.logical_and(noise > px, noise < (px + py + pz))
            # err_z[i] = np.logical_and(noise[i] > px, noise[i] < (px + py + pz))
            # syndrome_x = (err_z @ code.hx.T) % 2  # [num_shots, N_half]
            err_x[i] = noise < (px + py)
            # syndrome_z = (err_x @ code.hz.T) % 2
        syndrome_x = (err_z @ code.hx.T) % 2  # [num_shots, N_half]
        syndrome_z = (err_x @ code.hz.T) % 2

        errorxz = (err_x + 2 * err_z)  # % 3
        # syndromexz = syndrome
        syndromexz = np.append(syndrome_z, 2*syndrome_x, axis=1)
        syndrome_error_volume = np.append(syndromexz, errorxz, axis=1)
        #mwpm

        # perror = (pred_error[:,:d**2] + 2*pred_error[:,d**2:]) #% 3
        # # syndromexz = syndrome
        # syndromexz = np.append(syndrome[:, :(d ** 2 - 1) // 2], 2*syndrome[:, (d ** 2 - 1) // 2:], axis=1)
        # # syndrome_error_volume = np.append(syndrome, pred_error[:, :d ** 2], axis=1)
        # syndrome_error_volume = np.append(syndromexz, perror, axis=1)


    #print(time.time() - starttime)
    return syndrome_error_volume


def adapt_trainset(batch, code, num_classes=2, for_training=True):
    # batch is now np array  [syndrome error]
    # if for_training:
    # batch = np.unique(batch, axis=0)
    #print(f"{len(batch)} unique in train set")
    st = time.time()

    error_index = code.N
    batch_np = batch
    targets_all = torch.LongTensor(batch_np)
    inputs_all = targets_all[:, :error_index]
    inputs_all = nn.functional.one_hot(inputs_all, num_classes)
    zeros = torch.zeros((inputs_all.shape[0], code.N, num_classes)).long()
    # zeros = torch.zeros((inputs_all.shape[0]).long()
    inputs_all = torch.cat((inputs_all, zeros), dim=1)
    # inputs_all = nn.functional.one_hot(inputs_all, num_classes)
    trainset = list(zip(inputs_all, targets_all))
    #print("adapt dataset",time.time()-st)
    return trainset


def init_log_probs_of_decoder(decoder, my_log_probs):
    #print("old ", decoder.log_prob_ratios)

    for i in range(len(decoder.log_prob_ratios)):
        decoder.set_log_prob(i, my_log_probs[i])

    #print("new ", decoder.log_prob_ratios)
    
def unwrap_ddp(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

def logical_error_rate(gnn, testloader, code, osd_decoder=None, enable_osd=False, n_iters=0):
    #keep enable_osd = false during training
    size = 2 * code.N
    error_index = code.N
    gnn.eval()
    device = gnn.device
    with torch.no_grad():
        n_test = 0#torch.tensor(0,dtype=torch.int,device=device)
        n_l_error = 0#torch.tensor(0,dtype=torch.int,device=device)
        n_codespace_error = 0#torch.tensor(0,dtype=torch.int,device=device)
        n_total_ler = 0#torch.tensor(0,dtype=torch.int,device=device)
        #total_entropy = 0
        hx = torch.tensor(code.hx,dtype=torch.float16,device=device)
        hz = torch.tensor(code.hz,dtype=torch.float16,device=device)
        lx = torch.tensor(code.lx,dtype=torch.float16,device=device)
        lz = torch.tensor(code.lz,dtype=torch.float16,device=device)

        for i, (inputs, targets, src_ids, dst_ids) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
            # batch_size = inputs.size(0) // size
            
            #outputs_mc = gnn.predict_mc(inputs, src_ids, dst_ids, n_mc_samples=n_mc_samples)
            #outputs = outputs_mc.mean(dim=0)  # average over MC samples
            
            outputs = gnn(inputs, src_ids, dst_ids)  # [n_iters, batch*n_nodes, 9]
            encoding = outputs.shape[-1]
            if enable_osd:
                # final_solution = osd(outputs,targets,code,osd_decoder)
                n_l, n_c, n_t, batch_size = osd(outputs,targets,code,hx,hz,unwrap_ddp(gnn).n_iters,osd_decoder)
                # solution = outputs.view(gnn.n_iters, -1, size, encoding)
                # final_solution = solution[-1, :, error_index:].argmax(dim=2).cpu()

                n_l_error += n_l
                n_codespace_error += n_c
                n_total_ler += n_t
                n_test += batch_size

            else:
                solution = outputs.view(unwrap_ddp(gnn).n_iters, -1, size, encoding)
                final_solution = solution[:, :, error_index:].argmax(dim=-1)
                batch_size = final_solution.shape[1]

                final_targets = targets.view(batch_size, size)[:, error_index:]
                final_targetsx = torch.where(final_targets == 1, final_targets, 0) + torch.where(final_targets == 3,
                                                                                                 final_targets, 0) // 3
                final_targetsz = torch.where(final_targets == 2, final_targets, 0) // 2 + torch.where(final_targets == 3,
                                                                                                      final_targets, 0) // 3

                final_solutionx = torch.where(final_solution == 1, final_solution, 0) + torch.where(final_solution == 3,
                                                                                                    final_solution, 0) // 3
                final_solutionz = torch.where(final_solution == 2, final_solution, 0) // 2 + torch.where(
                    final_solution == 3, final_solution, 0) // 3

                # final_solution = torch.cat((final_solutionx, final_solutionz), dim=1)
                # final_targets = torch.cat((final_targetsx, final_targetsz), dim=1)
                n_iters = final_solution.shape[0]
                final_targetsx = final_targetsx.unsqueeze(0).repeat(n_iters, 1, 1)
                final_targetsz = final_targetsz.unsqueeze(0).repeat(n_iters, 1, 1)

                rfx = ((final_targetsx + final_solutionx) % 2).type(torch.float16)
                rfz = ((final_targetsz + final_solutionz) % 2).type(torch.float16)

                # rfx = rfx.reshape(-1, rfx.shape[-1])
                # ms = np.append((rfx.reshape(-1, rfx.shape[-1]) @ code.hz.T) % 2, (rfz.reshape(-1, rfz.shape[-1]) @ code.hx.T) % 2, axis=-1)
                # ms = ms.reshape(n_iters, batch_size, -1)
                # mseitr = np.any(ms, axis=2)
                # n_codespace_error += mseitr.sum(axis=-1).min()
                
                # compute entropy for aleatoric uncertainty
                #probs = F.softmax(outputs[-1], dim=-1)  # take last iteration logits
                #entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # shape: [batch * size]
                #entropy = entropy.view(batch_size, size)  # [batch_size, num_nodes]
                #entropy_data = entropy[:, error_index:]  # just error qubits
                #avg_entropy = entropy_data.mean().item()

                ms = torch.cat(((rfx @ hz.T) % 2,(rfz @ hx.T) % 2),dim=-1).type(torch.int)
                mseitr = torch.any(ms, axis=2)
                minitr = torch.argmin(mseitr.sum(axis=-1))
                mse = mseitr[minitr]
                n_codespace_error += mse.sum().item()


                # minitr = np.argmin(mseitr.sum(axis=-1))
                # msi = np.where(mseitr[minitr] == 0)

                # l = np.append((rfx[minitr][msi] @ code.lz.T) % 2, (rfz[minitr][msi] @ code.lx.T) % 2, axis=-1)
                # l = np.any(l, axis=1)
                # n_l_error += l.sum()

                l = torch.cat(((rfx[minitr] @ lz.T) % 2, (rfz[minitr] @ lx.T) % 2), dim=-1).type(torch.int)
                l = torch.any(l, axis=1)
                n_l_error += l.sum().item()
                
                #total_entropy += avg_entropy * batch_size

                n_total_ler += torch.logical_or(l, mse).sum().item()
                # n_total_ler += l.sum() + mse.sum()
                n_test += batch_size
        # n_total_ler = (n_l_error + n_codespace_error)
        return (n_l_error / n_test), (n_codespace_error / n_test), (n_total_ler/n_test)

def osd(outputs, targets, code, hx, hz, n_iters, osd_decoder=None):
    x_decoder, z_decoder = osd_decoder  
    
    size = 2 * code.N
    error_index = code.N
    # n_iters=out.shape[0]
    encoding = outputs.shape[-1]
    solution = outputs.view(n_iters, -1, size, encoding)
    final_solution = solution[:, :, error_index:].argmax(dim=-1)
    batch_size = final_solution.shape[1]

    final_targets = targets.view(batch_size, size)[:, error_index:]
    final_targetsx = torch.where(final_targets == 1, final_targets, 0) + torch.where(final_targets == 3,
                                                                                     final_targets, 0) // 3
    final_targetsz = torch.where(final_targets == 2, final_targets, 0) // 2 + torch.where(final_targets == 3,
                                                                                          final_targets, 0) // 3

    final_solutionx = torch.where(final_solution == 1, final_solution, 0) + torch.where(final_solution == 3,
                                                                                        final_solution, 0) // 3
    final_solutionz = torch.where(final_solution == 2, final_solution, 0) // 2 + torch.where(
        final_solution == 3, final_solution, 0) // 3

    # final_solution = torch.cat((final_solutionx, final_solutionz), dim=1)
    # final_targets = torch.cat((final_targetsx, final_targetsz), dim=1)
    n_iters = final_solution.shape[0]
    final_targetsx = final_targetsx.unsqueeze(0).repeat(n_iters, 1, 1)
    final_targetsz = final_targetsz.unsqueeze(0).repeat(n_iters, 1, 1)

    # gpu part
    rfx = ((final_targetsx + final_solutionx) % 2).type(torch.float16)
    rfz = ((final_targetsz + final_solutionz) % 2).type(torch.float16)

    ms = torch.cat(((rfx @ hz.T) % 2, (rfz @ hx.T) % 2), dim=-1).type(torch.int)
    mseitr = torch.any(ms, axis=2)
    minitr = torch.argmin(mseitr.sum(axis=-1))
    mse = np.array(mseitr[minitr].type(torch.int).cpu())
    nonzero_syn_id = np.nonzero(mse.astype("uint8"))[0]
    # cpu part

    # rfx = np.array(((final_targetsx + final_solutionx) % 2).type(torch.int).cpu())
    # rfz = np.array(((final_targetsz + final_solutionz) % 2).type(torch.int).cpu())
    #
    # ms = np.append((rfx @ code.hz.T) % 2, (rfz @ code.hx.T) % 2, axis=-1)
    # mseitr = np.any(ms, axis=2)
    # minitr = np.argmin(mseitr.sum(axis=-1))
    # mse = mseitr[minitr]
    # nonzero_syn_id = np.nonzero(mse.astype("uint8"))[0]

    #cpu part
    final_solution = np.array(nn.functional.softmax(solution[minitr, :, error_index:], dim=2).cpu())

    fllrx = np.log((final_solution[:, :, 1] + final_solution[:, :, 3]) / final_solution[:, :, 0])  # works better
    fllrz = np.log((final_solution[:, :, 2] + final_solution[:, :, 3]) / final_solution[:, :, 0])


    final_syn = np.array((targets.view(batch_size, size)[:, :error_index]).cpu())
    final_syn = np.append(final_syn[:,:error_index//2],final_syn[:,error_index//2:]//2 , axis = 1)

    # final_solution = solution[minitr, :, error_index:].argmax(dim=2)
    # osd_out = np.array(final_solution.cpu())
    osd_err_x = np.array(final_solutionx[minitr].cpu())
    osd_err_z = np.array(final_solutionz[minitr].cpu())

    for i in nonzero_syn_id:
        # dec = decoder(code,mwpm_decoder.error_model,mwpm_decoder.error_rate,weights=(fllrx[i],fllrz[i]))
        #init_log_probs_of_decoder(x_decoder, fllrx[i])    # x with z originally
        #init_log_probs_of_decoder(z_decoder, fllrz[i])
        # init_log_probs_of_decoder(osd_decoder.x_decoder, fnllr_sig[i])
        osd_err_x[i] = x_decoder.decode(final_syn[i,:error_index//2])
        osd_err_z[i] = z_decoder.decode(final_syn[i,error_index//2:])

    rfx = ((np.array(final_targetsx[0].cpu()) + osd_err_x) % 2)
    rfz = ((np.array(final_targetsz[0].cpu()) + osd_err_z) % 2)

    ms = np.append((rfx @ code.hz.T) % 2, (rfz @ code.hx.T) % 2, axis=-1)
    mse = np.any(ms, axis=1)
    n_codespace_error = mse.sum()

    l = np.append((rfx @ code.lz.T) % 2, (rfz @ code.lx.T) % 2, axis=-1)
    l = np.any(l, axis=1)
    n_l_error = l.sum()

    n_total_ler = np.logical_or(l, mse).sum()
    # n_total_ler = n_l_error
    # n_codespace_error = 0
    # n_total_ler += l.sum() + mse.sum()
    n_test = batch_size

    return n_l_error, n_codespace_error , n_total_ler, n_test

def ler_loss(out, targets, code):
    size = 2 * code.N
    error_index = code.N
    device = out.device
    # n_iters=out.shape[0]
    encoding = out.shape[-1]
    # outputs = gnn(inputs, src_ids, dst_ids)  # [n_iters, batch*n_nodes, 9]
    solution = (out.view(-1, size, encoding))
    final_solution = nn.functional.softmax(solution[:, error_index:, :], dim=2)
    # final_solution =  nn.functional.sigmoid(solution[:, error_index:, :])
    batch_size = final_solution.shape[0]
    # ax = 0
    # az = 0
    msx = 0
    msz = 0
    #hx = code.Hx.toarray()
    #hxperp = torch.FloatTensor(kernel(hx)[0]).to(device)
    hxperp= QuBA.hxperp
    #hz = code.Hz.toarray()
    #hzperp = torch.FloatTensor(kernel(hz)[0]).to(device)
    hzperp = QuBA.hzperp
    # lz = torch.Tensor(code.logicals_z)
    # # residual = torch.tensor([0.0], requires_grad=True)
    final_targets = targets.view(batch_size, size)[:, error_index:]
    final_targetsx = torch.where(final_targets == 1, final_targets, 0) + torch.where(final_targets == 3, final_targets, 0) // 3
    final_targetsz = torch.where(final_targets == 2, final_targets, 0) // 2 + torch.where(final_targets == 3, final_targets, 0) // 3

    rx = final_targetsx + final_solution[:, :, 1] + final_solution[:, :, 3]
    rfx = (torch.abs(torch.sin(torch.pi * rx / 2)))
    msx_batch = torch.mean(torch.abs(torch.sin(torch.pi * ((rfx @ hxperp.T)) / 2)), dim=1)
    msx = msx_batch.sum()

    rz = final_targetsz + final_solution[:, :, 2] + final_solution[:, :, 3]
    rfz = (torch.abs(torch.sin(torch.pi * rz / 2)))
    msz_batch = torch.mean(torch.abs(torch.sin(torch.pi * ((rfz @ hzperp.T)) / 2)), dim=1)
    msz = msz_batch.sum()


    # loss=nn.functional.cross_entropy(out,targets)
    n_l_error = msx + msz#+ ax#+ eloss # + ax #+ sloss +
    # n_x_error = eloss + msz + msx #+ ax #+ sloss +
    # n_x_error = ms
    # n_z_error = az
    return n_l_error / batch_size #+ loss

def compute_accuracy(gnn, testloader, code):
    gnn.eval()
    size = 2 * code.N
    error_index = code.N
    # device = torch.device('cpu')
    device = gnn.device
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        losses = []
        for i, (inputs, targets, src_ids, dst_ids) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)

            outputs = gnn(inputs, src_ids, dst_ids)
            encoding = outputs.shape[-1]
            loss = 0
            # eloss=0
            # sloss=0
            # loss_itr_l1=[]
            # loss_itr_l2 = []
            for out in outputs:
                # loss += criterion(out, targets)
                eloss = criterion(out.view(-1, size, encoding)[:, error_index:].reshape(-1, encoding),
                                  targets.view(-1, size)[:, error_index:].flatten())
                sloss = criterion(out.view(-1, size, encoding)[:, :error_index].reshape(-1, encoding),
                                  targets.view(-1, size)[:, :error_index].flatten())
                l1 = ler_loss(out, targets, code)
                l2 = sloss + eloss
                loss += l1+l2
                # loss_itr_l1.append(l1)
                # loss_itr_l2.append(l2)
                # xloss, zloss = ler_loss(out, targets, surface_code)
                # # eloss += criterion(out.view(-1, size,n_node_inputs)[:, :syndrome_index].reshape(-1,n_node_inputs), targets.view(-1, size)[:,:syndrome_index].flatten())
                # sloss = criterion(out.view(-1, size, n_node_inputs)[:, syndrome_index:].reshape(-1,n_node_inputs), targets.view(-1, size)[:,syndrome_index:].flatten())
                # loss += ler_loss(out, targets, code) + sloss
                # loss += xloss + zloss
            # loss = eloss + sloss
            loss /= outputs.shape[0]

            losses.append(loss.detach())

        losses = torch.mean(torch.tensor(losses)).item()
    return losses

def bb_code(d):
    code,A_list,B_list = 0, 0, 0
    if(d==6):
        # [[72, 12, 6]]
        code, A_list, B_list = create_bivariate_bicycle_codes(6, 6, [3], [1, 2], [1, 2], [3])
    elif(d==10):
        # [[90,8,10]]
        code, A_list, B_list = create_bivariate_bicycle_codes(15, 3, [9], [1,2], [2,7], [0])
    elif(d==12):
        # [[144,12,12]]
        code, A_list, B_list = create_bivariate_bicycle_codes(12, 6, [3], [1, 2], [1, 2], [3])
    elif(d==18):
        # [[288,12,18]]
        code, A_list, B_list = create_bivariate_bicycle_codes(12, 12, [3], [2,7], [1,2], [3])
    elif(d==24):
        #[[360,12,<=24]]
        code, A_list, B_list = create_bivariate_bicycle_codes(30, 6, [9], [1,2], [25,26], [3])
    elif(d==34):
        # [[756,16,<=34]]
        code, A_list, B_list = create_bivariate_bicycle_codes(21,18, [3], [10,17], [3,19], [5])

    else:
        raise ValueError("wrong distance")

    return code


def copbb_code(d):
    code = None
    if d == 6:
        # [[30, 4, 6]]
        code = create_copBB_code(3, 5, [0, 1, 2], [1, 3, 8])
    #if d == 6:
        # [[42, 6, 6]]
        #code = create_copBB_code(3, 7, [0, 2, 3], [1, 3, 11])
    #if d == 6:
        # [[108, 12, 6]]
    #    code = create_copBB_code(2, 27, [2, 5, 44], [8, 14, 47])
    elif d == 8:
        # [[70, 6, 8]]
        code = create_copBB_code(5, 7, [0, 1, 5], [0, 1, 12])
    elif d == 10:
        # [[126, 12, 10]]
        code = create_copBB_code(7, 9, [0, 1, 58], [0, 13, 41])
    elif d == 16:
        # [[154, 6, 16]]
        code = create_copBB_code(7, 11, [0, 1, 31], [0, 19, 53])
    else:
        raise ValueError(f"Unsupported coprime BB code for distance {d}")

    return code


def save_model(model, filename, confirm=True):
    if confirm:
        try:
            save = input('Do you want to save the model (type yes to confirm)? ').lower()
            if save != 'yes':
                print('Model not saved.')
                return
        except:
            raise Exception('The notebook should be run or validated with skip_training=True.')

    torch.save(model.state_dict(), filename)
    print('Model saved to %s.' % (filename))


def load_model(model, filename, device):
    filesize = os.path.getsize(filename)
    if filesize > 30000000:
        raise 'The file size should be smaller than 30Mb. Please try to reduce the number of model parameters.'
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()