import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import itertools

def get_current_sparsity(current_step, total_steps, final_sparsity, anneal_frac=0.5):
    anneal_steps = int(total_steps * anneal_frac)
    if current_step < anneal_steps:
        return final_sparsity * (current_step / anneal_steps)
    else:
        return final_sparsity

def binary_gate(x):
    return (x > 0).float() - torch.sigmoid(x).detach() + torch.sigmoid(x)

def get_gate_hook(gate_param):
    def hook(module, input, output):
        return output * binary_gate(gate_param)
    return hook

def get_hard_mask_hook(indices, device):
    def hook(module, input, output):
        mask = torch.zeros(1, output.shape[1], 1, 1).to(device)
        if len(indices) > 0:
            mask[:, indices, :, :] = 1.0
        return output * mask
    return hook

def apply_weight_sparsity(model, sparsity_level=0.90, min_alive=4):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                flat_param = param.abs().flatten()
                num_keep = int((1 - sparsity_level) * flat_param.numel())
                if num_keep < 1: num_keep = 1
                
                threshold = torch.topk(flat_param, num_keep).values[-1]
                mask = (param.abs() >= threshold).float()

                if 'conv' in name and param.dim() == 4:
                    for i in range(param.shape[0]):
                        filter_weights = param[i]
                        alive_count = (mask[i] > 0).sum().item()
                        
                        if alive_count < min_alive:
                            top_k_vals = torch.topk(filter_weights.abs().flatten(), min_alive).values
                            if top_k_vals.numel() > 0:
                                revival_threshold = top_k_vals[-1]
                                revival_mask = (filter_weights.abs() >= revival_threshold).float()
                                mask[i] = torch.max(mask[i], revival_mask)
                
                param.data.mul_(mask)

def analyze_iou(circuit_storage):
    analyzed_names = list(circuit_storage.keys())
    if len(analyzed_names) < 2: return
    layers = list(circuit_storage[analyzed_names[0]].keys())
    for layer in layers:
        print(f"\nLayer: {layer}")
        for c1, c2 in itertools.combinations(analyzed_names, 2):
            set1, set2 = set(circuit_storage[c1][layer]), set(circuit_storage[c2][layer])
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            iou = intersection / union if union > 0 else 0
            print(f"  IoU ({c1} vs {c2}): {iou:.4f}")