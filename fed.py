import torch
import torch.nn as nn
import torch.optim as optim
import copy
import json
import os
import glob
import numpy as np
from tqdm import tqdm
from train import get_gate_hook, get_hard_mask_hook, apply_weight_sparsity, get_current_sparsity

def save_checkpoint(global_model, round_num, all_circuits, config):
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    path = os.path.join(config.checkpoint_dir, f"checkpoint_round_{round_num}.pt")
    state = {
        'round': round_num,
        'model_state_dict': global_model.state_dict(),
        'all_circuits': all_circuits,
        'config': config 
    }
    torch.save(state, path)
    print(f"  [Checkpoint] Saved round {round_num} to {path}")

def load_latest_checkpoint(global_model, config):
    if not os.path.exists(config.checkpoint_dir):
        print("  [Resume] No checkpoint directory found. Starting from scratch.")
        return 0, {}
    files = glob.glob(os.path.join(config.checkpoint_dir, "checkpoint_round_*.pt"))
    if not files:
        print("  [Resume] No checkpoint files found. Starting from scratch.")
        return 0, {}
    latest_file = max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    print(f"  [Resume] Loading checkpoint: {latest_file}")
    checkpoint = torch.load(latest_file, map_location=config.device)
    global_model.load_state_dict(checkpoint['model_state_dict'])
    start_round = checkpoint['round']
    all_circuits = checkpoint['all_circuits']
    print(f"  [Resume] Successfully loaded. Resuming from Round {start_round + 1}")
    return start_round, all_circuits

def evaluate_circuit(model, testloader, circuit, target_class, config):
    device = config.device
    hooks = []
    for layer_name, indices in circuit.items():
        module = model.get_submodule(layer_name)
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        hooks.append(module.register_forward_hook(get_hard_mask_hook(idx_tensor, device)))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            mask = (labels == target_class)
            if mask.sum() == 0: continue
            outputs = model(inputs[mask])
            _, predicted = torch.max(outputs.data, 1)
            total += mask.sum().item()
            correct += (predicted == labels[mask]).sum().item()
    for h in hooks: h.remove()
    return (100 * correct / total) if total > 0 else 0.0

def local_train(model, dataloader, config):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_steps = len(dataloader) * config.local_epochs
    current_step = 0
    for epoch in range(config.local_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.local_epochs}", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            
            if getattr(config, 'train_mode', 'dense') == 'sparse':
                sparsity_to_apply = get_current_sparsity(
                    current_step=current_step,
                    total_steps=total_steps,
                    final_sparsity=config.target_sparsity,
                    anneal_frac=getattr(config, 'anneal_frac', 0.5)
                )
                apply_weight_sparsity(model, sparsity_to_apply)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", sparsity=f"{sparsity_to_apply:.2f}")
            else:
                 progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            current_step += 1
            
    return model

def fedavg(global_model, client_models):
    weights = [1.0 / len(client_models)] * len(client_models)
    global_state = global_model.state_dict()
    target_device = next(global_model.parameters()).device
    for key in global_state.keys():
        global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32).to(target_device)
        for i, client_model in enumerate(client_models):
            global_state[key] += weights[i] * client_model.state_dict()[key].to(target_device).float()
    global_model.load_state_dict(global_state)

def evaluate(model, testloader, config):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def discover_client_circuit(model, dataloader, target_class, config):
    device = config.device
    criterion = nn.CrossEntropyLoss()
    original_grads = {name: param.requires_grad for name, param in model.named_parameters()}
    model.eval()
    for param in model.parameters(): param.requires_grad = False
    gate_params, hooks, layers = {}, [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers.append(name)
            gate_params[name] = nn.Parameter(torch.ones(1, module.out_channels, 1, 1).to(device) * 2.0)
            hooks.append(module.register_forward_hook(get_gate_hook(gate_params[name])))
    optimizer = optim.Adam(gate_params.values(), lr=config.gate_lr)
    data_iter = iter(dataloader)
    for _ in range(config.discovery_steps):
        try: inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, labels = next(data_iter)
        inputs, labels = inputs.to(device), labels.to(device)
        mask = (labels == target_class)
        if mask.sum() == 0: continue
        optimizer.zero_grad()
        l0_loss = sum(torch.sigmoid(p).sum() for p in gate_params.values())
        loss = criterion(model(inputs[mask]), labels[mask]) + (config.l0_lambda * l0_loss)
        loss.backward()
        optimizer.step()
    circuit = {name: np.where((gate_params[name] > 0).float().cpu().numpy().flatten() == 1)[0].tolist() for name in layers}
    for h in hooks: h.remove()
    for name, param in model.named_parameters(): param.requires_grad = original_grads.get(name, True)
    return circuit

def save_circuits_to_json(all_circuits, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, 'w') as f:
        json.dump(all_circuits, f, indent=2)
    print(f"Circuits saved to {path}")

def run_federated_round(global_model, client_dataloaders, testloader, config, round_num, class_names, all_circuits):
    client_models = []
    round_circuits = {"clients_local_model": {}, "clients_global_model": {}}
    print(f"\n--- Round {round_num + 1}/{config.num_rounds} ---")
    for i, dl in enumerate(client_dataloaders):
        print(f"  Client {i}: Local Training...")
        c_model = local_train(copy.deepcopy(global_model), dl, config)
        client_models.append(c_model)
        c_circs = {}
        for tc in config.classes_to_analyze:
            name = class_names[tc]
            circ = discover_client_circuit(c_model, dl, tc, config)
            c_circs[name] = circ
            evaluate_circuit(c_model, testloader, circ, tc, config)
        round_circuits["clients_local_model"][f"client_{i}"] = c_circs
    fedavg(global_model, client_models)
    print(f"  Global Model: Running per-client circuit discovery & evaluation...")
    for i, dl in enumerate(client_dataloaders):
        gm_copy = copy.deepcopy(global_model)
        cg_circs = {}
        for tc in config.classes_to_analyze:
            name = class_names[tc]
            circ = discover_client_circuit(gm_copy, dl, tc, config)
            cg_circs[name] = circ
            acc = evaluate_circuit(gm_copy, testloader, circ, tc, config)
            counts = {layer: len(idx) for layer, idx in circ.items()}
            print(f"    Global + Client {i} Data - {name}: {counts} | Acc: {acc:.2f}%")
        round_circuits["clients_global_model"][f"client_{i}"] = cg_circs
        del gm_copy
    all_circuits[f"round_{round_num + 1}"] = round_circuits
    return global_model

def run_federated_training(global_model, client_dataloaders, testloader, config, class_names):
    print("\n=== Federated Training with Circuit Discovery ===")
    mode = getattr(config, 'train_mode', 'sparse')
    print(f"Training Mode: {mode.upper()}")
    all_circuits = {}
    start_round = 0
    if config.resume:
        start_round, loaded_circuits = load_latest_checkpoint(global_model, config)
        all_circuits = loaded_circuits
        config.resume = False
    if start_round >= config.num_rounds:
        print("  [Resume] Training already completed in checkpoints!")
        return global_model, all_circuits
    for round_num in range(start_round, config.num_rounds):
        global_model = run_federated_round(
            global_model, client_dataloaders, testloader, config, round_num, class_names, all_circuits
        )
        acc = evaluate(global_model, testloader, config)
        print(f"  Round {round_num + 1} Global Acc: {acc:.2f}%")
        save_checkpoint(global_model, round_num + 1, all_circuits, config)
        save_circuits_to_json(all_circuits, "circuits_json_files/circuits_per_round.json")
    for param in global_model.parameters():
        param.requires_grad = False
    return global_model, all_circuits