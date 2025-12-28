import torch
import argparse
import os
import sys
import json
import csv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from config import Config
from dataset import get_dataset, get_test_dataloader

# ... (Hook functions remain the same) ...
captured_activation = None
config = None

def get_capture_hook():
    def hook(model, input, output):
        global captured_activation; captured_activation = output.detach().clone()
    return hook

def get_patch_hook(patch_value, neuron_idx=None):
    def hook(model, input, output):
        if neuron_idx is None: return patch_value
        patched_output = output.clone()
        patched_output[:, neuron_idx, :, :] = patch_value[:, neuron_idx, :, :]
        return patched_output
    return hook

def load_model(path, config_obj):
    if not os.path.exists(path): return None
    model = get_model(config_obj)
    checkpoint = torch.load(path, map_location=config_obj.device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- NEW FUNCTION: Find a guaranteed failure case ---
def find_failed_image(model, dataloader, target_class):
    """Finds the first image of target_class that the model misclassifies."""
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            mask = (labels == target_class)
            if mask.sum() == 0: continue
            
            # Filter for the target class
            class_images = images[mask]
            class_labels = labels[mask]
            
            outputs = model(class_images.to(config.device))
            _, predicted = torch.max(outputs, 1)
            
            # Find a misclassification
            wrong_mask = (predicted.cpu() != class_labels)
            if wrong_mask.sum() > 0:
                fail_idx = wrong_mask.nonzero(as_tuple=True)[0][0]
                return class_images[fail_idx].unsqueeze(0), class_labels[fail_idx]
    return None, None
# ---------------------------------------------------

def run_pass(model, image, label, target_layer=None, patch_value=None, neuron_idx=None):
    # ... (This function remains unchanged)
    hook_handle = None
    if target_layer:
        try:
            module = dict(model.named_modules())[target_layer]
            if patch_value is not None:
                hook_handle = module.register_forward_hook(get_patch_hook(patch_value, neuron_idx))
            else:
                hook_handle = module.register_forward_hook(get_capture_hook())
        except KeyError: return False, True
    with torch.no_grad():
        output = model(image.to(config.device))
        _, predicted = torch.max(output, 1)
    if hook_handle: hook_handle.remove()
    return predicted.item() == label.item(), False

def main():
    parser = argparse.ArgumentParser(description="Automated Causal Patching Pipeline.")
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--round", type=str, default="round_10")
    parser.add_argument("--candidates_file", type=str, default="patching_candidates.json")
    parser.add_argument("--results_file", type=str, default="patching_results.csv")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of candidates to test.")
    args = parser.parse_args()
    
    global config
    config = Config()

    # --- Load Data, Models, and Candidates ---
    _, testset = get_dataset(config)
    testloader = get_test_dataloader(testset, config)
    
    with open(args.candidates_file, 'r') as f:
        candidates = json.load(f)

    base_path = os.path.join(args.dir, 'controlled_noniid')
    global_path = os.path.join(base_path, f"checkpoint_{args.round}.pt")
    model_global = load_model(global_path, config)
    
    if not model_global: print(f"Global model not found at {global_path}"); return

    # --- Setup CSV Logger ---
    csv_file = open(args.results_file, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Layer", "Neuron", "Source_Client", "Source_Class", "Baseline_Correct", "Neuron_Patch_Correct", "Layer_Patch_Correct"])

    # --- Main Loop ---
    print(f"Testing top {args.limit} candidates from {args.candidates_file}...")
    for i, candidate in enumerate(candidates):
        if i >= args.limit: break
        
        layer = candidate['layer']
        neuron = candidate['neuron']
        
        # We need a representative user to act as the "source"
        source_user = candidate['users'][0]
        source_client = source_user['client']
        source_class_name = source_user['class']
        source_class_idx = int(source_class_name.split(' - ')[0])

        print(f"\n--- Candidate {i+1}/{args.limit}: {layer}[{neuron}] (Source: {source_client}/{source_class_name}) ---")
        
        # Load the specialist local model
        round_dir = os.path.join(base_path, args.round)
        local_path = os.path.join(round_dir, f"{source_client}_model.pt")
        model_local = load_model(local_path, config)
        if not model_local: continue

        # --- KEY CHANGE: Find a FAILED image for the baseline ---
        image, label = find_failed_image(model_global, testloader, source_class_idx)
        if image is None:
            print(f"  > INFO: Global model is already 100% accurate on Class {source_class_idx}. No failures to test. Skipping.")
            continue
        # ---------------------------------------------------------

        # 1. Capture Golden Activation from Specialist
        run_pass(model_local, image, label, target_layer=layer)
        if captured_activation is None: continue

        # 2. Run Baseline (We know this will be False)
        baseline_correct, _ = run_pass(model_global, image, label)

        # 3. Run Neuron Patch
        neuron_patch_correct, _ = run_pass(model_global, image, label, target_layer=layer, patch_value=captured_activation, neuron_idx=neuron)
        
        # 4. Run Layer Patch
        layer_patch_correct, _ = run_pass(model_global, image, label, target_layer=layer, patch_value=captured_activation, neuron_idx=None)

        print(f"  > Results | Baseline: {baseline_correct}, Neuron Patch: {neuron_patch_correct}, Layer Patch: {layer_patch_correct}")
        
        # 5. Log to CSV
        writer.writerow([layer, neuron, source_client, source_class_name, baseline_correct, neuron_patch_correct, layer_patch_correct])
        csv_file.flush()
    
    csv_file.close()
    print(f"\nPipeline complete. Results saved to {args.results_file}")

if __name__ == "__main__":
    main()