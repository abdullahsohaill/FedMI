import csv
import matplotlib.pyplot as plt
import argparse
import os

def analyze_and_visualize_results(csv_path, output_path):
    """Reads the patching CSV and generates a summary table and text analysis."""
    
    results = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
    except FileNotFoundError:
        print(f"ERROR: Results file not found at {csv_path}")
        return

    if not results:
        print("No results to analyze.")
        return

    # --- Text Analysis ---
    print("\n--- Automated Analysis of Patching Results ---")
    
    upstream_example = None
    downstream_example = None
    
    for res in results:
        if res['Baseline_Correct'] == 'False' and res['Neuron_Patch_Correct'] == 'False' and res['Layer_Patch_Correct'] == 'True':
            if upstream_example is None:
                upstream_example = res
        
        if res['Baseline_Correct'] == 'False' and res['Neuron_Patch_Correct'] == 'False' and res['Layer_Patch_Correct'] == 'False':
            if downstream_example is None:
                downstream_example = res

    if upstream_example:
        print("\n[Finding 1] Evidence of UPSTREAM SIGNAL CORRUPTION:")
        print(f"  - Case: {upstream_example['Source_Class']} (from {upstream_example['Source_Client']})")
        print(f"  - When patching the full '{upstream_example['Layer']}' layer, performance was rescued (True).")
        print(f"  - Conclusion: The weights in later layers are functional. The error was caused by a corrupted signal originating before or at {upstream_example['Layer']}.")
    
    if downstream_example:
        print("\n[Finding 2] Evidence of DOWNSTREAM LOGIC CORRUPTION:")
        print(f"  - Case: {downstream_example['Source_Class']} (from {downstream_example['Source_Client']})")
        print(f"  - Even when patching the full '{downstream_example['Layer']}' layer, performance was NOT rescued (False).")
        print(f"  - Conclusion: The weights in a later layer (e.g., conv3 or fc) are fundamentally broken for this task.")

    # --- Table Visualization ---
    
    cell_text = []
    colors = []
    
    columns = ["Layer", "Neuron", "Class", "Baseline", "Neuron Patch", "Layer Patch", "Damage Type"]
    
    for res in results:
        baseline = res['Baseline_Correct']
        neuron_patch = res['Neuron_Patch_Correct']
        layer_patch = res['Layer_Patch_Correct']
        
        damage_type = "N/A"
        if baseline == 'False':
            if layer_patch == 'True':
                damage_type = "Upstream Signal"
            else:
                damage_type = "Downstream Logic"

        row = [
            res['Layer'], res['Neuron'], res['Source_Class'],
            baseline, neuron_patch, layer_patch, damage_type
        ]
        cell_text.append(row)
        
        color_row = ['white'] * len(columns)
        if baseline == 'False' and neuron_patch == 'True':
            color_row[4] = '#2ecc71' # Green
        if baseline == 'False' and layer_patch == 'True':
            color_row[5] = '#2ecc71' # Green
        colors.append(color_row)

    fig, ax = plt.subplots(figsize=(14, len(results) * 0.7 + 1))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=cell_text, colLabels=columns, cellColours=colors, loc='center')
    
    # --- THIS IS THE FIX ---
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    # -----------------------

    table.scale(1.2, 1.2)
    
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#34495e')
        else:
            cell.set_facecolor('#ecf0f1')
    
    fig.suptitle("Causal Patching Ablation Summary", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved summary table to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate summary report from patching results.")
    parser.add_argument("--file", type=str, required=True, help="Path to patching_results.csv")
    args = parser.parse_args()
    
    output_path = os.path.splitext(args.file)[0] + "_summary.png"
    analyze_and_visualize_results(args.file, output_path)

if __name__ == "__main__":
    main()