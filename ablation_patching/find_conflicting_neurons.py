import json
import argparse
import sys
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Find conflicting neurons and save them to a file.")
    parser.add_argument("--file", type=str, required=True, help="Path to the circuits_per_round...json file.")
    parser.add_argument("--round", type=str, default="round_10", help="Which round to analyze.")
    parser.add_argument("--output", type=str, default="patching_candidates.json", help="Output file for candidates.")
    args = parser.parse_args()

    try:
        with open(args.file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found at {args.file}"); sys.exit(1)

    if args.round not in data or "clients_global_model" not in data[args.round]:
        print(f"ERROR: Could not find path in JSON file."); sys.exit(1)
        
    circuits = data[args.round]["clients_global_model"]
    
    # Structure: neuron_usage[layer][neuron_index] = list_of_circuits_using_it
    neuron_usage = defaultdict(lambda: defaultdict(list))

    for client, class_data in circuits.items():
        for class_name, circuit_info in class_data.items():
            active_nodes = circuit_info.get("active_nodes", circuit_info)
            for layer, nodes in active_nodes.items():
                for neuron_idx in nodes:
                    neuron_usage[layer][neuron_idx].append({"client": client, "class": class_name})

    # --- NEW: Structure and save candidates ---
    candidates = []
    print("--- Finding Conflicting Neurons ---")
    for layer, neurons in sorted(neuron_usage.items()):
        for neuron_idx, users in sorted(neurons.items()):
            if len(users) > 1:
                user_clients = {u['client'] for u in users}
                if len(user_clients) > 1: # True inter-client conflict
                    candidates.append({
                        "layer": layer,
                        "neuron": neuron_idx,
                        "conflict_count": len(users),
                        "users": users
                    })

    # Sort candidates by how many circuits use them (most conflicted first)
    candidates.sort(key=lambda x: x['conflict_count'], reverse=True)

    with open(args.output, 'w') as f:
        json.dump(candidates, f, indent=2)
        
    print(f"Saved {len(candidates)} conflicting neuron candidates to {args.output}")

if __name__ == "__main__":
    main()