import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Network architecture
LAYER_SIZES = [32, 64, 128]
LAYER_NAMES = ['conv1', 'conv2', 'conv3']

def load_circuits(json_file):
    """Load the circuits JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def visualize_circuit(circuit_data, layer_sizes, layer_names, title, ax):
    """
    Visualize a single circuit showing active neurons and connections.
    
    Args:
        circuit_data: Dictionary with layer names as keys and neuron indices as values
        layer_sizes: List of integers representing neurons in each layer
        layer_names: List of layer names
        title: Title for the subplot
        ax: Matplotlib axis object
    """
    
    # Layout parameters
    layer_spacing = 3.0  # Horizontal spacing between layers
    neuron_radius = 0.15
    active_color = '#2ecc71'  # Green for active neurons
    inactive_color = '#ecf0f1'  # Light gray for inactive
    connection_color = '#3498db'  # Blue for connections
    connection_alpha = 0.3
    
    # Calculate vertical spacing for each layer to fit neurons
    max_neurons = max(layer_sizes)
    vertical_span = max_neurons * 0.4  # Total vertical space
    
    # Draw layers
    layer_positions = {}  # Store positions of neurons for drawing connections
    
    for i, (layer_name, layer_size) in enumerate(zip(layer_names, layer_sizes)):
        active_indices = set(circuit_data.get(layer_name, []))
        x = i * layer_spacing
        
        # Calculate vertical positions for this layer
        if layer_size <= 20:  # Show all neurons for smaller layers
            vertical_spacing = vertical_span / (layer_size + 1)
            y_positions = [vertical_span - (j + 1) * vertical_spacing for j in range(layer_size)]
            neurons_to_draw = range(layer_size)
        else:  # Sample neurons for larger layers
            # Always show active neurons + some context
            active_list = sorted(list(active_indices))
            inactive_sample = [j for j in range(layer_size) if j not in active_indices]
            
            # Sample some inactive neurons for context (max 15)
            num_context = min(15, len(inactive_sample))
            if num_context > 0:
                step = len(inactive_sample) / num_context
                context_indices = [inactive_sample[int(j * step)] for j in range(num_context)]
            else:
                context_indices = []
            
            neurons_to_draw = sorted(list(set(active_list + context_indices)))
            vertical_spacing = vertical_span / (len(neurons_to_draw) + 1)
            y_positions = [vertical_span - (j + 1) * vertical_spacing for j in range(len(neurons_to_draw))]
        
        layer_positions[layer_name] = {}
        
        # Draw neurons
        for idx, neuron_idx in enumerate(neurons_to_draw):
            y = y_positions[idx]
            layer_positions[layer_name][neuron_idx] = (x, y)
            
            is_active = neuron_idx in active_indices
            color = active_color if is_active else inactive_color
            alpha = 1.0 if is_active else 0.3
            
            circle = plt.Circle((x, y), neuron_radius, 
                              color=color, alpha=alpha, 
                              ec='black', linewidth=0.5 if is_active else 0.2,
                              zorder=3)
            ax.add_patch(circle)
            
            # Add neuron index label for active neurons
            if is_active:
                ax.text(x, y, str(neuron_idx), 
                       ha='center', va='center', 
                       fontsize=6, fontweight='bold',
                       zorder=4)
        
        # Add layer label
        ax.text(x, vertical_span + 0.5, layer_name, 
               ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
        ax.text(x, -0.5, f'{layer_size} neurons', 
               ha='center', va='top', 
               fontsize=8, color='gray')
    
    # Draw connections between consecutive layers
    for i in range(len(layer_names) - 1):
        current_layer = layer_names[i]
        next_layer = layer_names[i + 1]
        
        current_active = circuit_data.get(current_layer, [])
        next_active = circuit_data.get(next_layer, [])
        
        # Draw connections from all active neurons in current layer 
        # to all active neurons in next layer
        for curr_neuron in current_active:
            if curr_neuron in layer_positions[current_layer]:
                for next_neuron in next_active:
                    if next_neuron in layer_positions[next_layer]:
                        x1, y1 = layer_positions[current_layer][curr_neuron]
                        x2, y2 = layer_positions[next_layer][next_neuron]
                        
                        # Draw connection
                        arrow = FancyArrowPatch((x1 + neuron_radius, y1), 
                                              (x2 - neuron_radius, y2),
                                              arrowstyle='-',
                                              color=connection_color,
                                              alpha=connection_alpha,
                                              linewidth=0.5,
                                              zorder=1)
                        ax.add_patch(arrow)
    
    # Set axis properties
    ax.set_xlim(-0.5, len(layer_names) * layer_spacing - layer_spacing + 0.5)
    ax.set_ylim(-1, vertical_span + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

def create_all_visualizations(json_file, output_prefix='circuit_viz'):
    """
    Create visualizations for all clients and classes in round 10.
    
    Args:
        json_file: Path to the circuits JSON file
        output_prefix: Prefix for output files
    """
    # Load data
    data = load_circuits(json_file)
    
    # Get round 10 data
    round_10 = data['round_10']
    
    # Process both local and global models
    for model_type in ['clients_local_model', 'clients_global_model']:
        clients_data = round_10[model_type]
        num_clients = len(clients_data)
        
        # Get all classes (assuming same classes for all clients)
        sample_client = list(clients_data.values())[0]
        classes = list(sample_client.keys())
        num_classes = len(classes)
        
        # Create figure with subplots: rows = clients, cols = classes
        fig, axes = plt.subplots(num_clients, num_classes, 
                                figsize=(6 * num_classes, 5 * num_clients))
        
        # Ensure axes is 2D array even with single row/column
        if num_clients == 1 and num_classes == 1:
            axes = np.array([[axes]])
        elif num_clients == 1:
            axes = axes.reshape(1, -1)
        elif num_classes == 1:
            axes = axes.reshape(-1, 1)
        
        # Create visualizations
        for client_idx, (client_name, client_data) in enumerate(clients_data.items()):
            for class_idx, (class_name, circuit_data) in enumerate(client_data.items()):
                ax = axes[client_idx, class_idx]
                title = f"{client_name} - {class_name}"
                visualize_circuit(circuit_data, LAYER_SIZES, LAYER_NAMES, title, ax)
        
        # Adjust layout and save
        model_name = 'local' if 'local' in model_type else 'global'
        plt.tight_layout()
        output_file = f"{output_prefix}_round10_{model_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_file}")
        plt.close()

def create_single_client_visualizations(json_file, output_prefix='circuit_viz'):
    """
    Create separate visualizations for each client (one figure per client).
    Better for detailed examination.
    
    Args:
        json_file: Path to the circuits JSON file
        output_prefix: Prefix for output files
    """
    # Load data
    data = load_circuits(json_file)
    round_10 = data['round_10']
    
    # Process both local and global models
    for model_type in ['clients_local_model', 'clients_global_model']:
        clients_data = round_10[model_type]
        model_name = 'local' if 'local' in model_type else 'global'
        
        # Get all classes
        sample_client = list(clients_data.values())[0]
        classes = list(sample_client.keys())
        num_classes = len(classes)
        
        # Create one figure per client
        for client_name, client_data in clients_data.items():
            fig, axes = plt.subplots(1, num_classes, 
                                    figsize=(6 * num_classes, 6))
            
            # Ensure axes is array even with single subplot
            if num_classes == 1:
                axes = [axes]
            
            # Create visualizations for each class
            for class_idx, (class_name, circuit_data) in enumerate(client_data.items()):
                ax = axes[class_idx]
                title = f"{class_name}"
                visualize_circuit(circuit_data, LAYER_SIZES, LAYER_NAMES, title, ax)
            
            # Add main title
            fig.suptitle(f"{client_name} - {model_name.upper()} Model Circuits", 
                        fontsize=14, fontweight='bold', y=1.02)
            
            # Adjust layout and save
            plt.tight_layout()
            output_file = f"{output_prefix}_{client_name}_{model_name}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {output_file}")
            plt.close()

if __name__ == "__main__":
    json_file = "circuits_json_files/circuits_per_round.json"
    
    # Option 1: Create one large figure per model type (all clients & classes together)
    print("Creating combined visualizations...")
    create_all_visualizations(json_file)
    
    # Option 2: Create separate figures for each client (recommended for clarity)
    print("\nCreating individual client visualizations...")
    create_single_client_visualizations(json_file)
    
    print("\nVisualization complete!")