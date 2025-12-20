import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

    model_name: str = "SimpleCNN"
    # MNIST is simpler, so we can potentially use fewer channels, 
    # but keeping [64, 128, 256] is fine too.
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128]) 
    num_classes: int = 10

    # --- CHANGED ---
    dataset_name: str = "MNIST" 
    # ---------------
    
    data_root: str = "./data"
    batch_size: int = 256
    num_workers: int = 0

    learning_rate: float = 0.001
    target_sparsity: float = 0.99

    num_clients: int = 5
    num_rounds: int = 50
    local_epochs: int = 5 # Good for dense/sparse convergence

    gate_lr: float = 0.1
    l0_lambda: float = 0.01
    discovery_steps: int = 200
    classes_to_analyze: List[int] = field(default_factory=lambda: [0, 1, 3])
    
    train_mode: str = "sparse" 

    checkpoint_dir: str = "./checkpoints"
    resume: bool = False 

