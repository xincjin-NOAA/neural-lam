"""
Example training script for HeteroObservationGraphModel using PyTorch Lightning.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from pathlib import Path

from .models.hetero_observation_model import HeteroObservationGraphModel
from .graph_dataset import GraphDataset
from .graph_data_module import WeatherDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train HeteroObservationGraphModel with multi-GPU support")
    
    # Model parameters
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension size")
    model_group.add_argument("--hidden_layers", type=int, default=3,
                        help="Number of hidden layers")
    model_group.add_argument("--mesh_resolution", type=float, default=2.0,
                        help="Resolution of the mesh")
    model_group.add_argument("--cutoff_factor", type=float, default=0.6,
                        help="Cutoff factor for mesh connections")
    model_group.add_argument("--num_neighbors", type=int, default=3,
                        help="Number of neighbors for KNN")
    
    # Training parameters
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per GPU")
    train_group.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum number of epochs")
    train_group.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    train_group.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    
    # Multi-GPU parameters
    gpu_group = parser.add_argument_group('Multi-GPU Configuration')
    gpu_group.add_argument("--strategy", type=str, default="ddp", 
                        choices=["ddp", "ddp_spawn", "deepspeed", "fsdp"],
                        help="Distributed training strategy")
    gpu_group.add_argument("--accelerator", type=str, default="gpu",
                        help="Accelerator type (gpu, cpu, tpu)")
    gpu_group.add_argument("--devices", type=str, default="auto",
                        help="Number of GPUs to use (int) or 'auto' for all available GPUs")
    gpu_group.add_argument("--precision", type=str, default="32",
                        choices=["32", "16", "bf16"],
                        help="Training precision")
    
    # Data parameters
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument("--data_path", type=str, default="/path/to/your/data",
                        help="Path to data directory")
    data_group.add_argument("--start_date", type=str, default="2023-01-01",
                        help="Start date for training data")
    data_group.add_argument("--end_date", type=str, default="2023-12-31",
                        help="End date for training data")
    data_group.add_argument("--observation_config", type=dict, 
                        help="observation_config for data selection")
    data_group.add_argument("--data_config", type=str, 
                        help="observation_config for data selection")

    #args = parser.parse_args()
    # Use parse_known_args() to avoid crashing in notebook environments
    args, unknown = parser.parse_known_args()
    
    # # Validate paths
    # if args.data_path == "/path/to/your/data":
    #     print("WARNING: Using default data path. Please update --data_path to your actual data directory.")
    
    return args


def main(override_args=None):
    # Get command line arguments
    args = parse_args()
    
    # Override with custom arguments if provided
    if override_args:
        for key, value in override_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"INFO: Adding new argument '{key}' to args")
                setattr(args, key, value)
    
    # Convert devices to int if not 'auto'
    if args.devices != 'auto':
        args.devices = int(args.devices)
    
    # Initialize data module
    datamodule = WeatherDataModule(args)
    
    # Initialize model
    model = HeteroObservationGraphModel(args)
    
    # Set graph structures from dataset
    datamodule.setup()
    model.set_graph_structures(datamodule.dataset)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='hetero-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Performance profiler
    profiler = pl.profilers.PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./lightning_logs/profiler'),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20)
    )
    
    # Setup logger with multi-GPU support
    logger = TensorBoardLogger(
        'lightning_logs',
        name='hetero_model',
        default_hp_metric=False
    )
    
    # Initialize trainer with multi-GPU settings
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        gradient_clip_val=0.5,
        profiler=profiler,
        sync_batchnorm=True,  # Important for multi-GPU training
        use_distributed_sampler=True,  # Handles data distribution
        num_sanity_val_steps=2,
        accumulate_grad_batches=1,  # Adjust if needed for larger effective batch size
        deterministic=True  # Ensures reproducibility across GPUs
    )
    
    return trainer, model, datamodule

if __name__ == "__main__":
    # CONUS data path:
    data_path = "/scratch1/NCEPDEV/da/Ronald.McLaren/shared/ocelot/data_v2/"
    
    # Observation configuration, will move to a config file later.
    observation_config = {
        "satellite": {
            'atms': {
                "sat_ids": [224, 225],
                "features": [f"bt_channel_{i}" for i in range(1, 23)],
                "metadata": ["sensorZenithAngle", "solarZenithAngle", "solarAzimuthAngle"]
            },
            # "iasi": ,
            # "goes":,
            # "ascat":
        },
        "conventional": {
            # "radiosonde": ,
            "pressure": {
                "features": ["height", "stationPressure"]
            },
            # "surface_marine": ,
            # "surface_land":
        }
    }

    args_dict = {
        "data_path": data_path,
        "observation_config": observation_config,
        "start_date":  "2024-04-01",
        "end_date":  "2024-04-04",
        "levels": 4,
        "plot": False,
        "hierarchical": True,  # The last assignment overrides the previous one
        "data_config": "/scratch1/NCEPDEV/da/Xin.C.Jin/my_projects/neural_lam/scripts/data_config_15km.yaml",
        "output_std": False,
        "loss": "MSE",
        "restore_opt": False,
        "n_example_pred": 1,
        "lr": 0.0001,
        "graph": "hetero",
        "hidden_dim": 64,
        "hidden_layers": 16,
        "mesh_resolution": 4,
        "cutoff_factor": 0.67,
        "num_neighbors": 2,
        "step_length": 6
    } 
    
    # Run with custom arguments
    trainer, model, datamodule = main(override_args=args_dict)
    
    # Train the model
    trainer.fit(model, datamodule)
