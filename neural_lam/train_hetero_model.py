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

from neural_lam.models import HeteroObservationGraphModel
from neural_lam.graph_dataset import GraphDataset
from neural_lam import utils

class WeatherLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = HeteroObservationGraphModel(args)
        self.criterion = nn.MSELoss()
        
    def forward(self, batch):
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        predictions = self(batch)
        loss = 0
        
        # Compute loss for each observation type
        for obs_type in predictions:
            target = batch[f"{obs_type}_target"]
            type_loss = self.criterion(predictions[obs_type], target)
            self.log(f"train_loss_{obs_type}", type_loss)
            loss += type_loss
            
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        predictions = self(batch)
        loss = 0
        
        for obs_type in predictions:
            target = batch[f"{obs_type}_target"]
            type_loss = self.criterion(predictions[obs_type], target)
            self.log(f"val_loss_{obs_type}", type_loss)
            loss += type_loss
            
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

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
    data_group.add_argument("--satellite_id", type=str, default="sat1",
                        help="Satellite ID for data selection")
    
    args = parser.parse_args()
    
    # Validate paths
    if args.data_path == "/path/to/your/data":
        print("WARNING: Using default data path. Please update --data_path to your actual data directory.")
    
    return args


class WeatherDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = None
        
    def setup(self, stage=None):
        if self.dataset is None:
            # Create dataset
            self.dataset = GraphDataset(
                data_path=self.args.data_path,
                start_date=self.args.start_date,
                end_date=self.args.end_date,
                satellite_id=self.args.satellite_id,
                mesh_resolution=self.args.mesh_resolution,
                cutoff_factor=self.args.cutoff_factor,
                num_neighbors=self.args.num_neighbors
            )
            self.dataset.setup()
            
            # Split dataset
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset, [train_size, val_size]
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        )

def main(override_args=None):
    # Get command line arguments
    args = parse_args()
    
    # Override with custom arguments if provided
    if override_args:
        for key, value in override_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"WARNING: Unknown argument '{key}' in override_args")
    
    # Convert devices to int if not 'auto'
    if args.devices != 'auto':
        args.devices = int(args.devices)
    
    # Initialize data module
    datamodule = WeatherDataModule(args)
    
    # Initialize model
    model = WeatherLightningModule(args)
    
    # Set graph structures from dataset
    datamodule.setup()
    model.model.set_graph_structures(datamodule.dataset)
    
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
    # Example 1: Using command line arguments
    if len(sys.argv) > 1:
        trainer, model, datamodule = main()
    
    # Example 2: Using programmatic overrides
    else:
        # Override default arguments
        custom_args = {
            "data_path": "/path/to/your/data",  # Update this path
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "satellite_id": "sat1",
            "batch_size": 32,
            "hidden_dim": 64,
            "strategy": "ddp",
            "devices": "auto",
            "precision": "16",
            # Add any other arguments you want to override
        }
        
        # Run with custom arguments
        trainer, model, datamodule = main(override_args=custom_args)
    
    # Train the model
    trainer.fit(model, datamodule)
