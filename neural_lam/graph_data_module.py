"""
Example training script for HeteroObservationGraphModel using PyTorch Lightning.
"""

from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from .graph_dataset import GraphDataset
import pytorch_lightning as pl
import torch
from typing import Optional


class WeatherDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            # Create dataset for training/validation
            self.dataset = GraphDataset(
                data_path=self.args.data_path,
                start_date=self.args.start_date,
                end_date=self.args.end_date,
                observation_config=self.args.observation_config,
                mesh_structure=self.args.mesh_structure if hasattr(self.args, 'mesh_structure') else None,
                args=self.args,
                bin_size=self.args.bin_size if hasattr(self.args, 'bin_size') else '12h'
            )
            self.dataset.setup()
            
            # Split dataset for training and validation
            total_size = len(self.dataset)
            train_size = int(0.8 * total_size)
            val_size = total_size - train_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            
        if stage == 'test':
            # Create dataset for testing
            self.test_dataset = GraphDataset(
                data_path=self.args.test_data_path if hasattr(self.args, 'test_data_path') else self.args.data_path,
                start_date=self.args.test_start_date if hasattr(self.args, 'test_start_date') else self.args.start_date,
                end_date=self.args.test_end_date if hasattr(self.args, 'test_end_date') else self.args.end_date,
                observation_config=self.args.observation_config,
                mesh_structure=self.args.mesh_structure if hasattr(self.args, 'mesh_structure') else None,
                args=self.args,
                bin_size=self.args.bin_size if hasattr(self.args, 'bin_size') else '12h'
            )
            self.test_dataset.setup()
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers if hasattr(self.args, 'num_workers') else 4,
            persistent_workers=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers if hasattr(self.args, 'num_workers') else 4,
            persistent_workers=True,
            pin_memory=True
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers if hasattr(self.args, 'num_workers') else 4,
            persistent_workers=True,
            pin_memory=True
        )
