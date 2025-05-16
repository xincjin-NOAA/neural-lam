"""
Example training script for HeteroObservationGraphModel using PyTorch Lightning.
"""

from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from .graph_dataset import GraphDataset
import pytorch_lightning as pl


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
                observation_config=self.args.observation_config
            )
            self.dataset.setup()
            
            # # Split dataset
            # train_size = int(0.8 * len(self.dataset))
            # val_size = len(self.dataset) - train_size
            # self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            #     self.dataset, [train_size, val_size]
            # )
    
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
