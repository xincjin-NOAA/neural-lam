"""
Example training script for HeteroObservationGraphModel using PyTorch Lightning.
"""

from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from .graph_dataset import GraphDataset
import pytorch_lightning as pl
import torch
from typing import Optional, Dict, List, Any
from torch_geometric.data import Batch
from torch.utils.data._utils.collate import default_collate


def collate_weather_batch(batch: List[Dict]) -> Dict:
    # Each batch item has:
    # - observations: Dict[str, Tuple[Tensor, Tensor]]
    # - obs_graphs: Dict[str, Dict[str, Data]]
    
    observations = {}
    obs_graphs = {}
    
    # Process each observation type (temp, pressure, etc)
    for obs_type in batch[0]['observations'].keys():
        # Stack locations and values from all batch items
        locations = torch.stack([item['observations'][obs_type][0] for item in batch])
        values = torch.stack([item['observations'][obs_type][1] for item in batch])
        observations[obs_type] = (locations, values)
        
        # Batch the PyG graphs
        o2m_graphs = [item['obs_graphs'][obs_type]['o2m'] for item in batch]
        m2o_graphs = [item['obs_graphs'][obs_type]['m2o'] for item in batch]
        obs_graphs[obs_type] = {
            'o2m': Batch.from_data_list(o2m_graphs),
            'm2o': Batch.from_data_list(m2o_graphs)
        }
    
    return {
        'observations': observations,
        'obs_graphs': obs_graphs
    }

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
            num_workers=self.args.num_workers if hasattr(self.args, 'num_workers') else 4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_weather_batch
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers if hasattr(self.args, 'num_workers') else 4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_weather_batch
        )
