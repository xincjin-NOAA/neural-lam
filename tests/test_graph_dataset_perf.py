"""Performance benchmarking tests for GraphDataset"""

import pytest
import numpy as np
import torch
import zarr
import time
import psutil
import os
from pathlib import Path
from datetime import datetime

from neural_lam.graph_dataset import GraphDataset

class TestGraphDatasetPerformance:
    @pytest.fixture
    def large_data_path(self, tmp_path):
        """Create a large sample Zarr dataset"""
        data_path = tmp_path / "large_test_data.zarr"
        
        # Create sample data
        root = zarr.open(str(data_path), mode='w')
        
        # Add time dimension (24 hours)
        times = np.array([f'2025-01-01T{h:02d}:00:00' for h in range(24)], 
                        dtype='datetime64[s]')
        root.create_dataset('time', data=times)
        
        # Add sample weather data with larger dimensions
        shape = (24, 100, 100)  # time, lat, lon
        root.create_dataset('temperature', data=np.random.rand(*shape))
        root.create_dataset('pressure', data=np.random.rand(*shape))
        root.create_dataset('wind', data=np.random.rand(24, 100, 100, 2))
        
        return data_path
    
    @pytest.fixture
    def dataset_config(self):
        """Sample dataset configuration"""
        return {
            'mesh_resolution': 0.1,
            'cutoff_factor': 0.67,
            'num_neighbors': 4,
            'start_date': '2025-01-01',
            'end_date': '2025-01-02',
            'satellite_id': 'test_sat'
        }
    
    def measure_memory(self, func):
        """Measure peak memory usage of a function"""
        process = psutil.Process()
        mem_before = process.memory_info().rss
        result = func()
        mem_after = process.memory_info().rss
        peak_memory = (mem_after - mem_before) / 1024 / 1024  # MB
        return result, peak_memory
    
    def test_mesh_creation_scaling(self, large_data_path, dataset_config):
        """Test mesh creation performance with different sizes"""
        dataset = GraphDataset(
            data_path=str(large_data_path),
            save_path=str(Path(large_data_path).parent),
            **dataset_config
        )
        
        sizes = [10, 20, 50, 100]
        timings = []
        memory_usage = []
        
        for size in sizes:
            grid_coords = np.meshgrid(
                np.linspace(0, 1, size),
                np.linspace(0, 1, size)
            )
            
            def create_mesh():
                return dataset.create_mesh_structure(
                    xy=grid_coords,
                    args={'cutoff': dataset.cutoff_factor, 'num_neighbors': dataset.num_neighbors},
                    graph_dir_path=dataset.save_path
                )
            
            # Measure time and memory
            start_time = time.time()
            result, peak_mem = self.measure_memory(create_mesh)
            end_time = time.time()
            
            timings.append(end_time - start_time)
            memory_usage.append(peak_mem)
            
            # Verify structure size
            assert result['g2m_graph'].num_nodes == size * size
        
        # Check scaling behavior
        # Time should scale approximately quadratically (O(n²))
        scaling_factor = np.polyfit(np.log(sizes), np.log(timings), 1)[0]
        assert 1.5 < scaling_factor < 2.5, f"Unexpected time scaling: {scaling_factor}"
    
    def test_dynamic_graph_performance(self, large_data_path, dataset_config):
        """Test dynamic graph creation performance"""
        dataset = GraphDataset(
            data_path=str(large_data_path),
            save_path=str(Path(large_data_path).parent),
            **dataset_config
        )
        dataset.setup()
        
        batch_sizes = [1, 4, 8, 16]
        timings = []
        memory_usage = []
        
        for batch_size in batch_sizes:
            # Create batched weather data
            weather_data = {
                'temperature': (
                    torch.randn(batch_size, 100, 100),
                    torch.ones(batch_size, 100, 100)
                ),
                'wind': (
                    torch.randn(batch_size, 100, 100, 2),
                    torch.ones(batch_size, 100, 100)
                )
            }
            
            def create_graph():
                return dataset.create_dynamic_graph(weather_data)
            
            # Measure time and memory
            start_time = time.time()
            result, peak_mem = self.measure_memory(create_graph)
            end_time = time.time()
            
            timings.append(end_time - start_time)
            memory_usage.append(peak_mem)
            
            # Memory should scale roughly linearly with batch size
            if batch_size > 1:
                mem_ratio = memory_usage[-1] / memory_usage[0]
                expected_ratio = batch_size
                assert 0.5 * expected_ratio < mem_ratio < 1.5 * expected_ratio
    
    def test_cache_performance(self, large_data_path, dataset_config):
        """Test caching performance"""
        dataset = GraphDataset(
            data_path=str(large_data_path),
            save_path=str(Path(large_data_path).parent),
            **dataset_config
        )
        
        grid_coords = np.meshgrid(
            np.linspace(0, 1, 50),
            np.linspace(0, 1, 50)
        )
        
        # First creation (no cache)
        start_time = time.time()
        dataset.create_mesh_structure(
            xy=grid_coords,
            args={'cutoff': dataset.cutoff_factor, 'num_neighbors': dataset.num_neighbors},
            graph_dir_path=dataset.save_path
        )
        first_creation_time = time.time() - start_time
        
        # Second creation (with cache)
        start_time = time.time()
        dataset.create_mesh_structure(
            xy=grid_coords,
            args={'cutoff': dataset.cutoff_factor, 'num_neighbors': dataset.num_neighbors},
            graph_dir_path=dataset.save_path
        )
        cached_creation_time = time.time() - start_time
        
        # Cache should be significantly faster
        assert cached_creation_time < 0.1 * first_creation_time
