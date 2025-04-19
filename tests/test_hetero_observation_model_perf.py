"""Performance benchmark tests for HeteroObservationModel"""

import pytest
import torch
import numpy as np
import time
import psutil
import os
from torch_geometric.data import Data, Batch
from neural_lam.models.hetero_observation_model import HeteroObservationGraphModel

class TestHeteroObservationModelPerformance:
    @pytest.fixture
    def model_args(self):
        """Model configuration for benchmarking"""
        class Args:
            def __init__(self):
                self.hidden_dim = 64
                self.hidden_layers = 3
                self.observation_types = {
                    'temperature': {'dim': 1},
                    'wind': {'dim': 2},
                    'pressure': {'dim': 1}
                }
        return Args()
    
    def create_benchmark_data(self, n_mesh_nodes: int, n_obs_points: int, batch_size: int = 1):
        """Create synthetic data for benchmarking"""
        # Create mesh structure
        mesh_pos = torch.rand((n_mesh_nodes, 2))
        mesh_edges = torch.randint(0, n_mesh_nodes, (2, n_mesh_nodes * 4))  # 4 connections per node
        
        data = Data()
        data.mesh_structure = Data(pos=mesh_pos, edge_index=mesh_edges)
        
        # Add observation data
        for obs_type in ['temperature', 'wind', 'pressure']:
            # Create observation locations and values
            locations = torch.rand((batch_size, n_obs_points, 2))
            if obs_type == 'wind':
                values = torch.rand((batch_size, n_obs_points, 2))
            else:
                values = torch.rand((batch_size, n_obs_points, 1))
            
            # Create graph mappings (3 connections per observation)
            g2m_edges = torch.randint(0, n_obs_points, (2, n_obs_points * 3))
            m2g_edges = torch.randint(0, n_mesh_nodes, (2, n_obs_points * 3))
            
            data[f"{obs_type}_locations"] = locations
            data[f"{obs_type}_values"] = values
            data[f"{obs_type}_g2m_graph"] = Data(edge_index=g2m_edges)
            data[f"{obs_type}_m2g_graph"] = Data(edge_index=m2g_edges)
            data[f"{obs_type}_edge_features"] = {
                'weights': torch.rand(g2m_edges.size(1)),
                'attention': torch.rand(g2m_edges.size(1))
            }
        
        return data
    
    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("n_mesh_nodes", [100, 1000, 10000])
    def test_mesh_scaling(self, model_args, n_mesh_nodes):
        """Test performance scaling with mesh size"""
        model = HeteroObservationGraphModel(model_args)
        n_obs_points = 100  # Fixed number of observations
        
        # Measure initialization memory
        init_memory = self.get_memory_usage()
        
        # Create test data
        data = self.create_benchmark_data(n_mesh_nodes, n_obs_points)
        
        # Warmup
        model(data)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Measure inference time
        start_time = time.time()
        for _ in range(10):  # Average over 10 runs
            model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        avg_time = (time.time() - start_time) / 10
        
        # Measure peak memory
        peak_memory = self.get_memory_usage() - init_memory
        
        print(f"\nMesh Scaling (nodes={n_mesh_nodes}):")
        print(f"Average inference time: {avg_time*1000:.2f}ms")
        print(f"Peak memory usage: {peak_memory:.2f}MB")
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("n_obs_points", [100, 1000, 10000])
    def test_observation_scaling(self, model_args, n_obs_points):
        """Test performance scaling with number of observations"""
        model = HeteroObservationGraphModel(model_args)
        n_mesh_nodes = 1000  # Fixed mesh size
        
        init_memory = self.get_memory_usage()
        data = self.create_benchmark_data(n_mesh_nodes, n_obs_points)
        
        # Warmup
        model(data)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Measure inference time
        start_time = time.time()
        for _ in range(10):
            model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        avg_time = (time.time() - start_time) / 10
        
        peak_memory = self.get_memory_usage() - init_memory
        
        print(f"\nObservation Scaling (points={n_obs_points}):")
        print(f"Average inference time: {avg_time*1000:.2f}ms")
        print(f"Peak memory usage: {peak_memory:.2f}MB")
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
    def test_batch_scaling(self, model_args, batch_size):
        """Test performance scaling with batch size"""
        model = HeteroObservationGraphModel(model_args)
        n_mesh_nodes = 1000
        n_obs_points = 1000
        
        init_memory = self.get_memory_usage()
        data = self.create_benchmark_data(n_mesh_nodes, n_obs_points, batch_size)
        
        # Warmup
        model(data)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Measure inference time
        start_time = time.time()
        for _ in range(10):
            model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        avg_time = (time.time() - start_time) / 10
        
        peak_memory = self.get_memory_usage() - init_memory
        
        print(f"\nBatch Scaling (size={batch_size}):")
        print(f"Average inference time: {avg_time*1000:.2f}ms")
        print(f"Peak memory usage: {peak_memory:.2f}MB")
        print(f"Memory per sample: {peak_memory/batch_size:.2f}MB")
    
    @pytest.mark.benchmark
    def test_observation_type_impact(self, model_args):
        """Test performance impact of different observation types"""
        n_mesh_nodes = 1000
        n_obs_points = 1000
        batch_size = 4
        
        # Test each observation type individually
        for obs_type in ['temperature', 'wind', 'pressure']:
            # Create model with single observation type
            single_type_args = model_args
            single_type_args.observation_types = {obs_type: model_args.observation_types[obs_type]}
            model = HeteroObservationGraphModel(single_type_args)
            
            init_memory = self.get_memory_usage()
            data = self.create_benchmark_data(n_mesh_nodes, n_obs_points, batch_size)
            
            # Warmup
            model(data)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            # Measure inference time
            start_time = time.time()
            for _ in range(10):
                model(data)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            avg_time = (time.time() - start_time) / 10
            
            peak_memory = self.get_memory_usage() - init_memory
            
            print(f"\nObservation Type: {obs_type}")
            print(f"Average inference time: {avg_time*1000:.2f}ms")
            print(f"Peak memory usage: {peak_memory:.2f}MB")
    
    @pytest.mark.benchmark
    def test_gpu_vs_cpu(self, model_args):
        """Compare performance between GPU and CPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        n_mesh_nodes = 1000
        n_obs_points = 1000
        batch_size = 4
        
        # Create model and data
        model = HeteroObservationGraphModel(model_args)
        data = self.create_benchmark_data(n_mesh_nodes, n_obs_points, batch_size)
        
        # Test on CPU
        model.cpu()
        data = data.cpu()
        
        # Warmup
        model(data)
        
        # Measure CPU time
        start_time = time.time()
        for _ in range(5):
            model(data)
        cpu_time = (time.time() - start_time) / 5
        
        # Test on GPU
        model.cuda()
        data = data.cuda()
        
        # Warmup
        model(data)
        torch.cuda.synchronize()
        
        # Measure GPU time
        start_time = time.time()
        for _ in range(5):
            model(data)
            torch.cuda.synchronize()
        gpu_time = (time.time() - start_time) / 5
        
        print("\nCPU vs GPU Performance:")
        print(f"CPU average time: {cpu_time*1000:.2f}ms")
        print(f"GPU average time: {gpu_time*1000:.2f}ms")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
