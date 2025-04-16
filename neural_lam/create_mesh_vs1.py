# Standard library
import os
from argparse import ArgumentParser

# Third-party
import matplotlib
import matplotlib.pyplot as plt
import networkx
import numpy as np
import scipy.spatial
import torch
import torch_geometric as pyg
from torch_geometric.utils.convert import from_networkx

# Local
from . import config


def plot_graph(graph, title=None):
    fig, axis = plt.subplots(figsize=(8, 8), dpi=200)  # W,H
    edge_index = graph.edge_index
    pos = graph.pos

    # Fix for re-indexed edge indices only containing mesh nodes at
    # higher levels in hierarchy
    edge_index = edge_index - edge_index.min()

    if pyg.utils.is_undirected(edge_index):
        # Keep only 1 direction of edge_index
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # (2, M/2)
    # TODO: indicate direction of directed edges

    # Move all to cpu and numpy, compute (in)-degrees
    degrees = (
        pyg.utils.degree(edge_index[1], num_nodes=pos.shape[0]).cpu().numpy()
    )
    edge_index = edge_index.cpu().numpy()
    pos = pos.cpu().numpy()

    # Plot edges
    from_pos = pos[edge_index[0]]  # (M/2, 2)
    to_pos = pos[edge_index[1]]  # (M/2, 2)
    edge_lines = np.stack((from_pos, to_pos), axis=1)
    axis.add_collection(
        matplotlib.collections.LineCollection(
            edge_lines, lw=0.4, colors="black", zorder=1
        )
    )

    # Plot nodes
    node_scatter = axis.scatter(
        pos[:, 0],
        pos[:, 1],
        c=degrees,
        s=3,
        marker="o",
        zorder=2,
        cmap="viridis",
        clim=None,
    )

    plt.colorbar(node_scatter, aspect=50)

    if title is not None:
        axis.set_title(title)

    return fig, axis


def sort_nodes_internally(nx_graph):
    # For some reason the networkx .nodes() return list can not be sorted,
    # but this is the ordering used by pyg when converting.
    # This function fixes this.
    H = networkx.DiGraph()
    H.add_nodes_from(sorted(nx_graph.nodes(data=True)))
    H.add_edges_from(nx_graph.edges(data=True))
    return H


def save_edges(graph, name, base_path):
    torch.save(
        graph.edge_index, os.path.join(base_path, f"{name}_edge_index.pt")
    )
    edge_features = torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
        torch.float32
    )  # Save as float32
    torch.save(edge_features, os.path.join(base_path, f"{name}_features.pt"))


def save_edges_list(graphs, name, base_path):
    torch.save(
        [graph.edge_index for graph in graphs],
        os.path.join(base_path, f"{name}_edge_index.pt"),
    )
    edge_features = [
        torch.cat((graph.len.unsqueeze(1), graph.vdiff), dim=1).to(
            torch.float32
        )
        for graph in graphs
    ]  # Save as float32
    torch.save(edge_features, os.path.join(base_path, f"{name}_features.pt"))


def from_networkx_with_start_index(nx_graph, start_index):
    pyg_graph = from_networkx(nx_graph)
    pyg_graph.edge_index += start_index
    return pyg_graph


def mk_2d_graph(xy, nx, ny):
    xm, xM = np.amin(xy[0][0, :]), np.amax(xy[0][0, :])
    ym, yM = np.amin(xy[1][:, 0]), np.amax(xy[1][:, 0])

    # avoid nodes on border
    dx = (xM - xm) / nx
    dy = (yM - ym) / ny
    lx = np.linspace(xm + dx / 2, xM - dx / 2, nx)
    ly = np.linspace(ym + dy / 2, yM - dy / 2, ny)

    mg = np.meshgrid(lx, ly)
    g = networkx.grid_2d_graph(len(ly), len(lx))

    for node in g.nodes:
        g.nodes[node]["pos"] = np.array([mg[0][node], mg[1][node]])

    # add diagonal edges
    g.add_edges_from(
        [((x, y), (x + 1, y + 1)) for x in range(nx - 1) for y in range(ny - 1)]
        + [
            ((x + 1, y), (x, y + 1))
            for x in range(nx - 1)
            for y in range(ny - 1)
        ]
    )

    # turn into directed graph
    dg = networkx.DiGraph(g)
    for u, v in g.edges():
        d = np.sqrt(np.sum((g.nodes[u]["pos"] - g.nodes[v]["pos"]) ** 2))
        dg.edges[u, v]["len"] = d
        dg.edges[u, v]["vdiff"] = g.nodes[u]["pos"] - g.nodes[v]["pos"]
        dg.add_edge(v, u)
        dg.edges[v, u]["len"] = d
        dg.edges[v, u]["vdiff"] = g.nodes[v]["pos"] - g.nodes[u]["pos"]

    return dg


def prepend_node_index(graph, new_index):
    # Relabel node indices in graph, insert (graph_level, i, j)
    ijk = [tuple((new_index,) + x) for x in graph.nodes]
    to_mapping = dict(zip(graph.nodes, ijk))
    return networkx.relabel_nodes(graph, to_mapping, copy=True)

def create_mesh_structure(self, xy, args, graph_dir_path):
    """
    Create multi-resolution mesh structure with optional hierarchical organization.
    
    Args:
        xy: Grid coordinates array
        args: Arguments containing:
            - levels: Maximum number of mesh levels
            - hierarchical: Whether to create hierarchical mesh
            - plot: Whether to plot graphs
        graph_dir_path: Path to save graph data
    
    Returns:
        dict: Contains mesh graphs, positions, and related data
    """
    import networkx as nx
    import numpy as np
    import torch
    from scipy import spatial
    
    # Graph geometry parameters
    nx = 3  # number of children = nx**2
    nlev = int(np.log(max(xy.shape)) / np.log(nx))
    nleaf = nx**nlev  # leaves at the bottom = nleaf**2
    
    # Determine mesh levels
    mesh_levels = nlev - 1
    if args.levels:
        mesh_levels = min(mesh_levels, args.levels)
    
    print(f"nlev: {nlev}, nleaf: {nleaf}, mesh_levels: {mesh_levels}")
    
    # Create multi-resolution tree levels
    G = []
    for lev in range(1, mesh_levels + 1):
        n = int(nleaf / (nx**lev))
        g = self.mk_2d_graph(xy, n, n)
        if args.plot:
            self.plot_graph(from_networkx(g), title=f"Mesh graph, level {lev}")
            plt.show()
        G.append(g)
    
    if args.hierarchical:
        return self._create_hierarchical_mesh(G, mesh_levels, graph_dir_path, args)
    else:
        return self._create_flat_mesh(G, nx, graph_dir_path, args)

def _create_hierarchical_mesh(self, G, mesh_levels, graph_dir_path, args):
    """Create hierarchical mesh structure with inter-level connections."""
    # Relabel nodes with level index
    G = [self.prepend_node_index(graph, level_i) 
         for level_i, graph in enumerate(G)]
    
    # Calculate level indices
    num_nodes_level = np.array([len(g_level.nodes) for g_level in G])
    first_index_level = np.concatenate(
        (np.zeros(1, dtype=int), np.cumsum(num_nodes_level[:-1]))
    )
    
    # Create inter-level connections
    up_graphs, down_graphs = self._create_interlevel_connections(
        G, mesh_levels, first_index_level, args
    )
    
    # Save up and down edges
    self.save_edges_list(up_graphs, "mesh_up", graph_dir_path)
    self.save_edges_list(down_graphs, "mesh_down", graph_dir_path)
    
    # Create m2m graphs
    m2m_graphs = [
        self.from_networkx_with_start_index(
            nx.convert_node_labels_to_integers(
                level_graph, first_label=start_index, ordering="sorted"
            ),
            start_index,
        )
        for level_graph, start_index in zip(G, first_index_level)
    ]
    
    mesh_pos = [graph.pos.to(torch.float32) for graph in m2m_graphs]
    
    # Create combined mesh structure
    G_bottom_mesh = G[0]
    joint_mesh_graph = nx.union_all([graph for graph in G])
    
    return {
        'm2m_graphs': m2m_graphs,
        'mesh_pos': mesh_pos,
        'G_bottom_mesh': G_bottom_mesh,
        'all_mesh_nodes': joint_mesh_graph.nodes(data=True)
    }

def _create_flat_mesh(self, G, nx, graph_dir_path, args):
    """Create flat mesh structure combining all levels."""
    G_tot = G[0]
    
    # Combine all levels
    for lev in range(1, len(G)):
        nodes = list(G[lev - 1].nodes)
        n = int(np.sqrt(len(nodes)))
        ij = (
            np.array(nodes)
            .reshape((n, n, 2))[1::nx, 1::nx, :]
            .reshape(int(n / nx) ** 2, 2)
        )
        ij = [tuple(x) for x in ij]
        G[lev] = nx.relabel_nodes(G[lev], dict(zip(G[lev].nodes, ij)))
        G_tot = nx.compose(G_tot, G[lev])
    
    # Relabel and convert to integers
    G_tot = self.prepend_node_index(G_tot, 0)
    G_int = nx.convert_node_labels_to_integers(
        G_tot, first_label=0, ordering="sorted"
    )
    
    # Create PyG graph
    pyg_m2m = from_networkx(G_int)
    m2m_graphs = [pyg_m2m]
    mesh_pos = [pyg_m2m.pos.to(torch.float32)]
    
    if args.plot:
        self.plot_graph(pyg_m2m, title="Mesh-to-mesh")
        plt.show()
    
    return {
        'm2m_graphs': m2m_graphs,
        'mesh_pos': mesh_pos,
        'G_bottom_mesh': G_tot,
        'all_mesh_nodes': G_tot.nodes(data=True)
    }

def _create_interlevel_connections(self, G, mesh_levels, first_index_level, args):
    """Create connections between different mesh levels."""
    up_graphs = []
    down_graphs = []
    
    for from_level, to_level, G_from, G_to, start_index in zip(
        range(1, mesh_levels),
        range(0, mesh_levels - 1),
        G[1:],
        G[:-1],
        first_index_level[: mesh_levels - 1],
    ):
        # Create downward connections
        G_down = self._create_level_connections(
            G_from, G_to, start_index
        )
        
        # Create upward connections by inverting downward edges
        up_edges = torch.stack(
            (G_down.edge_index[1], G_down.edge_index[0]), dim=0
        )
        pyg_up = G_down.clone()
        pyg_up.edge_index = up_edges
        
        up_graphs.append(pyg_up)
        down_graphs.append(G_down)
        
        if args.plot:
            self.plot_graphs(G_down, pyg_up, from_level, to_level)
    
    return up_graphs, down_graphs


    """
    Create Grid-to-Mesh (g2m) graph structure.
    
    Args:
        xy: Grid coordinates array of shape [2, height, width]
        G_bottom_mesh: Bottom level mesh graph
        all_mesh_nodes: All mesh nodes with data
        args: Arguments containing plot option
    
    Returns:
        dict: Contains:
            - g2m_graph: PyTorch Geometric graph for grid-to-mesh
            - grid_graph: Base grid graph
            - mesh_distance: Distance between mesh nodes
    """
    import networkx as nx
    import numpy as np
    from scipy.spatial import KDTree
    
    # Constants
    DM_SCALE = 0.67  # radius scale for grid-mesh association
    
    def _calculate_mesh_distance(vm):
        """Calculate distance between mesh nodes."""
        return np.sqrt(
            np.sum((vm.data("pos")[(0, 1, 0)] - vm.data("pos")[(0, 0, 0)]) ** 2)
        )
    
    def _create_base_grid(xy, Ny, Nx):
        """Create and initialize base grid graph."""
        G_grid = nx.grid_2d_graph(Ny, Nx)
        G_grid.clear_edges()
        
        # Add position features to grid nodes
        for node in G_grid.nodes:
            G_grid.nodes[node]["pos"] = np.array([xy[0][node], xy[1][node]])
        
        # Separate grid nodes from mesh nodes with index 1000
        return self.prepend_node_index(G_grid, 1000)
    
    def _build_kd_tree(G_grid, xy):
        """Build KD-tree for grid points."""
        vg_list = list(G_grid.nodes)
        vg_xy = np.array([[xy[0][node[1:]], xy[1][node[1:]]] 
                         for node in vg_list])
        return vg_list, vg_xy, KDTree(vg_xy)
    
    def _create_g2m_edges(G_g2m, vm, vg_list, kdt_g, dm):
        """Create edges from grid to mesh nodes."""
        for v in vm:
            # Find neighbors within radius
            neigh_idxs = kdt_g.query_ball_point(vm[v]["pos"], dm * DM_SCALE)
            
            for i in neigh_idxs:
                u = vg_list[i]
                # Add edge from grid to mesh
                G_g2m.add_edge(u, v)
                
                # Calculate edge properties
                d = np.sqrt(
                    np.sum((G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]) ** 2)
                )
                G_g2m.edges[u, v]["len"] = d
                G_g2m.edges[u, v]["vdiff"] = (
                    G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]
                )
        return G_g2m
    
    # Main process
    try:
        # 1. Get mesh nodes and their positions
        vm = G_bottom_mesh.nodes
        vm_xy = np.array([xy for _, xy in vm.data("pos")])
        dm = _calculate_mesh_distance(vm)
        
        # 2. Create grid
        Ny, Nx = xy.shape[1:]
        G_grid = _create_base_grid(xy, Ny, Nx)
        
        # 3. Build KD-tree for grid points
        vg_list, vg_xy, kdt_g = _build_kd_tree(G_grid, xy)
        
        # 4. Add mesh nodes to grid
        G_grid.add_nodes_from(all_mesh_nodes)
        
        # 5. Create g2m graph with sorted nodes
        G_g2m = nx.Graph()
        G_g2m.add_nodes_from(sorted(G_grid.nodes(data=True)))
        G_g2m = nx.DiGraph(G_g2m)
        
        # 6. Add edges
        G_g2m = _create_g2m_edges(G_g2m, vm, vg_list, kdt_g, dm)
        
        # 7. Convert to PyTorch Geometric
        pyg_g2m = from_networkx(G_g2m)
        
        # 8. Optional plotting
        if args.plot:
            self.plot_graph(pyg_g2m, title="Grid-to-mesh")
            plt.show()
        
        return {
            'g2m_graph': pyg_g2m,
            'grid_graph': G_grid,
            'mesh_distance': dm
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to create grid-to-mesh graph: {e}")

        # After creating g2m graph
        m2g_data = self.create_mesh_to_grid(
            G_g2m=g2m_data['g2m_graph'],
            vm=mesh_data['G_bottom_mesh'].nodes,
            vm_xy=mesh_positions,
            vg_list=grid_nodes,
            args=args,
            graph_dir_path=save_path
        )
        
        self.m2g_graph = m2g_data['m2g_graph']
        self.edge_features = m2g_data['edge_features']


    """
    Create Mesh-to-Grid (m2g) graph structure.
    
    Args:
        G_g2m: Grid-to-Mesh graph
        vm: Mesh nodes
        vm_xy: Mesh node positions array
        vg_list: List of grid nodes
        args: Arguments containing plot option
        graph_dir_path: Path to save graph data
    
    Returns:
        dict: Contains:
            - m2g_graph: PyTorch Geometric graph for mesh-to-grid
            - edge_indices: Edge indices for the graph
            - edge_features: Edge features (length and vector differences)
    """
    import networkx as nx
    import numpy as np
    from scipy.spatial import KDTree
    
    def _create_m2g_edges(G_m2g, vm_list, kdt_m, num_neighbors=4):
        """Create edges from mesh to grid nodes."""
        for v in vg_list:
            # Find k nearest neighbors
            distances, neigh_idxs = kdt_m.query(G_m2g.nodes[v]["pos"], 
                                              k=num_neighbors)
            
            for i in neigh_idxs:
                u = vm_list[i]
                # Add edge from mesh to grid
                G_m2g.add_edge(u, v)
                
                # Calculate edge properties
                d = np.sqrt(
                    np.sum((G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]) ** 2)
                )
                G_m2g.edges[u, v]["len"] = d
                G_m2g.edges[u, v]["vdiff"] = (
                    G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]
                )
        return G_m2g
    
    try:
        # 1. Create base m2g graph from g2m
        G_m2g = G_g2m.copy()
        G_m2g.clear_edges()
        
        # 2. Build KD-tree for mesh points
        vm_list = list(vm)
        kdt_m = KDTree(vm_xy)
        
        # 3. Create edges from mesh to grid
        G_m2g = _create_m2g_edges(G_m2g, vm_list, kdt_m)
        
        # 4. Convert to integers with sorted labels
        G_m2g_int = nx.convert_node_labels_to_integers(
            G_m2g, first_label=0, ordering="sorted"
        )
        
        # 5. Convert to PyTorch Geometric
        pyg_m2g = from_networkx(G_m2g_int)
        
        # 6. Optional plotting
        if args.plot:
            self.plot_graph(pyg_m2g, title="Mesh-to-grid")
            plt.show()
        
        # 7. Save edge data
        self.save_edges(pyg_m2g, "m2g", graph_dir_path)
        
        return {
            'm2g_graph': pyg_m2g,
            'edge_indices': pyg_m2g.edge_index,
            'edge_features': {
                'length': torch.tensor([d['len'] for _, _, d in G_m2g.edges(data=True)]),
                'vector_diff': torch.tensor([d['vdiff'] for _, _, d in G_m2g.edges(data=True)])
            }
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to create mesh-to-grid graph: {e}")


    """
    Create Grid-to-Mesh (g2m) graph structure using Lambert Conformal projection coordinates.
    
    Args:
        coords: Array of shape [N, 2] containing [x, y] pairs in Lambert projection
               or dictionary with {'x': x_array, 'y': y_array}
        G_bottom_mesh: Bottom level mesh graph
        all_mesh_nodes: All mesh nodes with data
        args: Arguments containing plot option
        proj_params: Optional dictionary containing Lambert projection parameters:
                    - lat_1: First standard parallel
                    - lat_2: Second standard parallel
                    - lat_0: Latitude of origin
                    - lon_0: Central meridian
                    - earth_radius: Earth radius in meters (default: 6371000)
    
    Returns:
        dict: Contains:
            - g2m_graph: PyTorch Geometric graph for grid-to-mesh
            - grid_graph: Base grid graph
            - mesh_distance: Distance between mesh nodes
    """
    import networkx as nx
    import numpy as np
    from scipy.spatial import KDTree
    import pyproj
    
    # Constants
    DM_SCALE = 0.67  # radius scale for grid-mesh association
    
    def _setup_projection(params=None):
        """Setup Lambert Conformal projection."""
        default_params = {
            'lat_1': 30.0,
            'lat_2': 60.0,
            'lat_0': 40.0,
            'lon_0': -97.0,
            'earth_radius': 6371000
        }
        
        if params is not None:
            default_params.update(params)
            
        return pyproj.Proj(
            proj='lcc',
            lat_1=default_params['lat_1'],
            lat_2=default_params['lat_2'],
            lat_0=default_params['lat_0'],
            lon_0=default_params['lon_0'],
            R=default_params['earth_radius']
        )
    
    def _euclidean_distance(p1, p2):
        """Calculate Euclidean distance in projected space."""
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    def _calculate_mesh_distance(vm):
        """Calculate distance between mesh nodes in projected space."""
        pos1 = vm.data("pos")[(0, 1, 0)]
        pos2 = vm.data("pos")[(0, 0, 0)]
        return _euclidean_distance(pos1, pos2)
    
    def _create_base_grid(coords):
        """Create and initialize base grid graph from projected coordinates."""
        if isinstance(coords, dict):
            x, y = coords['x'], coords['y']
            grid_points = np.column_stack((x.flatten(), y.flatten()))
        else:
            grid_points = coords
            
        G_grid = nx.Graph()
        
        # Add nodes with positions
        for i, (x, y) in enumerate(grid_points):
            G_grid.add_node(i, pos=np.array([x, y]))
        
        return self.prepend_node_index(G_grid, 1000)
    
    def _build_kd_tree(G_grid):
        """Build KD-tree for grid points using projected coordinates."""
        vg_list = list(G_grid.nodes)
        vg_coords = np.array([G_grid.nodes[n]['pos'] for n in vg_list])
        return vg_list, vg_coords, KDTree(vg_coords)
    
    def _create_g2m_edges(G_g2m, vm, vg_list, kdt_g, dm):
        """Create edges from grid to mesh nodes using Euclidean distance in projected space."""
        for v in vm:
            v_pos = vm[v]["pos"]
            # Find neighbors within radius
            neigh_idxs = kdt_g.query_ball_point(v_pos, dm * DM_SCALE)
            
            for i in neigh_idxs:
                u = vg_list[i]
                # Add edge from grid to mesh
                G_g2m.add_edge(u, v)
                
                # Calculate edge properties in projected space
                u_pos = G_g2m.nodes[u]["pos"]
                d = _euclidean_distance(u_pos, v_pos)
                G_g2m.edges[u, v]["len"] = d
                G_g2m.edges[u, v]["vdiff"] = v_pos - u_pos
        return G_g2m
    
    try:
        # 0. Setup projection if needed
        proj = _setup_projection(proj_params)
        
        # 1. Get mesh nodes and their positions (already in projected space)
        vm = G_bottom_mesh.nodes
        vm_xy = np.array([pos for _, pos in vm.data("pos")])
        dm = _calculate_mesh_distance(vm)
        
        # 2. Create grid from projected coordinates
        G_grid = _create_base_grid(coords)
        
        # 3. Build KD-tree for grid points
        vg_list, vg_coords, kdt_g = _build_kd_tree(G_grid)
        
        # 4. Add mesh nodes to grid
        G_grid.add_nodes_from(all_mesh_nodes)
        
        # 5. Create g2m graph with sorted nodes
        G_g2m = nx.Graph()
        G_g2m.add_nodes_from(sorted(G_grid.nodes(data=True)))
        G_g2m = nx.DiGraph(G_g2m)
        
        # 6. Add edges using Euclidean distance in projected space
        G_g2m = _create_g2m_edges(G_g2m, vm, vg_list, kdt_g, dm)
        
        # 7. Convert to PyTorch Geometric
        pyg_g2m = from_networkx(G_g2m)
        
        # 8. Optional plotting
        if args.plot:
            self.plot_graph(pyg_g2m, title="Grid-to-mesh")
            plt.show()
        
        return {
            'g2m_graph': pyg_g2m,
            'grid_graph': G_grid,
            'mesh_distance': dm,
            'projection': proj
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to create grid-to-mesh graph: {e}")


    """
    Create Mesh-to-Grid (m2g) graph structure.
    
    Args:
        G_g2m: Grid-to-Mesh graph
        vm: Mesh nodes
        vm_xy: Mesh node positions array
        vg_list: List of grid nodes
        args: Arguments containing plot option
        graph_dir_path: Path to save graph data
    
    Returns:
        dict: Contains:
            - m2g_graph: PyTorch Geometric graph for mesh-to-grid
            - edge_indices: Edge indices for the graph
            - edge_features: Edge features (length and vector differences)
    """
    import networkx as nx
    import numpy as np
    from scipy.spatial import KDTree
    
    def _create_m2g_edges(G_m2g, vm_list, kdt_m, num_neighbors=4):
        """Create edges from mesh to grid nodes."""
        for v in vg_list:
            # Find k nearest neighbors
            distances, neigh_idxs = kdt_m.query(G_m2g.nodes[v]["pos"], 
                                              k=num_neighbors)
            
            for i in neigh_idxs:
                u = vm_list[i]
                # Add edge from mesh to grid
                G_m2g.add_edge(u, v)
                
                # Calculate edge properties
                d = np.sqrt(
                    np.sum((G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]) ** 2)
                )
                G_m2g.edges[u, v]["len"] = d
                G_m2g.edges[u, v]["vdiff"] = (
                    G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]
                )
        return G_m2g
    
    try:
        # 1. Create base m2g graph from g2m
        G_m2g = G_g2m.copy()
        G_m2g.clear_edges()
        
        # 2. Build KD-tree for mesh points
        vm_list = list(vm)
        kdt_m = KDTree(vm_xy)
        
        # 3. Create edges from mesh to grid
        G_m2g = _create_m2g_edges(G_m2g, vm_list, kdt_m)
        
        # 4. Convert to integers with sorted labels
        G_m2g_int = nx.convert_node_labels_to_integers(
            G_m2g, first_label=0, ordering="sorted"
        )
        
        # 5. Convert to PyTorch Geometric
        pyg_m2g = from_networkx(G_m2g_int)
        
        # 6. Optional plotting
        if args.plot:
            self.plot_graph(pyg_m2g, title="Mesh-to-grid")
            plt.show()
        
        # 7. Save edge data
        self.save_edges(pyg_m2g, "m2g", graph_dir_path)
        
        return {
            'm2g_graph': pyg_m2g,
            'edge_indices': pyg_m2g.edge_index,
            'edge_features': {
                'length': torch.tensor([d['len'] for _, _, d in G_m2g.edges(data=True)]),
                'vector_diff': torch.tensor([d['vdiff'] for _, _, d in G_m2g.edges(data=True)])
            }
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to create mesh-to-grid graph: {e}")


    """
    Create Mesh-to-Grid (m2g) graph structure using Lambert Conformal projection coordinates.
    
    Args:
        coords: Array of shape [N, 2] containing [lat, lon] pairs
               or dictionary with {'lat': lat_array, 'lon': lon_array}
        G_g2m: Grid-to-Mesh graph
        vm: Mesh nodes
        args: Arguments containing plot option
        graph_dir_path: Path to save graph data
        proj_params: Optional dictionary containing Lambert projection parameters:
                    - lat_1: First standard parallel
                    - lat_2: Second standard parallel
                    - lat_0: Latitude of origin
                    - lon_0: Central meridian
                    - earth_radius: Earth radius in meters (default: 6371000)
    
    Returns:
        dict: Contains:
            - m2g_graph: PyTorch Geometric graph for mesh-to-grid
            - edge_indices: Edge indices for the graph
            - edge_features: Edge features (length and vector differences)
            - projection: Lambert projection object
    """
    import networkx as nx
    import numpy as np
    from scipy.spatial import KDTree
    import pyproj
    
    def _setup_projection(params=None):
        """Setup Lambert Conformal projection."""
        default_params = {
            'lat_1': 30.0,
            'lat_2': 60.0,
            'lat_0': 40.0,
            'lon_0': -97.0,
            'earth_radius': 6371000
        }
        
        if params is not None:
            default_params.update(params)
            
        return pyproj.Proj(
            proj='lcc',
            lat_1=default_params['lat_1'],
            lat_2=default_params['lat_2'],
            lat_0=default_params['lat_0'],
            lon_0=default_params['lon_0'],
            R=default_params['earth_radius']
        )
    
    def _project_coordinates(coords, proj):
        """Project lat/lon coordinates to Lambert projection."""
        if isinstance(coords, dict):
            lats, lons = coords['lat'].flatten(), coords['lon'].flatten()
        else:
            lats, lons = coords[:, 0], coords[:, 1]
        
        x, y = proj(lons, lats)
        return np.column_stack((x, y))
    
    def _euclidean_distance(p1, p2):
        """Calculate Euclidean distance in projected space."""
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    def _create_m2g_edges(G_m2g, vm_list, kdt_m, grid_points, num_neighbors=4):
        """Create edges from mesh to grid nodes using projected coordinates."""
        for i, v_pos in enumerate(grid_points):
            v = vg_list[i]
            # Find k nearest neighbors in projected space
            distances, neigh_idxs = kdt_m.query(v_pos, k=num_neighbors)
            
            for idx in neigh_idxs:
                u = vm_list[idx]
                # Add edge from mesh to grid
                G_m2g.add_edge(u, v)
                
                # Calculate edge properties in projected space
                u_pos = G_m2g.nodes[u]["pos"]
                d = _euclidean_distance(u_pos, v_pos)
                G_m2g.edges[u, v]["len"] = d
                G_m2g.edges[u, v]["vdiff"] = v_pos - u_pos
        return G_m2g
    
    try:
        # 0. Setup projection
        proj = _setup_projection(proj_params)
        
        # 1. Project coordinates to Lambert space
        projected_points = _project_coordinates(coords, proj)
        
        # 2. Create base m2g graph from g2m
        G_m2g = G_g2m.copy()
        G_m2g.clear_edges()
        
        # 3. Get mesh nodes and project their positions if needed
        vm_list = list(vm)
        vm_positions = np.array([vm[v]["pos"] for v in vm_list])
        if vm_positions.shape[1] == 2:  # If positions are lat/lon
            vm_positions = _project_coordinates(vm_positions, proj)
        
        # 4. Build KD-tree for mesh points in projected space
        kdt_m = KDTree(vm_positions)
        
        # 5. Create edges from mesh to grid
        G_m2g = _create_m2g_edges(G_m2g, vm_list, kdt_m, projected_points)
        
        # 6. Convert to integers with sorted labels
        G_m2g_int = nx.convert_node_labels_to_integers(
            G_m2g, first_label=0, ordering="sorted"
        )
        
        # 7. Convert to PyTorch Geometric
        pyg_m2g = from_networkx(G_m2g_int)
        
        # 8. Optional plotting
        if args.plot:
            self.plot_graph(pyg_m2g, title="Mesh-to-grid")
            plt.show()
        
        # 9. Save edge data
        self.save_edges(pyg_m2g, "m2g", graph_dir_path)
        
        return {
            'm2g_graph': pyg_m2g,
            'edge_indices': pyg_m2g.edge_index,
            'edge_features': {
                'length': torch.tensor([d['len'] for _, _, d in G_m2g.edges(data=True)]),
                'vector_diff': torch.tensor([d['vdiff'] for _, _, d in G_m2g.edges(data=True)])
            },
            'projection': proj
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to create mesh-to-grid graph: {e}")

def create_grid_to_mesh(self, coords, G_bottom_mesh, all_mesh_nodes, args):
    """
    Create Grid-to-Mesh (g2m) graph structure.
    
    Args:
        coords: Array of shape [N, 2] containing [x, y] coordinates
               or dictionary with {'x': x_array, 'y': y_array}
        G_bottom_mesh: Bottom level mesh graph
        all_mesh_nodes: All mesh nodes with data
        args: Arguments containing plot option
    
    Returns:
        dict: Contains:
            - g2m_graph: PyTorch Geometric graph for grid-to-mesh
            - grid_graph: Base grid graph
            - mesh_distance: Distance between mesh nodes
    """
    import networkx as nx
    import numpy as np
    from scipy.spatial import KDTree
    
    # Constants
    DM_SCALE = 0.67  # radius scale for grid-mesh association
    
    def _euclidean_distance(p1, p2):
        """Calculate Euclidean distance."""
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    def _calculate_mesh_distance(vm):
        """Calculate distance between mesh nodes."""
        pos1 = vm.data("pos")[(0, 1, 0)]
        pos2 = vm.data("pos")[(0, 0, 0)]
        return _euclidean_distance(pos1, pos2)
    
    def _get_coordinates(coords):
        """Get coordinates from input format."""
        if isinstance(coords, dict):
            x, y = coords['x'].flatten(), coords['y'].flatten()
            return np.column_stack((x, y))
        return coords
    
    def _create_base_grid(points):
        """Create base grid graph from coordinates."""
        G_grid = nx.Graph()
        
        # Add nodes with positions
        for i, pos in enumerate(points):
            G_grid.add_node(i, pos=pos)
        
        return self.prepend_node_index(G_grid, 1000)
    
    def _create_g2m_edges(G_g2m, vm, vg_list, kdt_g, dm):
        """Create edges from grid to mesh nodes."""
        for v in vm:
            v_pos = vm[v]["pos"]
            # Find neighbors within radius
            neigh_idxs = kdt_g.query_ball_point(v_pos, dm * DM_SCALE)
            
            for i in neigh_idxs:
                u = vg_list[i]
                # Add edge from grid to mesh
                G_g2m.add_edge(u, v)
                
                # Calculate edge properties
                u_pos = G_g2m.nodes[u]["pos"]
                d = _euclidean_distance(u_pos, v_pos)
                G_g2m.edges[u, v]["len"] = d
                G_g2m.edges[u, v]["vdiff"] = v_pos - u_pos
        return G_g2m
    
    try:
        # 1. Get mesh nodes and their positions
        vm = G_bottom_mesh.nodes
        vm_xy = np.array([pos for _, pos in vm.data("pos")])
        dm = _calculate_mesh_distance(vm)
        
        # 2. Get grid points and create grid
        grid_points = _get_coordinates(coords)
        G_grid = _create_base_grid(grid_points)
        
        # 3. Build KD-tree for grid points
        vg_list = list(G_grid.nodes)
        vg_coords = np.array([G_grid.nodes[n]['pos'] for n in vg_list])
        kdt_g = KDTree(vg_coords)
        
        # 4. Add mesh nodes to grid
        G_grid.add_nodes_from(all_mesh_nodes)
        
        # 5. Create g2m graph with sorted nodes
        G_g2m = nx.Graph()
        G_g2m.add_nodes_from(sorted(G_grid.nodes(data=True)))
        G_g2m = nx.DiGraph(G_g2m)
        
        # 6. Add edges
        G_g2m = _create_g2m_edges(G_g2m, vm, vg_list, kdt_g, dm)
        
        # 7. Convert to PyTorch Geometric
        pyg_g2m = from_networkx(G_g2m)
        
        # 8. Optional plotting
        if args.plot:
            self.plot_graph(pyg_g2m, title="Grid-to-mesh")
            plt.show()
        
        return {
            'g2m_graph': pyg_g2m,
            'grid_graph': G_grid,
            'mesh_distance': dm
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to create grid-to-mesh graph: {e}")

def create_mesh_to_grid(self, coords, vm, args, graph_dir_path):
    """
    Create Mesh-to-Grid (m2g) graph structure.
    
    Args:
        coords: Array of shape [N, 2] containing [x, y] coordinates
               or dictionary with {'x': x_array, 'y': y_array}
        vm: Mesh nodes
        args: Arguments containing plot option
        graph_dir_path: Path to save graph data
    
    Returns:
        dict: Contains:
            - m2g_graph: PyTorch Geometric graph for mesh-to-grid
            - edge_indices: Edge indices for the graph
            - edge_features: Edge features (length and vector differences)
    """
    import networkx as nx
    import numpy as np
    from scipy.spatial import KDTree
    
    def _euclidean_distance(p1, p2):
        """Calculate Euclidean distance."""
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    def _get_coordinates(coords):
        """Get coordinates from input format."""
        if isinstance(coords, dict):
            x, y = coords['x'].flatten(), coords['y'].flatten()
            return np.column_stack((x, y))
        return coords
    
    def _create_base_grid(points):
        """Create base grid graph from coordinates."""
        G = nx.Graph()
        for i, pos in enumerate(points):
            G.add_node(i, pos=pos)
        return G
    
    def _create_m2g_edges(G_m2g, vm_list, vm_positions, grid_points, num_neighbors=4):
        """Create edges from mesh to grid nodes."""
        kdt_m = KDTree(vm_positions)
        
        for i, v_pos in enumerate(grid_points):
            # Find k nearest neighbors
            distances, neigh_idxs = kdt_m.query(v_pos, k=num_neighbors)
            
            for idx in neigh_idxs:
                u = vm_list[idx]
                v = i  # Grid node index
                # Add edge from mesh to grid
                G_m2g.add_edge(u, v)
                
                # Calculate edge properties
                u_pos = vm_positions[idx]
                d = _euclidean_distance(u_pos, v_pos)
                G_m2g.edges[u, v]["len"] = d
                G_m2g.edges[u, v]["vdiff"] = v_pos - u_pos
        return G_m2g
    
    try:
        # 1. Get grid points
        grid_points = _get_coordinates(coords)
        
        # 2. Create base grid graph
        G_m2g = _create_base_grid(grid_points)
        
        # 3. Get mesh nodes positions
        vm_list = list(vm)
        vm_positions = np.array([vm[v]["pos"] for v in vm_list])
        
        # 4. Add mesh nodes to graph
        for i, v in enumerate(vm_list):
            G_m2g.add_node(v, pos=vm_positions[i])
        
        # 5. Create edges from mesh to grid
        G_m2g = _create_m2g_edges(G_m2g, vm_list, vm_positions, grid_points)
        
        # 6. Convert to integers with sorted labels
        G_m2g_int = nx.convert_node_labels_to_integers(
            G_m2g, first_label=0, ordering="sorted"
        )
        
        # 7. Convert to PyTorch Geometric
        pyg_m2g = from_networkx(G_m2g_int)
        
        # 8. Optional plotting
        if args.plot:
            self.plot_graph(pyg_m2g, title="Mesh-to-grid")
            plt.show()
        
        # 9. Save edge data
        self.save_edges(pyg_m2g, "m2g", graph_dir_path)
        
        return {
            'm2g_graph': pyg_m2g,
            'edge_indices': pyg_m2g.edge_index,
            'edge_features': {
                'length': torch.tensor([d['len'] for _, _, d in G_m2g.edges(data=True)]),
                'vector_diff': torch.tensor([d['vdiff'] for _, _, d in G_m2g.edges(data=True)])
            }
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to create mesh-to-grid graph: {e}")


def main(input_args=None):
    parser = ArgumentParser(description="Graph generation arguments")
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file (default: neural_lam/data_config.yaml)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="multiscale",
        help="Name to save graph as (default: multiscale)",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=0,
        help="If graphs should be plotted during generation "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--levels",
        type=int,
        help="Limit multi-scale mesh to given number of levels, "
        "from bottom up (default: None (no limit))",
    )
    parser.add_argument(
        "--hierarchical",
        type=int,
        default=0,
        help="Generate hierarchical mesh graph (default: 0, no)",
    )
    args = parser.parse_args(input_args)

    # Load grid positions
    config_loader = config.Config.from_file(args.data_config)
    static_dir_path = os.path.join("data", config_loader.dataset.name, "static")
    graph_dir_path = os.path.join("graphs", args.graph)
    os.makedirs(graph_dir_path, exist_ok=True)

    xy = np.load(os.path.join(static_dir_path, "nwp_xy.npy"))

    grid_xy = torch.tensor(xy)
    pos_max = torch.max(torch.abs(grid_xy))

    #
    # Mesh
    #

    # graph geometry
    nx = 3  # number of children = nx**2
    nlev = int(np.log(max(xy.shape)) / np.log(nx))
    nleaf = nx**nlev  # leaves at the bottom = nleaf**2

    mesh_levels = nlev - 1
    if args.levels:
        # Limit the levels in mesh graph
        mesh_levels = min(mesh_levels, args.levels)

    print(f"nlev: {nlev}, nleaf: {nleaf}, mesh_levels: {mesh_levels}")

    # multi resolution tree levels
    G = []
    for lev in range(1, mesh_levels + 1):
        n = int(nleaf / (nx**lev))
        g = mk_2d_graph(xy, n, n)
        if args.plot:
            plot_graph(from_networkx(g), title=f"Mesh graph, level {lev}")
            plt.show()

        G.append(g)

    if args.hierarchical:
        # Relabel nodes of each level with level index first
        G = [
            prepend_node_index(graph, level_i)
            for level_i, graph in enumerate(G)
        ]

        num_nodes_level = np.array([len(g_level.nodes) for g_level in G])
        # First node index in each level in the hierarchical graph
        first_index_level = np.concatenate(
            (np.zeros(1, dtype=int), np.cumsum(num_nodes_level[:-1]))
        )

        # Create inter-level mesh edges
        up_graphs = []
        down_graphs = []
        for from_level, to_level, G_from, G_to, start_index in zip(
            range(1, mesh_levels),
            range(0, mesh_levels - 1),
            G[1:],
            G[:-1],
            first_index_level[: mesh_levels - 1],
        ):
            # start out from graph at from level
            G_down = G_from.copy()
            G_down.clear_edges()
            G_down = networkx.DiGraph(G_down)

            # Add nodes of to level
            G_down.add_nodes_from(G_to.nodes(data=True))

            # build kd tree for mesh point pos
            # order in vm should be same as in vm_xy
            v_to_list = list(G_to.nodes)
            v_from_list = list(G_from.nodes)
            v_from_xy = np.array([xy for _, xy in G_from.nodes.data("pos")])
            kdt_m = scipy.spatial.KDTree(v_from_xy)

            # add edges from mesh to grid
            for v in v_to_list:
                # find 1(?) nearest neighbours (index to vm_xy)
                neigh_idx = kdt_m.query(G_down.nodes[v]["pos"], 1)[1]
                u = v_from_list[neigh_idx]

                # add edge from mesh to grid
                G_down.add_edge(u, v)
                d = np.sqrt(
                    np.sum(
                        (G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]) ** 2
                    )
                )
                G_down.edges[u, v]["len"] = d
                G_down.edges[u, v]["vdiff"] = (
                    G_down.nodes[u]["pos"] - G_down.nodes[v]["pos"]
                )

            # relabel nodes to integers (sorted)
            G_down_int = networkx.convert_node_labels_to_integers(
                G_down, first_label=start_index, ordering="sorted"
            )  # Issue with sorting here
            G_down_int = sort_nodes_internally(G_down_int)
            pyg_down = from_networkx_with_start_index(G_down_int, start_index)

            # Create up graph, invert downwards edges
            up_edges = torch.stack(
                (pyg_down.edge_index[1], pyg_down.edge_index[0]), dim=0
            )
            pyg_up = pyg_down.clone()
            pyg_up.edge_index = up_edges

            up_graphs.append(pyg_up)
            down_graphs.append(pyg_down)

            if args.plot:
                plot_graph(
                    pyg_down, title=f"Down graph, {from_level} -> {to_level}"
                )
                plt.show()

                plot_graph(
                    pyg_down, title=f"Up graph, {to_level} -> {from_level}"
                )
                plt.show()

        # Save up and down edges
        save_edges_list(up_graphs, "mesh_up", graph_dir_path)
        save_edges_list(down_graphs, "mesh_down", graph_dir_path)

        # Extract intra-level edges for m2m
        m2m_graphs = [
            from_networkx_with_start_index(
                networkx.convert_node_labels_to_integers(
                    level_graph, first_label=start_index, ordering="sorted"
                ),
                start_index,
            )
            for level_graph, start_index in zip(G, first_index_level)
        ]

        mesh_pos = [graph.pos.to(torch.float32) for graph in m2m_graphs]

        # For use in g2m and m2g
        G_bottom_mesh = G[0]

        joint_mesh_graph = networkx.union_all([graph for graph in G])
        all_mesh_nodes = joint_mesh_graph.nodes(data=True)

    else:
        # combine all levels to one graph
        G_tot = G[0]
        for lev in range(1, len(G)):
            nodes = list(G[lev - 1].nodes)
            n = int(np.sqrt(len(nodes)))
            ij = (
                np.array(nodes)
                .reshape((n, n, 2))[1::nx, 1::nx, :]
                .reshape(int(n / nx) ** 2, 2)
            )
            ij = [tuple(x) for x in ij]
            G[lev] = networkx.relabel_nodes(G[lev], dict(zip(G[lev].nodes, ij)))
            G_tot = networkx.compose(G_tot, G[lev])

        # Relabel mesh nodes to start with 0
        G_tot = prepend_node_index(G_tot, 0)

        # relabel nodes to integers (sorted)
        G_int = networkx.convert_node_labels_to_integers(
            G_tot, first_label=0, ordering="sorted"
        )

        # Graph to use in g2m and m2g
        G_bottom_mesh = G_tot
        all_mesh_nodes = G_tot.nodes(data=True)

        # export the nx graph to PyTorch geometric
        pyg_m2m = from_networkx(G_int)
        m2m_graphs = [pyg_m2m]
        mesh_pos = [pyg_m2m.pos.to(torch.float32)]

        if args.plot:
            plot_graph(pyg_m2m, title="Mesh-to-mesh")
            plt.show()

    # Save m2m edges
    save_edges_list(m2m_graphs, "m2m", graph_dir_path)

    # Divide mesh node pos by max coordinate of grid cell
    mesh_pos = [pos / pos_max for pos in mesh_pos]

    # Save mesh positions
    torch.save(
        mesh_pos, os.path.join(graph_dir_path, "mesh_features.pt")
    )  # mesh pos, in float32

    #
    # Grid2Mesh
    #

    # radius within which grid nodes are associated with a mesh node
    # (in terms of mesh distance)
    DM_SCALE = 0.67

    # mesh nodes on lowest level
    vm = G_bottom_mesh.nodes
    vm_xy = np.array([xy for _, xy in vm.data("pos")])
    # distance between mesh nodes
    dm = np.sqrt(
        np.sum((vm.data("pos")[(0, 1, 0)] - vm.data("pos")[(0, 0, 0)]) ** 2)
    )

    # grid nodes
    Ny, Nx = xy.shape[1:]

    G_grid = networkx.grid_2d_graph(Ny, Nx)
    G_grid.clear_edges()

    # vg features (only pos introduced here)
    for node in G_grid.nodes:
        # pos is in feature but here explicit for convenience
        G_grid.nodes[node]["pos"] = np.array([xy[0][node], xy[1][node]])

    # add 1000 to node key to separate grid nodes (1000,i,j) from mesh nodes
    # (i,j) and impose sorting order such that vm are the first nodes
    G_grid = prepend_node_index(G_grid, 1000)

    # build kd tree for grid point pos
    # order in vg_list should be same as in vg_xy
    vg_list = list(G_grid.nodes)
    vg_xy = np.array([[xy[0][node[1:]], xy[1][node[1:]]] for node in vg_list])
    kdt_g = scipy.spatial.KDTree(vg_xy)

    # now add (all) mesh nodes, include features (pos)
    G_grid.add_nodes_from(all_mesh_nodes)

    # Re-create graph with sorted node indices
    # Need to do sorting of nodes this way for indices to map correctly to pyg
    G_g2m = networkx.Graph()
    G_g2m.add_nodes_from(sorted(G_grid.nodes(data=True)))

    # turn into directed graph
    G_g2m = networkx.DiGraph(G_g2m)

    # add edges
    for v in vm:
        # find neighbours (index to vg_xy)
        neigh_idxs = kdt_g.query_ball_point(vm[v]["pos"], dm * DM_SCALE)
        for i in neigh_idxs:
            u = vg_list[i]
            # add edge from grid to mesh
            G_g2m.add_edge(u, v)
            d = np.sqrt(
                np.sum((G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]) ** 2)
            )
            G_g2m.edges[u, v]["len"] = d
            G_g2m.edges[u, v]["vdiff"] = (
                G_g2m.nodes[u]["pos"] - G_g2m.nodes[v]["pos"]
            )

    pyg_g2m = from_networkx(G_g2m)

    if args.plot:
        plot_graph(pyg_g2m, title="Grid-to-mesh")
        plt.show()

    #
    # Mesh2Grid
    #

    # start out from Grid2Mesh and then replace edges
    G_m2g = G_g2m.copy()
    G_m2g.clear_edges()

    # build kd tree for mesh point pos
    # order in vm should be same as in vm_xy
    vm_list = list(vm)
    kdt_m = scipy.spatial.KDTree(vm_xy)

    # add edges from mesh to grid
    for v in vg_list:
        # find 4 nearest neighbours (index to vm_xy)
        neigh_idxs = kdt_m.query(G_m2g.nodes[v]["pos"], 4)[1]
        for i in neigh_idxs:
            u = vm_list[i]
            # add edge from mesh to grid
            G_m2g.add_edge(u, v)
            d = np.sqrt(
                np.sum((G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]) ** 2)
            )
            G_m2g.edges[u, v]["len"] = d
            G_m2g.edges[u, v]["vdiff"] = (
                G_m2g.nodes[u]["pos"] - G_m2g.nodes[v]["pos"]
            )

    # relabel nodes to integers (sorted)
    G_m2g_int = networkx.convert_node_labels_to_integers(
        G_m2g, first_label=0, ordering="sorted"
    )
    pyg_m2g = from_networkx(G_m2g_int)

    if args.plot:
        plot_graph(pyg_m2g, title="Mesh-to-grid")
        plt.show()

    # Save g2m and m2g everything
    # g2m
    save_edges(pyg_g2m, "g2m", graph_dir_path)
    # m2g
    save_edges(pyg_m2g, "m2g", graph_dir_path)

# Example usage in another function
def some_function(self, coords, proj_params=None):
    if proj_params is not None:
        # Setup projection if needed
        proj = self.setup_lambert_projection(proj_params)
        # Project coordinates
        projected_coords = self.project_coordinates(coords, proj)
    else:
        # Use coordinates as is
        projected_coords = coords

def setup_lambert_projection(self, params=None):
    """
    Setup Lambert Conformal projection with default or custom parameters.
    
    Args:
        params: Optional dictionary containing Lambert projection parameters:
                - lat_1: First standard parallel (default: 30.0)
                - lat_2: Second standard parallel (default: 60.0)
                - lat_0: Latitude of origin (default: 40.0)
                - lon_0: Central meridian (default: -97.0)
                - earth_radius: Earth radius in meters (default: 6371000)
    
    Returns:
        pyproj.Proj: Configured Lambert Conformal projection object
    """
    import pyproj
    
    default_params = {
        'lat_1': 30.0,
        'lat_2': 60.0,
        'lat_0': 40.0,
        'lon_0': -97.0,
        'earth_radius': 6371000
    }
    
    if params is not None:
        default_params.update(params)
        
    return pyproj.Proj(
        proj='lcc',
        lat_1=default_params['lat_1'],
        lat_2=default_params['lat_2'],
        lat_0=default_params['lat_0'],
        lon_0=default_params['lon_0'],
        R=default_params['earth_radius']
    )

def project_coordinates(self, coords, proj):
    """
    Project latitude/longitude coordinates to Lambert projection space.
    
    Args:
        coords: Array of shape [N, 2] containing [lat, lon] pairs
               or dictionary with {'lat': lat_array, 'lon': lon_array}
        proj: pyproj.Proj object for Lambert projection
    
    Returns:
        numpy.ndarray: Array of shape [N, 2] containing projected [x, y] coordinates
    """
    if isinstance(coords, dict):
        lats, lons = coords['lat'].flatten(), coords['lon'].flatten()
    else:
        lats, lons = coords[:, 0], coords[:, 1]
    
    x, y = proj(lons, lats)
    return np.column_stack((x, y))


if __name__ == "__main__":
    main()
