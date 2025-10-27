"""
Connectome computation utilities
"""

import numpy as np
from typing import Tuple, Optional
import logging


def compute_functional_connectivity(
    timeseries: np.ndarray,
    method: str = 'correlation'
) -> np.ndarray:
    """
    Compute functional connectivity from fMRI timeseries
    
    Args:
        timeseries: BOLD timeseries [n_timepoints, n_regions]
        method: Connectivity method ('correlation', 'partial_correlation')
        
    Returns:
        Connectivity matrix [n_regions, n_regions]
    """
    if method == 'correlation':
        # Pearson correlation
        connectivity = np.corrcoef(timeseries.T)
    
    elif method == 'partial_correlation':
        # Partial correlation (using inverse covariance)
        from sklearn.covariance import GraphicalLassoCV
        model = GraphicalLassoCV()
        model.fit(timeseries)
        connectivity = model.precision_
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Set diagonal to 0
    np.fill_diagonal(connectivity, 0)
    
    return connectivity


def compute_structural_connectivity(
    dwi_data: np.ndarray,
    parcellation: np.ndarray,
    method: str = 'fiber_count'
) -> np.ndarray:
    """
    Compute structural connectivity from DTI data
    
    Args:
        dwi_data: DTI data
        parcellation: Brain parcellation labels
        method: Connectivity method
        
    Returns:
        Structural connectivity matrix
    """
    # This is a placeholder - actual implementation would involve tractography
    n_regions = len(np.unique(parcellation)) - 1  # Exclude background
    connectivity = np.zeros((n_regions, n_regions))
    
    # Placeholder: generate random structural connectivity
    connectivity = np.random.randint(0, 100, (n_regions, n_regions))
    connectivity = (connectivity + connectivity.T) / 2
    np.fill_diagonal(connectivity, 0)
    
    return connectivity


def compute_graph_metrics(connectivity_matrix: np.ndarray) -> dict:
    """
    Compute graph theory metrics from connectivity matrix
    
    Args:
        connectivity_matrix: Connectivity matrix
        
    Returns:
        Dictionary of graph metrics
    """
    import networkx as nx
    
    # Create graph
    G = nx.from_numpy_array(connectivity_matrix)
    
    # Compute metrics
    metrics = {
        'degree': dict(G.degree()),
        'clustering': nx.clustering(G),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
    }
    
    # Global metrics
    metrics['average_clustering'] = nx.average_clustering(G)
    metrics['transitivity'] = nx.transitivity(G)
    
    try:
        metrics['average_shortest_path'] = nx.average_shortest_path_length(G)
    except:
        metrics['average_shortest_path'] = np.nan
    
    return metrics


def extract_connectivity_features(connectivity_matrix: np.ndarray) -> np.ndarray:
    """
    Extract summary features from connectivity matrix
    
    Args:
        connectivity_matrix: Connectivity matrix
        
    Returns:
        Feature vector
    """
    features = []
    
    # Statistical features
    features.extend([
        np.mean(connectivity_matrix),
        np.std(connectivity_matrix),
        np.median(connectivity_matrix),
        np.percentile(connectivity_matrix, 25),
        np.percentile(connectivity_matrix, 75),
    ])
    
    # Network features
    features.extend([
        np.mean(np.sum(connectivity_matrix, axis=1)),  # Average degree
        np.std(np.sum(connectivity_matrix, axis=1)),   # Std of degree
    ])
    
    return np.array(features)
