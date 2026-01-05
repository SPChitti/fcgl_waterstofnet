"""
Graph Environment Module for FCGL
Parses adjacency list CSV and provides graph utilities
"""

import pandas as pd
from typing import Dict, List, Any


class GraphEnvironment:
    """
    Graph environment for flow network operations
    """
    
    def __init__(self, csv_path: str):
        """
        Initialize graph from adjacency list CSV
        
        Args:
            csv_path: Path to adjacency list CSV (supports both single-modal and multi-modal formats)
        """
        self.csv_path = csv_path
        self.graph = {}
        self._load_graph()
    
    def _load_graph(self):
        """
        Load graph from CSV and build adjacency list representation
        Supports both old format (source/target) and new multi-modal format (node/neighbor/mode)
        """
        # Read CSV
        df = pd.read_csv(self.csv_path)
        
        # Detect format based on columns
        if 'node' in df.columns and 'neighbor' in df.columns:
            # New multi-modal format
            required_cols = ['node', 'neighbor', 'mode', 'cost', 'co2_kg', 'distance_km']
            source_col, target_col = 'node', 'neighbor'
            distance_col, co2_col = 'distance_km', 'co2_kg'
        else:
            # Old single-modal format
            required_cols = ['source', 'target', 'cost', 'co2', 'distance']
            source_col, target_col = 'source', 'target'
            distance_col, co2_col = 'distance', 'co2'
        
        # Validate required columns
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert all numeric fields to numeric types
        df['cost'] = pd.to_numeric(df['cost'])
        df[co2_col] = pd.to_numeric(df[co2_col])
        df[distance_col] = pd.to_numeric(df[distance_col])
        
        # Optional: capacity and time if present
        if 'capacity_kg' in df.columns:
            df['capacity_kg'] = pd.to_numeric(df['capacity_kg'])
        elif 'capacity' in df.columns:
            df['capacity'] = pd.to_numeric(df['capacity'])
        if 'time_hours' in df.columns:
            df['time_hours'] = pd.to_numeric(df['time_hours'])
        elif 'time' in df.columns:
            df['time'] = pd.to_numeric(df['time'])
        
        # Build directed graph as adjacency list
        # Initialize all nodes (including those that only appear as destinations)
        all_nodes = set(df[source_col].unique()) | set(df[target_col].unique())
        for node in all_nodes:
            self.graph[node] = []
        
        # Add edges
        for _, row in df.iterrows():
            source = row[source_col]
            target = row[target_col]
            
            edge = {
                'to': target,
                'cost': float(row['cost']),
                'co2': float(row[co2_col]),
                'dist': float(row[distance_col])
            }
            
            # Add mode information if present (multi-modal format)
            if 'mode' in row:
                edge['mode'] = str(row['mode'])
            if 'mode_type' in row:
                edge['mode_type'] = str(row['mode_type'])
            
            # Add capacity (try both column names)
            if 'capacity_kg' in row:
                edge['capacity'] = float(row['capacity_kg'])
            elif 'capacity' in row:
                edge['capacity'] = float(row['capacity'])
            
            # Add time (try both column names)
            if 'time_hours' in row:
                edge['time'] = float(row['time_hours'])
            elif 'time' in row:
                edge['time'] = float(row['time'])
            
            # Legacy: truck_type for backward compatibility
            if 'truck_type' in row:
                edge['truck_type'] = str(row['truck_type'])
            
            self.graph[source].append(edge)
        
        print(f"Graph loaded from {self.csv_path}")
        print(f"  - Nodes: {len(self.graph)}")
        print(f"  - Edges (arcs): {len(df)}")
        print(f"  - Sources: {len([n for n in self.graph if self.is_source(n)])}")
        print(f"  - Terminals: {len([n for n in self.graph if self.is_terminal(n)])}")
    
    def get_outgoing(self, node: str) -> List[Dict[str, Any]]:
        """
        Get all outgoing edges from a node
        
        Args:
            node: Node ID
            
        Returns:
            List of edge dictionaries with keys: to, cost, co2, dist
        """
        return self.graph.get(node, [])
    
    def get_all_nodes(self) -> List[str]:
        """
        Get list of all nodes in the graph
        
        Returns:
            List of all node IDs
        """
        return list(self.graph.keys())
    
    def is_terminal(self, node: str) -> bool:
        """
        Check if node is a terminal (sink) node
        
        Args:
            node: Node ID
            
        Returns:
            True if node starts with "D" (destination/sink)
        """
        return node.startswith('D')
    
    def is_source(self, node: str) -> bool:
        """
        Check if node is a source node
        
        Args:
            node: Node ID
            
        Returns:
            True if node starts with "S" (source)
        """
        return node.startswith('S')
    
    def get_sources(self) -> List[str]:
        """
        Get all source nodes
        
        Returns:
            List of source node IDs
        """
        return [node for node in self.graph if self.is_source(node)]
    
    def get_terminals(self) -> List[str]:
        """
        Get all terminal (sink) nodes
        
        Returns:
            List of terminal node IDs
        """
        return [node for node in self.graph if self.is_terminal(node)]
    
    def get_intermediate_nodes(self) -> List[str]:
        """
        Get all intermediate (transshipment) nodes
        
        Returns:
            List of intermediate node IDs
        """
        return [node for node in self.graph 
                if not self.is_source(node) and not self.is_terminal(node)]
    
    def get_neighbors(self, node: str) -> List[str]:
        """
        Get all neighbor nodes (destinations of outgoing edges)
        
        Args:
            node: Node ID
            
        Returns:
            List of neighbor node IDs
        """
        return [edge['to'] for edge in self.get_outgoing(node)]
    
    def has_edge(self, source: str, target: str) -> bool:
        """
        Check if there's a direct edge from source to target
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            True if edge exists
        """
        neighbors = self.get_neighbors(source)
        return target in neighbors
    
    def get_edge_data(self, source: str, target: str) -> Dict[str, Any]:
        """
        Get edge data between two nodes
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Edge dictionary or None if edge doesn't exist
        """
        for edge in self.get_outgoing(source):
            if edge['to'] == target:
                return edge
        return None
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph
        
        Returns:
            Dictionary with graph statistics
        """
        all_nodes = self.get_all_nodes()
        total_edges = sum(len(edges) for edges in self.graph.values())
        
        return {
            'total_nodes': len(all_nodes),
            'total_edges': total_edges,
            'num_sources': len(self.get_sources()),
            'num_terminals': len(self.get_terminals()),
            'num_intermediate': len(self.get_intermediate_nodes()),
            'sources': self.get_sources(),
            'terminals': self.get_terminals()
        }
    
    def __repr__(self):
        stats = self.get_graph_stats()
        return (f"GraphEnvironment(nodes={stats['total_nodes']}, "
                f"edges={stats['total_edges']}, "
                f"sources={stats['num_sources']}, "
                f"terminals={stats['num_terminals']})")


# Convenience function to load graph
def load_graph(csv_path: str) -> GraphEnvironment:
    """
    Load graph from adjacency list CSV
    
    Args:
        csv_path: Path to adjacency list CSV (supports both single-modal and multi-modal formats)
        
    Returns:
        GraphEnvironment instance
    """
    return GraphEnvironment(csv_path)


if __name__ == "__main__":
    # Test the module
    print("Testing GraphEnvironment with multi-modal graph...")
    
    # Load multi-modal graph
    graph_env = load_graph('graph_config/adjacency_list_multimodal.csv')
    
    print("\n" + "="*70)
    print("GRAPH STATISTICS")
    print("="*70)
    stats = graph_env.get_graph_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*70)
    print("TESTING HELPER FUNCTIONS")
    print("="*70)
    
    # Test source nodes
    sources = graph_env.get_sources()
    print(f"\nSources: {sources}")
    for src in sources[:1]:  # Test first source
        print(f"\nOutgoing edges from {src}:")
        for edge in graph_env.get_outgoing(src)[:3]:  # Show first 3
            print(f"  → {edge['to']}: cost={edge['cost']:.2f}, "
                  f"co2={edge['co2']:.2f}, dist={edge['dist']:.2f}")
    
    # Test terminal nodes
    terminals = graph_env.get_terminals()
    print(f"\nTerminals: {terminals}")
    for term in terminals[:1]:  # Test first terminal
        print(f"\nIncoming edges to {term}:")
        # Find nodes that have edges to this terminal
        incoming = [node for node in graph_env.get_all_nodes() 
                   if graph_env.has_edge(node, term)]
        print(f"  {len(incoming)} nodes connect to {term}")
        for node in incoming[:3]:  # Show first 3
            edge = graph_env.get_edge_data(node, term)
            print(f"  {node} → {term}: cost={edge['cost']:.2f}")
    
    # Test intermediate node
    intermediates = graph_env.get_intermediate_nodes()
    if intermediates:
        test_node = intermediates[0]
        print(f"\nTesting intermediate node: {test_node}")
        print(f"  is_source: {graph_env.is_source(test_node)}")
        print(f"  is_terminal: {graph_env.is_terminal(test_node)}")
        print(f"  neighbors: {graph_env.get_neighbors(test_node)[:5]}")
    
    print("\n" + "="*70)
    print("✓ All tests completed successfully!")
    print("="*70)
