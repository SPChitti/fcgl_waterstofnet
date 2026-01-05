"""
FCGL Query API
Provides a simple, human-friendly API for querying trained FCGL models
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import os
import glob
import re

from graph_env import GraphEnvironment
from fcgl_env import FCGLSupplyChainEnv
from fcgl_policy import FCGLPolicy
from detailed_metrics import DetailedMetricsCalculator


# ==================== 1. MODEL & GRAPH LOADING ====================

class FCGLQueryAPI:
    """
    Main API class for querying trained FCGL models
    """
    
    def __init__(self, model_path: Optional[str] = None, graph_path: Optional[str] = None):
        """
        Initialize the query API
        
        Args:
            model_path: Path to trained policy checkpoint (default: latest in training_outputs/)
            graph_path: Path to graph adjacency list CSV (default: graph_config/adjacency_list_multimodal.csv)
        """
        self.policy = None
        self.env = None
        self.graph = None
        self.nx_graph = None
        self.node_list = None
        self.metrics_calculator = DetailedMetricsCalculator()
        
        # Load model and graph
        self.load_fcgl(model_path, graph_path)
    
    def load_fcgl(self, model_path: Optional[str] = None, graph_path: Optional[str] = None):
        """
        Load trained FCGL model and graph environment
        
        Args:
            model_path: Path to policy checkpoint
            graph_path: Path to adjacency list CSV
            
        Returns:
            Tuple of (policy, env, graph)
        """
        # Default graph path
        if graph_path is None:
            graph_path = "graph_config/adjacency_list_multimodal.csv"
        
        # Load graph environment
        print(f"Loading graph from: {graph_path}")
        graph_env = GraphEnvironment(csv_path=graph_path)
        self.env = FCGLSupplyChainEnv(graph=graph_env, default_volume=1000.0)  # Match training default
        self.graph = graph_env.graph
        self.node_list = graph_env.get_all_nodes()
        
        # Build NetworkX graph for shortest path queries
        self.nx_graph = self._build_networkx_graph()
        
        # Default model path - find latest training run
        if model_path is None:
            model_path = self._find_latest_model()
        
        if model_path is None:
            print("⚠ No trained model found. Some query functions will be unavailable.")
            return None, self.env, self.graph
        
        # Load policy
        print(f"Loading policy from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize policy network
        self.policy = FCGLPolicy(
            node_list=self.node_list,
            embedding_dim=checkpoint.get('embedding_dim', 32),
            hidden_dim=checkpoint.get('hidden_dim', 128)
        )
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()
        
        print("✓ Model and graph loaded successfully")
        return self.policy, self.env, self.graph
    
    def _find_latest_model(self) -> Optional[str]:
        """Find the most recent trained model"""
        pattern = "training_outputs/*/policy_final.pt"
        models = glob.glob(pattern)
        
        if not models:
            return None
        
        # Sort by modification time
        latest = max(models, key=os.path.getmtime)
        return latest
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX directed graph for shortest path queries"""
        G = nx.DiGraph()
        
        for source, neighbors in self.graph.items():
            for neighbor_data in neighbors:
                target = neighbor_data['to']
                cost = neighbor_data['cost']
                co2 = neighbor_data['co2']
                dist = neighbor_data['dist']
                time = neighbor_data.get('time', 0.0)
                
                G.add_edge(
                    source, target,
                    cost=cost,
                    co2=co2,
                    distance=dist,
                    time=time
                )
        
        return G


    # ==================== 2. DETERMINISTIC POLICY QUERIES ====================
    
    def get_greedy_path(self, source: str, sink: Optional[str] = None) -> Dict[str, Any]:
        """
        Get deterministic greedy path from source using argmax actions
        
        Args:
            source: Starting node
            sink: Optional target sink (if specified, checks if path reaches it)
            
        Returns:
            Dictionary with path, cost, co2, steps, and match status
        """
        if self.policy is None:
            return {"error": "No trained policy loaded"}
        
        # Initialize environment
        self.env.reset(start_node=source)
        
        trajectory = [source]
        total_cost = 0.0
        total_co2 = 0.0
        total_dist = 0.0
        total_time = 0.0
        
        with torch.no_grad():
            while not self.env.is_terminal(self.env.current_state):
                state = self.env.get_state()
                current_node, volume = state
                
                # Get outgoing edges
                outgoing_edges = self.env.actions(state)
                if not outgoing_edges:
                    break
                
                # Sample action from policy
                action_idx, log_prob = self.policy.sample_action(
                    state, outgoing_edges, deterministic=True
                )
                
                # Take action
                next_edge = outgoing_edges[action_idx]
                next_node = next_edge['to']
                
                next_state, edge_cost, is_terminal = self.env.step(state, next_edge)
                trajectory.append(next_node)
                
                # Accumulate metrics
                total_cost += next_edge['cost']
                total_co2 += next_edge['co2']
                total_dist += next_edge['dist']
                total_time += next_edge.get('time', 0.0)
        
        result = {
            "path": trajectory,
            "path_str": " → ".join(trajectory),
            "cost": round(total_cost, 2),
            "co2": round(total_co2, 2),
            "distance": round(total_dist, 2),
            "time": round(total_time, 2),
            "steps": len(trajectory) - 1,
            "terminal": trajectory[-1] if trajectory else None
        }
        
        # Check if matches target sink
        if sink is not None:
            result["target_sink"] = sink
            result["matches_target"] = (trajectory[-1] == sink) if trajectory else False
        
        return result


    # ==================== 3. MIN-COST BASELINE QUERIES ====================
    
    def get_min_cost_path(self, source: str, sink: str) -> Dict[str, Any]:
        """
        Get minimum cost path using Dijkstra's algorithm
        
        Args:
            source: Starting node
            sink: Target sink node
            
        Returns:
            Dictionary with optimal path and metrics
        """
        try:
            # Use NetworkX shortest path with cost weight
            path = nx.shortest_path(
                self.nx_graph, source, sink, weight='cost'
            )
            
            # Compute metrics
            total_cost = 0.0
            total_co2 = 0.0
            total_dist = 0.0
            total_time = 0.0
            
            for i in range(len(path) - 1):
                edge_data = self.nx_graph[path[i]][path[i+1]]
                total_cost += edge_data['cost']
                total_co2 += edge_data['co2']
                total_dist += edge_data['distance']
                total_time += edge_data['time']
            
            return {
                "source": source,
                "sink": sink,
                "path": path,
                "path_str": " → ".join(path),
                "cost": round(total_cost, 2),
                "co2": round(total_co2, 2),
                "distance": round(total_dist, 2),
                "time": round(total_time, 2),
                "steps": len(path) - 1,
                "algorithm": "dijkstra"
            }
            
        except nx.NetworkXNoPath:
            return {
                "source": source,
                "sink": sink,
                "error": f"No path exists from {source} to {sink}"
            }


    # ==================== 4. SAMPLING-BASED STOCHASTIC QUERIES ====================
    
    def sample_paths(self, source: str, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Sample multiple stochastic paths from source
        
        Args:
            source: Starting node
            num_samples: Number of trajectories to sample
            
        Returns:
            List of trajectory dictionaries
        """
        if self.policy is None:
            raise RuntimeError("No trained policy loaded")
        
        trajectories = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                self.env.reset(start_node=source)
                
                trajectory = [source]
                total_cost = 0.0
                total_co2 = 0.0
                total_dist = 0.0
                total_time = 0.0
                
                while not self.env.is_terminal(self.env.current_state):
                    state = self.env.get_state()
                    current_node, volume = state
                    
                    outgoing_edges = self.env.actions(state)
                    if not outgoing_edges:
                        break
                    
                    # Stochastic sampling
                    action_idx, log_prob = self.policy.sample_action(
                        state, outgoing_edges, deterministic=False
                    )
                    
                    next_edge = outgoing_edges[action_idx]
                    next_node = next_edge['to']
                    
                    next_state, edge_cost, is_terminal = self.env.step(state, next_edge)
                    trajectory.append(next_node)
                    
                    total_cost += next_edge['cost']
                    total_co2 += next_edge['co2']
                    total_dist += next_edge['dist']
                    total_time += next_edge.get('time', 0.0)
                
                trajectories.append({
                    "path": trajectory,
                    "cost": round(total_cost, 2),
                    "co2": round(total_co2, 2),
                    "distance": round(total_dist, 2),
                    "time": round(total_time, 2),
                    "steps": len(trajectory) - 1,
                    "terminal": trajectory[-1] if trajectory else None
                })
        
        return trajectories
    
    def get_expected_cost(self, source: str, sink: str, num_samples: int = 200) -> Dict[str, Any]:
        """
        Compute expected cost statistics for paths ending at specific sink
        
        Args:
            source: Starting node
            sink: Target sink node
            num_samples: Number of samples to draw
            
        Returns:
            Dictionary with mean, min, max, std, count
        """
        try:
            trajectories = self.sample_paths(source, num_samples)
        except RuntimeError as e:
            return {
                "source": source,
                "sink": sink,
                "error": str(e)
            }
        
        # Filter by sink
        sink_trajectories = [t for t in trajectories if t.get('terminal') == sink]
        
        if not sink_trajectories:
            return {
                "source": source,
                "sink": sink,
                "count": 0,
                "error": f"No paths reached {sink} in {num_samples} samples"
            }
        
        costs = [t['cost'] for t in sink_trajectories]
        co2s = [t['co2'] for t in sink_trajectories]
        
        return {
            "source": source,
            "sink": sink,
            "count": len(sink_trajectories),
            "sample_rate": round(len(sink_trajectories) / num_samples * 100, 1),
            "cost": {
                "mean": round(np.mean(costs), 2),
                "min": round(np.min(costs), 2),
                "max": round(np.max(costs), 2),
                "std": round(np.std(costs), 2)
            },
            "co2": {
                "mean": round(np.mean(co2s), 2),
                "min": round(np.min(co2s), 2),
                "max": round(np.max(co2s), 2),
                "std": round(np.std(co2s), 2)
            }
        }
    
    def get_terminal_probability(self, source: str, sink: str, num_samples: int = 200) -> Dict[str, Any]:
        """
        Compute probability of reaching a specific sink from source
        
        Args:
            source: Starting node
            sink: Target sink node
            num_samples: Number of samples
            
        Returns:
            Dictionary with probability and count
        """
        try:
            trajectories = self.sample_paths(source, num_samples)
        except RuntimeError as e:
            return {
                "source": source,
                "sink": sink,
                "error": str(e)
            }
        
        sink_count = sum(1 for t in trajectories if t.get('terminal') == sink)
        probability = sink_count / num_samples
        
        return {
            "source": source,
            "sink": sink,
            "probability": round(probability, 4),
            "percentage": round(probability * 100, 2),
            "count": sink_count,
            "total_samples": num_samples
        }


    # ==================== 5. DISTRIBUTION & ANALYTICS QUERIES ====================
    
    def get_cost_distribution(self, source: str, sink: str, num_samples: int = 300) -> Dict[str, Any]:
        """
        Get cost distribution for paths to specific sink
        
        Args:
            source: Starting node
            sink: Target sink node
            num_samples: Number of samples
            
        Returns:
            Dictionary with cost histogram and raw values
        """
        try:
            trajectories = self.sample_paths(source, num_samples)
        except RuntimeError as e:
            return {
                "source": source,
                "sink": sink,
                "error": str(e)
            }
        
        # Filter by sink
        sink_trajectories = [t for t in trajectories if t.get('terminal') == sink]
        
        if not sink_trajectories:
            return {
                "error": f"No paths reached {sink} in {num_samples} samples"
            }
        
        costs = [t['cost'] for t in sink_trajectories]
        
        # Create histogram bins
        hist, bin_edges = np.histogram(costs, bins=10)
        
        histogram = []
        for i in range(len(hist)):
            histogram.append({
                "bin_start": round(bin_edges[i], 2),
                "bin_end": round(bin_edges[i+1], 2),
                "count": int(hist[i]),
                "percentage": round(hist[i] / len(costs) * 100, 1)
            })
        
        return {
            "source": source,
            "sink": sink,
            "count": len(costs),
            "costs": costs,
            "histogram": histogram,
            "statistics": {
                "mean": round(np.mean(costs), 2),
                "median": round(np.median(costs), 2),
                "min": round(np.min(costs), 2),
                "max": round(np.max(costs), 2),
                "std": round(np.std(costs), 2)
            }
        }
    
    def get_most_probable_path(self, source: str, sink: str, num_samples: int = 300) -> Dict[str, Any]:
        """
        Find the most frequently occurring path to sink
        
        Args:
            source: Starting node
            sink: Target sink node
            num_samples: Number of samples
            
        Returns:
            Dictionary with most common path and frequency
        """
        try:
            trajectories = self.sample_paths(source, num_samples)
        except RuntimeError as e:
            return {
                "source": source,
                "sink": sink,
                "error": str(e)
            }
        
        # Filter by sink
        sink_trajectories = [t for t in trajectories if t.get('terminal') == sink]
        
        if not sink_trajectories:
            return {
                "error": f"No paths reached {sink} in {num_samples} samples"
            }
        
        # Count path frequencies
        path_counter = Counter()
        path_costs = {}
        
        for t in sink_trajectories:
            path_tuple = tuple(t['path'])
            path_counter[path_tuple] += 1
            if path_tuple not in path_costs:
                path_costs[path_tuple] = t['cost']
        
        # Get most common
        most_common_path, frequency = path_counter.most_common(1)[0]
        
        return {
            "source": source,
            "sink": sink,
            "path": list(most_common_path),
            "path_str": " → ".join(most_common_path),
            "frequency": frequency,
            "percentage": round(frequency / len(sink_trajectories) * 100, 2),
            "cost": path_costs[most_common_path],
            "total_paths": len(sink_trajectories)
        }
    
    def get_source_contributions(self, sink: str, num_samples: int = 300) -> Dict[str, Any]:
        """
        Analyze which sources contribute paths to specific sink
        
        Args:
            sink: Target sink node
            num_samples: Number of samples per source
            
        Returns:
            Dictionary with source contributions
        """
        # Get all source nodes
        sources = [node for node in self.node_list if node.startswith('S')]
        
        contributions = {}
        total_count = 0
        
        for source in sources:
            try:
                trajectories = self.sample_paths(source, num_samples)
            except RuntimeError:
                contributions[source] = 0
                continue
            sink_count = sum(1 for t in trajectories if t.get('terminal') == sink)
            contributions[source] = sink_count
            total_count += sink_count
        
        # Convert to percentages
        percentages = {
            source: round(count / total_count * 100, 2) if total_count > 0 else 0
            for source, count in contributions.items()
        }
        
        return {
            "sink": sink,
            "samples_per_source": num_samples,
            "total_paths_to_sink": total_count,
            "contributions": contributions,
            "percentages": percentages,
            "sources": sources
        }
    
    def get_all_terminal_probabilities(self, source: str, num_samples: int = 300) -> Dict[str, Any]:
        """
        Get probability distribution over all terminals from source
        
        Args:
            source: Starting node
            num_samples: Number of samples
            
        Returns:
            Dictionary with probabilities for all sinks
        """
        try:
            trajectories = self.sample_paths(source, num_samples)
        except RuntimeError as e:
            return {
                "source": source,
                "total_samples": num_samples,
                "probabilities": {},
                "most_likely": None,
                "error": str(e)
            }
        
        # Count terminal visits
        terminal_counts = Counter(t.get('terminal') for t in trajectories if t.get('terminal'))
        
        # Get all sink nodes
        sinks = [node for node in self.node_list if node.startswith('D')]
        
        probabilities = {}
        for sink in sinks:
            count = terminal_counts.get(sink, 0)
            probabilities[sink] = {
                "probability": round(count / num_samples, 4),
                "percentage": round(count / num_samples * 100, 2),
                "count": count
            }
        
        # Guard for empty probabilities
        if not probabilities:
            return {
                "source": source,
                "total_samples": num_samples,
                "probabilities": {},
                "most_likely": None,
                "error": "No terminal nodes (D*) found in graph."
            }
        
        return {
            "source": source,
            "total_samples": num_samples,
            "probabilities": probabilities,
            "most_likely": max(probabilities.items(), key=lambda x: x[1]['probability'])[0]
        }


    # ==================== 6. HIGH-LEVEL NATURAL LANGUAGE QUERY ====================
    
    def ask(self, query_string: str) -> Dict[str, Any]:
        """
        Parse and answer natural language queries
        
        Args:
            query_string: Human-readable query like "min price at D2"
            
        Returns:
            Result dictionary based on query type
        """
        query = query_string.lower().strip()
        
        # Extract source and sink
        source = self._extract_source(query)
        sink = self._extract_sink(query)
        
        # Determine query type and route to appropriate method
        
        # Min-cost / optimal queries
        if any(word in query for word in ['min', 'minimum', 'optimal', 'cheapest', 'best']):
            if source and sink:
                result = self.get_min_cost_path(source, sink)
                result['query_type'] = 'min_cost'
                return result
            else:
                return {"error": "Need both source and sink for min-cost query"}
        
        # Expected cost queries
        elif any(w in query for w in ['expected', 'average', 'mean']) and ('cost' in query or 'price' in query):
            if source and sink:
                result = self.get_expected_cost(source, sink)
                result['query_type'] = 'expected_cost'
                return result
            else:
                return {"error": "Need both source and sink for expected/average/mean cost query"}
        
        # Most likely path queries
        elif 'most likely path' in query or 'most probable path' in query or 'most common' in query:
            if source and sink:
                result = self.get_most_probable_path(source, sink)
                result['query_type'] = 'most_probable_path'
                return result
            else:
                return {"error": "Need both source and sink for most likely path query"}
        
        # Probability queries
        elif 'probability' in query or 'chance' in query or 'likelihood' in query:
            if source and sink:
                result = self.get_terminal_probability(source, sink)
                result['query_type'] = 'probability'
                return result
            elif source:
                result = self.get_all_terminal_probabilities(source)
                result['query_type'] = 'all_probabilities'
                return result
            else:
                return {"error": "Need at least source for probability query"}
        
        # Distribution queries
        elif 'distribution' in query:
            if source and sink:
                result = self.get_cost_distribution(source, sink)
                result['query_type'] = 'cost_distribution'
                return result
            else:
                return {"error": "Need both source and sink for distribution query"}
        
        # Greedy/deterministic path queries
        elif 'greedy' in query or 'deterministic' in query or 'policy path' in query:
            if source:
                result = self.get_greedy_path(source, sink)
                result['query_type'] = 'greedy_path'
                return result
            else:
                return {"error": "Need source for greedy path query"}
        
        # Source contributions
        elif 'contribution' in query or 'which source' in query:
            if sink:
                result = self.get_source_contributions(sink)
                result['query_type'] = 'source_contributions'
                return result
            else:
                return {"error": "Need sink for source contribution query"}
        
        else:
            return {
                "error": "Could not parse query",
                "query": query_string,
                "hint": "Try queries like: 'min cost from S1 to D2', 'expected cost S1 D2', 'probability S1 D2', 'distribution S1 D2'"
            }
    
    def _extract_source(self, query: str) -> Optional[str]:
        """Extract source node from query string"""
        # Look for patterns like S1, S2, etc.
        match = re.search(r'\bs\d+\b', query, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        
        # Look for "from S1"
        match = re.search(r'from\s+(\w+)', query, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        return None
    
    def _extract_sink(self, query: str) -> Optional[str]:
        """Extract sink node from query string"""
        # Look for patterns like D1, D2, D3
        match = re.search(r'\bd\d+\b', query, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        
        # Look for "to D1" or "at D1"
        match = re.search(r'(?:to|at)\s+(\w+)', query, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        return None


    # ==================== 7. UTILITY FUNCTIONS ====================
    
    def pretty_print(self, result: Dict[str, Any], indent: int = 0) -> None:
        """
        Pretty print query results
        
        Args:
            result: Result dictionary from any query
            indent: Indentation level
        """
        prefix = "  " * indent
        
        if "error" in result:
            print(f"{prefix}❌ Error: {result['error']}")
            return
        
        query_type = result.get('query_type', 'unknown')
        
        print(f"{prefix}{'='*80}")
        print(f"{prefix}Query Type: {query_type.upper().replace('_', ' ')}")
        print(f"{prefix}{'='*80}")
        
        if query_type == 'min_cost':
            print(f"{prefix}📍 Source: {result.get('source', result['path'][0])}")
            print(f"{prefix}🎯 Sink: {result.get('sink', result['path'][-1])}")
            print(f"{prefix}🛣️  Path: {result['path_str']}")
            print(f"{prefix}💰 Cost: ${result['cost']:.2f}")
            print(f"{prefix}🌱 CO2: {result['co2']:.2f} kg")
            print(f"{prefix}📏 Distance: {result['distance']:.2f} km")
            print(f"{prefix}⏱️  Time: {result['time']:.2f} hours")
            print(f"{prefix}📊 Steps: {result['steps']}")
        
        elif query_type == 'greedy_path':
            print(f"{prefix}📍 Source: {result['path'][0]}")
            print(f"{prefix}🎯 Terminal: {result['terminal']}")
            print(f"{prefix}🛣️  Path: {result['path_str']}")
            print(f"{prefix}💰 Cost: ${result['cost']:.2f}")
            print(f"{prefix}🌱 CO2: {result['co2']:.2f} kg")
            print(f"{prefix}📊 Steps: {result['steps']}")
            if 'matches_target' in result:
                match_icon = "✅" if result['matches_target'] else "❌"
                print(f"{prefix}{match_icon} Target Match: {result['matches_target']}")
        
        elif query_type == 'expected_cost':
            print(f"{prefix}📍 Source: {result['source']}")
            print(f"{prefix}🎯 Sink: {result['sink']}")
            print(f"{prefix}📊 Samples: {result['count']}/{result.get('total_samples', 'N/A')} ({result.get('sample_rate', 0)}%)")
            print(f"{prefix}💰 Expected Cost: ${result['cost']['mean']:.2f} ± ${result['cost']['std']:.2f}")
            print(f"{prefix}   ├─ Min: ${result['cost']['min']:.2f}")
            print(f"{prefix}   └─ Max: ${result['cost']['max']:.2f}")
            print(f"{prefix}🌱 Expected CO2: {result['co2']['mean']:.2f} ± {result['co2']['std']:.2f} kg")
        
        elif query_type == 'probability':
            print(f"{prefix}📍 Source: {result['source']}")
            print(f"{prefix}🎯 Sink: {result['sink']}")
            print(f"{prefix}📊 Samples: {result['total_samples']}")
            print(f"{prefix}🎲 Probability: {result['probability']:.4f} ({result['percentage']:.2f}%)")
            print(f"{prefix}✓ Visits: {result['count']}")
        
        elif query_type == 'all_probabilities':
            print(f"{prefix}📍 Source: {result['source']}")
            print(f"{prefix}📊 Samples: {result['total_samples']}")
            print(f"{prefix}🎯 Most Likely: {result['most_likely']}")
            print(f"{prefix}\nTerminal Probabilities:")
            for sink, data in sorted(result['probabilities'].items(), key=lambda x: x[1]['percentage'], reverse=True):
                bar_length = int(data['percentage'] / 2)
                bar = "█" * bar_length
                print(f"{prefix}  {sink}: {data['percentage']:5.2f}% {bar}")
        
        elif query_type == 'most_probable_path':
            print(f"{prefix}📍 Source: {result['source']}")
            print(f"{prefix}🎯 Sink: {result['sink']}")
            print(f"{prefix}🛣️  Most Common Path: {result['path_str']}")
            print(f"{prefix}🎲 Frequency: {result['frequency']}/{result['total_paths']} ({result['percentage']:.2f}%)")
            print(f"{prefix}💰 Cost: ${result['cost']:.2f}")
        
        elif query_type == 'cost_distribution':
            print(f"{prefix}📍 Source: {result['source']}")
            print(f"{prefix}🎯 Sink: {result['sink']}")
            print(f"{prefix}📊 Samples: {result['count']}")
            print(f"{prefix}\nStatistics:")
            stats = result['statistics']
            print(f"{prefix}  Mean: ${stats['mean']:.2f}")
            print(f"{prefix}  Median: ${stats['median']:.2f}")
            print(f"{prefix}  Std: ${stats['std']:.2f}")
            print(f"{prefix}  Range: ${stats['min']:.2f} - ${stats['max']:.2f}")
            print(f"{prefix}\nHistogram:")
            for bin_data in result['histogram'][:5]:  # Show first 5 bins
                bar_length = int(bin_data['percentage'] / 2)
                bar = "█" * bar_length
                print(f"{prefix}  ${bin_data['bin_start']:.0f}-${bin_data['bin_end']:.0f}: {bar} {bin_data['count']}")
        
        elif query_type == 'source_contributions':
            print(f"{prefix}🎯 Sink: {result['sink']}")
            print(f"{prefix}📊 Samples per source: {result['samples_per_source']}")
            print(f"{prefix}✓ Total paths to sink: {result['total_paths_to_sink']}")
            print(f"{prefix}\nContributions:")
            for source in result['sources']:
                count = result['contributions'][source]
                pct = result['percentages'][source]
                bar_length = int(pct / 2)
                bar = "█" * bar_length
                print(f"{prefix}  {source}: {pct:5.2f}% ({count}) {bar}")
        
        print(f"{prefix}{'='*80}")
    
    def convert_path_to_string(self, path: List[str]) -> str:
        """Convert path list to readable string"""
        return " → ".join(path)
    
    def compute_cost_of_path(self, path: List[str]) -> Dict[str, float]:
        """
        Compute total cost and metrics for a given path
        
        Args:
            path: List of node IDs
            
        Returns:
            Dictionary with cost, co2, distance, time
        """
        if len(path) < 2:
            return {"cost": 0.0, "co2": 0.0, "distance": 0.0, "time": 0.0}
        
        total_cost = 0.0
        total_co2 = 0.0
        total_dist = 0.0
        total_time = 0.0
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            if source in self.graph:
                for edge in self.graph[source]:
                    if edge['to'] == target:
                        total_cost += edge['cost']
                        total_co2 += edge['co2']
                        total_dist += edge['dist']
                        total_time += edge.get('time', 0.0)
                        break
        
        return {
            "cost": round(total_cost, 2),
            "co2": round(total_co2, 2),
            "distance": round(total_dist, 2),
            "time": round(total_time, 2)
        }
    
    def get_detailed_metrics_for_path(self, path: List[str]) -> Dict[str, Any]:
        """
        Get comprehensive detailed metrics for a given path
        
        Args:
            path: List of node IDs representing the route
            
        Returns:
            Dictionary with detailed cost, consumption, and operational metrics
        """
        if len(path) < 2:
            return self.metrics_calculator._empty_metrics()
        
        # Collect all edges for this path
        path_edges = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            if source in self.graph:
                for edge in self.graph[source]:
                    if edge['to'] == target:
                        path_edges.append(edge)
                        break
        
        # Calculate detailed metrics
        return self.metrics_calculator.calculate_detailed_metrics(path_edges)
    
    def get_greedy_path_with_details(self, source: str, sink: Optional[str] = None) -> Dict[str, Any]:
        """
        Get deterministic greedy path with comprehensive detailed metrics
        
        Args:
            source: Starting node
            sink: Optional target sink
            
        Returns:
            Dictionary with path and detailed metrics
        """
        # Get basic path first
        result = self.get_greedy_path(source, sink)
        
        if 'error' in result:
            return result
        
        # Add detailed metrics
        path = result['path']
        detailed_metrics = self.get_detailed_metrics_for_path(path)
        
        result['detailed_metrics'] = detailed_metrics
        
        return result
    
    def get_min_cost_path_with_details(self, source: str, sink: str) -> Dict[str, Any]:
        """
        Get minimum cost path with comprehensive detailed metrics
        
        Args:
            source: Starting node
            sink: Target sink node
            
        Returns:
            Dictionary with path and detailed metrics
        """
        # Get basic path first
        result = self.get_min_cost_path(source, sink)
        
        if 'error' in result:
            return result
        
        # Add detailed metrics
        path = result['path']
        detailed_metrics = self.get_detailed_metrics_for_path(path)
        
        result['detailed_metrics'] = detailed_metrics
        
        return result


# ==================== STANDALONE FUNCTIONS ====================

def load_fcgl(model_path: Optional[str] = None, graph_path: Optional[str] = None):
    """
    Convenience function to load FCGL model and return API instance
    
    Args:
        model_path: Path to trained policy
        graph_path: Path to graph CSV
        
    Returns:
        FCGLQueryAPI instance
    """
    return FCGLQueryAPI(model_path, graph_path)


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="FCGL Query API - Query trained GFlowNet policies for supply chain routing"
    )
    parser.add_argument(
        "--query",
        type=str,
        help='Query string (e.g., "min cost from S1 to D2", "probability S1 D3")'
    )
    args = parser.parse_args()
    
    # Initialize API
    api = FCGLQueryAPI()
    
    if args.query:
        # Single query mode
        print("="*80)
        print("FCGL QUERY API - SINGLE QUERY MODE")
        print("="*80)
        print(f"\nQuery: {args.query}")
        print()
        
        result = api.ask(args.query)
        api.pretty_print(result)
        
        print("\n" + "="*80)
    else:
        # Demonstration mode
        print("="*80)
        print("FCGL QUERY API - DEMONSTRATION")
        print("="*80)
        
        print("\n" + "="*80)
        print("EXAMPLE QUERIES")
        print("="*80)
        
        # Example 1: Min-cost path
        print("\n1️⃣  Min-cost query:")
        result = api.ask("min cost from S1 to D1")
        api.pretty_print(result)
        
        # Example 2: Expected cost
        print("\n2️⃣  Expected cost query:")
        result = api.ask("expected cost S1 D1")
        api.pretty_print(result)
        
        # Example 3: Probability
        print("\n3️⃣  Probability query:")
        result = api.ask("probability of reaching D1 from S1")
        api.pretty_print(result)
        
        # Example 4: All terminal probabilities
        print("\n4️⃣  All terminal probabilities:")
        result = api.ask("probability from S1")
        api.pretty_print(result)
        
        # Example 5: Most probable path
        print("\n5️⃣  Most likely path query:")
        result = api.ask("most likely path from S1 to D1")
        api.pretty_print(result)
        
        print("\n" + "="*80)
        print("✓ Demonstration Complete!")
        print("="*80)
