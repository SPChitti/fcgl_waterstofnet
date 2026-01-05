import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def visualize_nodes_by_level():
    """Place nodes by level without edges to see structure"""
    
    # Load graph
    graph_file = Path(__file__).parent / 'graph_network' / 'graph_network.json'
    with open(graph_file, 'r') as f:
        graph = json.load(f)
    
    nodes = graph['nodes']
    edges = graph['edges']
    
    print("="*80)
    print("POSITIONING NODES BY LEVEL")
    print("="*80)
    
    # Build networkx graph
    G = nx.DiGraph()
    
    for node_id, node_data in nodes.items():
        G.add_node(int(node_id), **node_data)
    
    for edge in edges:
        key = (edge['from_node'], edge['to_node'])
        G.add_edge(edge['from_node'], edge['to_node'])
    
    # Identify node types
    source_nodes = sorted([int(nid) for nid, n in nodes.items() if n['type'] == 'source'])
    dest_nodes = sorted([int(nid) for nid, n in nodes.items() if n['type'] == 'destination'])
    junction_nodes = sorted([int(nid) for nid, n in nodes.items() if n['type'] == 'convergence'])
    
    print(f"\nNodes: {len(nodes)} (Sources: {len(source_nodes)}, Destinations: {len(dest_nodes)}, Junctions: {len(junction_nodes)})")
    
    # Compute layers using BFS
    print("Computing levels using BFS from sources...")
    
    layers = {}
    
    for src in source_nodes:
        layers[src] = 0
    
    # BFS from all sources
    queue = [(src, 0) for src in source_nodes]
    visited = set(source_nodes)
    
    while queue:
        current, dist = queue.pop(0)
        
        for neighbor in G.successors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                layers[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
            else:
                # Update to minimum distance
                if neighbor not in dest_nodes:
                    layers[neighbor] = min(layers.get(neighbor, 999), dist + 1)
    
    # FORCE all destinations to the same level (max level)
    max_dest_level = max(layers[d] for d in dest_nodes)
    for dest in dest_nodes:
        layers[dest] = max_dest_level
    
    # Group nodes by layer
    layer_groups = defaultdict(list)
    for node_id, layer in layers.items():
        layer_groups[layer].append(node_id)
    
    max_level = max(layers.values())
    print(f"Total levels: {max_level + 1} (0 to {max_level})")
    
    # Position nodes
    print("Positioning nodes...")
    pos = {}
    
    # Find max nodes in any layer for consistent alignment
    max_nodes_in_layer = max(len(nodes_list) for nodes_list in layer_groups.values())
    print(f"Max nodes in any layer: {max_nodes_in_layer}")
    
    for layer in range(max_level + 1):
        nodes_in_layer = sorted(layer_groups.get(layer, []))
        num_nodes = len(nodes_in_layer)
        
        if num_nodes == 0:
            continue
        
        # X position = layer number
        x = layer
        
        # Y positions: CENTER nodes relative to the longest column
        y_spacing = 1.0
        # Calculate offset to center this column
        total_height = (max_nodes_in_layer - 1) * y_spacing
        column_height = (num_nodes - 1) * y_spacing
        y_offset = (total_height - column_height) / 2
        
        # Position nodes from top, but offset to center
        y_positions = [y_offset + i * y_spacing for i in range(num_nodes)]
        
        for node_id, y in zip(nodes_in_layer, y_positions):
            pos[node_id] = (x, -y)
    
    print(f"Positioned {len(pos)} nodes")
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Calculate figure size based on max level
    fig_width = max(20, max_level * 0.8)
    fig_height = 16
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor('#f5f5f5')
    
    # DRAW EDGES FIRST (so nodes appear on top)
    print("Drawing edges...")
    nx.draw_networkx_edges(G, pos,
                          edge_color='#34495e',
                          width=1.5,
                          alpha=0.4,
                          arrows=True,
                          arrowsize=10,
                          arrowstyle='->',
                          connectionstyle='arc3,rad=0.0',
                          ax=ax)
    
    # Draw nodes WITHOUT edges
    # Sources - Green squares
    nx.draw_networkx_nodes(G, pos, nodelist=source_nodes,
                          node_color='#27ae60',
                          node_size=800,
                          node_shape='s',
                          edgecolors='#1e8449',
                          linewidths=3,
                          ax=ax,
                          label=f'Sources ({len(source_nodes)})')
    
    # Destinations - Red squares
    nx.draw_networkx_nodes(G, pos, nodelist=dest_nodes,
                          node_color='#e74c3c',
                          node_size=800,
                          node_shape='s',
                          edgecolors='#c0392b',
                          linewidths=3,
                          ax=ax,
                          label=f'Destinations ({len(dest_nodes)})')
    
    # Junctions - Blue circles
    nx.draw_networkx_nodes(G, pos, nodelist=junction_nodes,
                          node_color='#5dade2',
                          node_size=150,
                          node_shape='o',
                          edgecolors='#2874a6',
                          linewidths=1,
                          alpha=0.8,
                          ax=ax,
                          label=f'Junctions ({len(junction_nodes)})')
    
    # Labels for sources and destinations
    labels = {}
    for node_id in source_nodes + dest_nodes:
        labels[node_id] = nodes[str(node_id)]['id']
    
    nx.draw_networkx_labels(G, pos, labels, 
                           font_size=10, 
                           font_weight='bold', 
                           font_color='white',
                           ax=ax)
    
    # Draw vertical grid lines for each level
    for layer in range(max_level + 1):
        ax.axvline(x=layer, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)
    
    # Add level labels on x-axis
    ax.set_xlabel('Level (Distance from Sources)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Nodes', fontsize=14, fontweight='bold')
    
    # Title
    ax.set_title(f'Graph Structure with Edges Connected\n' +
                 f'{len(source_nodes)} Sources (Left) → {len(dest_nodes)} Destinations (Right) across {max_level + 1} levels\n' +
                 f'{G.number_of_edges()} directed edges',
                 fontsize=18, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.margins(0.02)
    ax.grid(True, alpha=0.1)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / 'maps' / 'graph_structure_by_level.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Saved: {output_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("LEVEL STATISTICS")
    print("="*80)
    
    for layer in sorted(layer_groups.keys())[:10]:
        nodes_in_layer = layer_groups[layer]
        node_info = []
        for nid in sorted(nodes_in_layer)[:5]:
            node_type = nodes[str(nid)]['type'][0].upper()
            node_name = nodes[str(nid)].get('name', f'N{nid}')[:20]
            node_info.append(f"{nid}({node_type})")
        
        more = f" ... +{len(nodes_in_layer)-5}" if len(nodes_in_layer) > 5 else ""
        print(f"  Level {layer:2d}: {len(nodes_in_layer):3d} nodes - {', '.join(node_info)}{more}")
    
    if max_level > 10:
        print(f"  ...")
        for layer in sorted(layer_groups.keys())[-5:]:
            nodes_in_layer = layer_groups[layer]
            node_info = []
            for nid in sorted(nodes_in_layer)[:5]:
                node_type = nodes[str(nid)]['type'][0].upper()
                node_info.append(f"{nid}({node_type})")
            print(f"  Level {layer:2d}: {len(nodes_in_layer):3d} nodes - {', '.join(node_info)}")
    
    print("="*80)
    print("\n✓ Visualization complete - nodes positioned with directed edges drawn")

if __name__ == '__main__':
    visualize_nodes_by_level()
