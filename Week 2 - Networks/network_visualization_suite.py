#!/usr/bin/env python3
"""
Network Visualization Suite for Week Two - Networks
===================================================

This program provides comprehensive network analysis and visualization capabilities,
including real-world network data (Facebook) and theoretical network models.

Features:
- Load and analyze Facebook social network data
- Generate and compare theoretical network models (ER, WS, BA)
- Multiple visualization layouts and styles
- Network property analysis and comparison
- Interactive plots and detailed statistics

Author: Network Analysis Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import gzip
import os
from collections import defaultdict
import seaborn as sns
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NetworkVisualizationSuite:
    """Comprehensive network analysis and visualization suite."""
    
    def __init__(self, data_dir="datasets"):
        """Initialize the suite with data directory."""
        self.data_dir = data_dir
        self.networks = {} #dictionary to store the networks
        self.network_stats = {} #dictionary to store the network statistics
        
    def load_facebook_network(self, filename="facebook_combined.txt.gz"):
        """Load Facebook social network from compressed file."""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found!")
            return None
            
        print(f"Loading Facebook network from {filename}...")
        
        # create an empty graph with networkX
        G = nx.Graph()
        
        with gzip.open(filepath, 'rt') as f:
            for line in f: #for each line in the file
                if line.strip():
                    u, v = map(int, line.strip().split()) #split the line into two integers
                    G.add_edge(u, v) #add an edge between the two nodes
        
        print(f"Loaded Facebook network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        self.networks['Facebook'] = G #add the graph to the networks dictionary
        return G
    
    def generate_theoretical_networks(self, n=None, target_edges=None):
        """Generate theoretical network models for comparison."""
        
        # Use Facebook network size if available, otherwise default values
        if 'Facebook' in self.networks and n is None:
            fb_graph = self.networks['Facebook']
            n = min(1000, fb_graph.number_of_nodes())  # Limit for visualization
            target_edges = fb_graph.number_of_edges() * (n / fb_graph.number_of_nodes())**2
        elif n is None:
            n = 500
            target_edges = 1500
        
        print(f"Generating theoretical networks with n={n}, target_edges≈{int(target_edges)}")
        
        # Calculate probability for ER graph
        p = (2 * target_edges) / (n * (n - 1)) #define p based on the target number of edges from facebook network
        p = min(p, 0.1)  # Cap probability for reasonable computation
        
        # Erdős-Rényi
        print("  - Generating Erdős-Rényi graph...")
        er_graph = nx.erdos_renyi_graph(n, p, seed=42) 
        self.networks['Erdős-Rényi'] = er_graph
        
        # Watts-Strogatz (small-world)
        print("  - Generating Watts-Strogatz graph...")
        k = max(4, int(2 * target_edges / n))  # Average degree
        k = k if k % 2 == 0 else k + 1  # Ensure even for WS
        ws_graph = nx.watts_strogatz_graph(n, k, 0.3, seed=42)
        self.networks['Watts-Strogatz'] = ws_graph
        
        # Barabási-Albert (scale-free)
        print("  - Generating Barabási-Albert graph...")
        m = max(2, int(target_edges / n))  # Edges per new node
        ba_graph = nx.barabasi_albert_graph(n, m, seed=42)
        self.networks['Barabási-Albert'] = ba_graph
        
        print("Theoretical networks generated successfully!")
    
    def analyze_network_properties(self):
        """Analyze key properties of all networks."""
        print("\nAnalyzing network properties...")
        
        for name, G in self.networks.items(): #for each network in the networks dictionary
            print(f"\nAnalyzing {name} network...")
            
            stats = {}
            stats['nodes'] = G.number_of_nodes()
            stats['edges'] = G.number_of_edges()
            stats['density'] = nx.density(G)
            
            # Degree statistics
            degrees = [d for n, d in G.degree()]
            stats['avg_degree'] = np.mean(degrees)
            stats['max_degree'] = max(degrees)
            stats['degree_std'] = np.std(degrees)
            
            # Connectivity
            stats['is_connected'] = nx.is_connected(G)
            stats['num_components'] = nx.number_connected_components(G)
            
            # Path lengths (for largest component if disconnected)
            if stats['is_connected']:
                largest_cc = G
            else:
                largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
            
            if largest_cc.number_of_nodes() > 1:
                stats['avg_path_length'] = nx.average_shortest_path_length(largest_cc)
                stats['diameter'] = nx.diameter(largest_cc)
            else:
                stats['avg_path_length'] = 0
                stats['diameter'] = 0
            
            # Clustering
            stats['avg_clustering'] = nx.average_clustering(G)
            stats['global_clustering'] = nx.transitivity(G)
            
            self.network_stats[name] = stats  #store the stats of this network
    
    def create_network_comparison_plot(self):
        """Create comprehensive comparison plots of network properties."""
        if not self.network_stats:
            self.analyze_network_properties()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Network Properties Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data
        networks = list(self.network_stats.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(networks)))
        
        # 1. Basic Network Size
        ax = axes[0, 0]
        nodes = [self.network_stats[net]['nodes'] for net in networks]
        edges = [self.network_stats[net]['edges'] for net in networks]
        
        x = np.arange(len(networks))
        width = 0.35
        ax.bar(x - width/2, nodes, width, label='Nodes', alpha=0.8)
        ax.bar(x + width/2, edges, width, label='Edges', alpha=0.8)
        ax.set_xlabel('Network Type')
        ax.set_ylabel('Count')
        ax.set_title('Network Size Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(networks, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Degree Distribution
        ax = axes[0, 1]
        for i, (name, G) in enumerate(self.networks.items()):
            degrees = [d for n, d in G.degree()]
            ax.hist(degrees, bins=30, alpha=0.6, label=name, color=colors[i], density=True)
        ax.set_xlabel('Degree')
        ax.set_ylabel('Density')
        ax.set_title('Degree Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Clustering vs Average Degree
        ax = axes[0, 2]
        avg_degrees = [self.network_stats[net]['avg_degree'] for net in networks]
        clusterings = [self.network_stats[net]['avg_clustering'] for net in networks]
        
        for i, net in enumerate(networks):
            ax.scatter(avg_degrees[i], clusterings[i], s=100, 
                      color=colors[i], label=net, alpha=0.8)
        ax.set_xlabel('Average Degree')
        ax.set_ylabel('Average Clustering')
        ax.set_title('Clustering vs Degree')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Path Length vs Clustering (Small World)
        ax = axes[1, 0]
        path_lengths = [self.network_stats[net]['avg_path_length'] for net in networks]
        
        for i, net in enumerate(networks):
            ax.scatter(clusterings[i], path_lengths[i], s=100, 
                      color=colors[i], label=net, alpha=0.8)
        ax.set_xlabel('Average Clustering')
        ax.set_ylabel('Average Path Length')
        ax.set_title('Small-World Properties')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Network Density
        ax = axes[1, 1]
        densities = [self.network_stats[net]['density'] for net in networks]
        bars = ax.bar(networks, densities, color=colors, alpha=0.8)
        ax.set_ylabel('Network Density')
        ax.set_title('Network Density Comparison')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, density in zip(bars, densities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{density:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Component Analysis
        ax = axes[1, 2]
        components = [self.network_stats[net]['num_components'] for net in networks]
        connected = [1 if self.network_stats[net]['is_connected'] else 0 for net in networks]
        
        x = np.arange(len(networks))
        bars1 = ax.bar(x, components, alpha=0.8, label='Total Components', color='lightcoral')
        bars2 = ax.bar(x, connected, alpha=0.8, label='Is Connected', color='lightgreen')
        
        ax.set_xlabel('Network Type')
        ax.set_ylabel('Count')
        ax.set_title('Connectivity Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(networks, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('network_properties_comparison.png', dpi=300, bbox_inches='tight')
        print("Network comparison plot saved as 'network_properties_comparison.png'")
        return fig
    
    def visualize_networks(self, layout_type='spring', sample_size=200):
        """Visualize networks using different layouts."""
        print(f"\nCreating network visualizations with {layout_type} layout...")
        
        # Determine number of networks to plot
        num_networks = len(self.networks)
        if num_networks == 0:
            print("No networks to visualize!")
            return
        
        # Create subplot grid
        cols = min(2, num_networks)
        rows = (num_networks + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(12*cols, 8*rows))
        if num_networks == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Network Visualizations ({layout_type.title()} Layout)', 
                    fontsize=16, fontweight='bold')
        
        for idx, (name, G) in enumerate(self.networks.items()):
            ax = axes[idx]
            
            # Sample large networks for visualization
            if G.number_of_nodes() > sample_size:
                print(f"  Sampling {sample_size} nodes from {name} network for visualization...")
                nodes = list(G.nodes())
                sampled_nodes = np.random.choice(nodes, sample_size, replace=False)
                G_vis = G.subgraph(sampled_nodes)
            else:
                G_vis = G
            
            # Choose layout
            if layout_type == 'spring':
                pos = nx.spring_layout(G_vis, k=1, iterations=50, seed=42)
            elif layout_type == 'circular':
                pos = nx.circular_layout(G_vis)
            elif layout_type == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G_vis)
            elif layout_type == 'random':
                pos = nx.random_layout(G_vis, seed=42)
            else:
                pos = nx.spring_layout(G_vis, seed=42)
            
            # Color nodes by degree
            degrees = [G_vis.degree(n) for n in G_vis.nodes()]
            
            # Draw network
            nx.draw_networkx_edges(G_vis, pos, ax=ax, alpha=0.3, width=0.5, edge_color='gray')
            nodes = nx.draw_networkx_nodes(G_vis, pos, ax=ax, 
                                         node_color=degrees, 
                                         node_size=30,
                                         cmap='viridis', 
                                         alpha=0.8)
            
            # Add colorbar for degree
            if nodes is not None:
                cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
                cbar.set_label('Node Degree', rotation=270, labelpad=15)
            
            ax.set_title(f'{name}\n{G_vis.number_of_nodes()} nodes, {G_vis.number_of_edges()} edges')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_networks, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        filename = f'network_visualizations_{layout_type}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Network visualizations saved as '{filename}'")
        return fig
    
    def print_summary_statistics(self):
        """Print comprehensive summary statistics."""
        if not self.network_stats:
            self.analyze_network_properties()
        
        print("\n" + "="*80)
        print("NETWORK ANALYSIS SUMMARY")
        print("="*80)
        
        # Create summary table
        headers = ['Network', 'Nodes', 'Edges', 'Avg Degree', 'Clustering', 'Path Length', 'Connected']
        print(f"{'Network':<15} {'Nodes':<8} {'Edges':<8} {'Avg Degree':<12} {'Clustering':<12} {'Path Length':<12} {'Connected':<10}")
        print("-" * 85)
        
        for name, stats in self.network_stats.items():
            connected_str = "Yes" if stats['is_connected'] else f"No ({stats['num_components']} comp.)"
            print(f"{name:<15} {stats['nodes']:<8} {stats['edges']:<8} "
                  f"{stats['avg_degree']:<12.2f} {stats['avg_clustering']:<12.4f} "
                  f"{stats['avg_path_length']:<12.2f} {connected_str:<10}")
        
        print("\n" + "="*80)
        print("KEY INSIGHTS:")
        print("="*80)
        
        # Find networks with highest clustering
        if self.network_stats:
            max_clustering = max(stats['avg_clustering'] for stats in self.network_stats.values())
            min_path = min(stats['avg_path_length'] for stats in self.network_stats.values() 
                          if stats['avg_path_length'] > 0)
            
            print(f"• Highest clustering: {max_clustering:.4f}")
            print(f"• Shortest average path: {min_path:.2f}")
            
            # Small-world analysis
            for name, stats in self.network_stats.items():
                if stats['avg_clustering'] > 0.1 and stats['avg_path_length'] < 10:
                    print(f"• {name} exhibits small-world properties (high clustering + short paths)")
        
        print("\n")

def main():
    """Main function to run the network visualization suite."""
    print("Network Visualization Suite for Week Two - Networks")
    print("="*55)
    
    # Initialize suite
    suite = NetworkVisualizationSuite()
    
    # Load Facebook network
    facebook_graph = suite.load_facebook_network()
    
    # Generate theoretical networks
    suite.generate_theoretical_networks()
    
    # Analyze properties
    suite.analyze_network_properties()
    
    # Print summary
    suite.print_summary_statistics()
    
    # Create comparison plots
    suite.create_network_comparison_plot()
    
    # Create network visualizations
    suite.visualize_networks('spring')
    suite.visualize_networks('kamada_kawai')
    
    print("\n" + "="*55)
    print("Analysis complete! Check the generated PNG files for visualizations.")
    print("Files created:") 
    print("  • network_properties_comparison.png")
    print("  • network_visualizations_spring.png") 
    print("  • network_visualizations_kamada_kawai.png")

if __name__ == "__main__":
    main()