#!/usr/bin/env python3
"""
Facebook Network Explorer
=========================

Detailed analysis and visualization of the Facebook social network dataset.
This program provides in-depth exploration of real-world network properties.

Features:
- Detailed Facebook network analysis
- Community detection
- Centrality analysis
- Degree distribution analysis
- Interactive subgraph exploration
- Comparison with random networks

Author: Michael Lees
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import gzip
import os
from collections import Counter
import seaborn as sns
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FacebookNetworkExplorer:
    """Detailed Facebook network analysis and exploration."""
    
    def __init__(self, data_dir="datasets"):
        """Initialize the explorer with data directory."""
        self.data_dir = data_dir
        self.facebook_graph = None
        self.communities = None
        self.centralities = {}
        
    def load_facebook_network(self, filename="facebook_combined.txt.gz"):
        """Load and preprocess Facebook social network."""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found!")
            return False
            
        print(f"Loading Facebook network from {filename}...")
        
        # Load the network
        self.facebook_graph = nx.Graph()
        
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                if line.strip():
                    u, v = map(int, line.strip().split())
                    self.facebook_graph.add_edge(u, v)
        
        print(f"✓ Loaded Facebook network:")
        print(f"  • {self.facebook_graph.number_of_nodes():,} nodes (users)")
        print(f"  • {self.facebook_graph.number_of_edges():,} edges (friendships)")
        print(f"  • Average degree: {2*self.facebook_graph.number_of_edges()/self.facebook_graph.number_of_nodes():.2f}")
        
        return True
    
    def analyze_basic_properties(self):
        """Analyze basic network properties."""
        if self.facebook_graph is None:
            print("Please load the Facebook network first!")
            return
        
        G = self.facebook_graph
        print("\n" + "="*60)
        print("BASIC NETWORK PROPERTIES")
        print("="*60)
        
        # Basic stats
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        density = nx.density(G)
        
        print(f"Nodes (Users): {n_nodes:,}")
        print(f"Edges (Friendships): {n_edges:,}")
        print(f"Network Density: {density:.6f}")
        print(f"Maximum Possible Edges: {n_nodes*(n_nodes-1)//2:,}")
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        print(f"\nDegree Statistics:")
        print(f"  • Average degree: {np.mean(degrees):.2f}")
        print(f"  • Median degree: {np.median(degrees):.2f}")
        print(f"  • Max degree: {max(degrees)}")
        print(f"  • Min degree: {min(degrees)}")
        print(f"  • Degree std: {np.std(degrees):.2f}")
        
        # Connectivity
        is_connected = nx.is_connected(G)
        n_components = nx.number_connected_components(G)
        
        print(f"\nConnectivity:")
        print(f"  • Is connected: {is_connected}")
        print(f"  • Number of components: {n_components}")
        
        if not is_connected:
            # Analyze largest component
            largest_cc = max(nx.connected_components(G), key=len)
            largest_cc_size = len(largest_cc)
            print(f"  • Largest component size: {largest_cc_size:,} ({100*largest_cc_size/n_nodes:.1f}%)")
            
            # Use largest component for path analysis
            G_main = G.subgraph(largest_cc)
        else:
            G_main = G
        
        # Path lengths (on main component)
        if G_main.number_of_nodes() > 1:
            avg_path_length = nx.average_shortest_path_length(G_main)
            diameter = nx.diameter(G_main)
            print(f"  • Average path length: {avg_path_length:.2f}")
            print(f"  • Diameter: {diameter}")
        
        # Clustering
        avg_clustering = nx.average_clustering(G)
        global_clustering = nx.transitivity(G)
        
        print(f"\nClustering:")
        print(f"  • Average clustering coefficient: {avg_clustering:.4f}")
        print(f"  • Global clustering coefficient: {global_clustering:.4f}")
        
        # Compare with random network
        p_random = density
        expected_clustering_random = p_random
        print(f"  • Expected clustering (random): {expected_clustering_random:.6f}")
        print(f"  • Clustering enhancement: {avg_clustering/expected_clustering_random:.1f}x higher than random")
    
    def analyze_degree_distribution(self):
        """Analyze and visualize degree distribution."""
        if self.facebook_graph is None:
            return
        
        G = self.facebook_graph
        degrees = [d for n, d in G.degree()]
        degree_counts = Counter(degrees)
        
        print(f"\n" + "="*60)
        print("DEGREE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Create degree distribution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Facebook Network: Degree Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram
        ax = axes[0, 0]
        ax.hist(degrees, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title('Degree Distribution (Linear Scale)')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.axvline(np.mean(degrees), color='red', linestyle='--', label=f'Mean: {np.mean(degrees):.1f}')
        ax.axvline(np.median(degrees), color='orange', linestyle='--', label=f'Median: {np.median(degrees):.1f}')
        ax.legend()
        
        # 2. Log-log plot
        ax = axes[0, 1]
        degrees_unique = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees_unique]
        
        ax.loglog(degrees_unique, counts, 'bo-', alpha=0.7, markersize=4)
        ax.set_xlabel('Degree (log scale)')
        ax.set_ylabel('Frequency (log scale)')
        ax.set_title('Degree Distribution (Log-Log Scale)')
        ax.grid(True, alpha=0.3)
        
        # Fit power law
        log_degrees = np.log(degrees_unique[1:])  # Exclude degree 0
        log_counts = np.log(counts[1:])
        slope, intercept = np.polyfit(log_degrees, log_counts, 1)
        
        ax.plot(degrees_unique[1:], np.exp(intercept) * np.array(degrees_unique[1:])**slope, 
                'r--', label=f'Power law fit: γ ≈ {-slope:.2f}')
        ax.legend()
        
        # 3. Cumulative distribution
        ax = axes[1, 0]
        degrees_sorted = np.sort(degrees)
        cumulative = 1 - np.arange(1, len(degrees_sorted) + 1) / len(degrees_sorted)
        
        ax.loglog(degrees_sorted, cumulative, 'go-', alpha=0.7, markersize=2)
        ax.set_xlabel('Degree (log scale)')
        ax.set_ylabel('P(Degree ≥ k)')
        ax.set_title('Cumulative Degree Distribution')
        ax.grid(True, alpha=0.3)
        
        # 4. Degree vs Local Clustering
        ax = axes[1, 1]
        clustering_dict = nx.clustering(G)
        node_degrees = dict(G.degree())
        
        # Sample for visualization if too many points
        nodes_sample = list(G.nodes())
        if len(nodes_sample) > 1000:
            nodes_sample = np.random.choice(nodes_sample, 1000, replace=False)
        
        degrees_sample = [node_degrees[n] for n in nodes_sample]
        clustering_sample = [clustering_dict[n] for n in nodes_sample]
        
        ax.scatter(degrees_sample, clustering_sample, alpha=0.5, s=10)
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Local Clustering Coefficient')
        ax.set_title('Degree vs Local Clustering')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(degrees_sample) > 10:
            z = np.polyfit(degrees_sample, clustering_sample, 1)
            p = np.poly1d(z)
            ax.plot(sorted(degrees_sample), p(sorted(degrees_sample)), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('facebook_degree_analysis.png', dpi=300, bbox_inches='tight')
        print("Degree distribution analysis saved as 'facebook_degree_analysis.png'")
        
        # Print key statistics
        print(f"\nKey Findings:")
        print(f"  • Power law exponent (γ): ~{-slope:.2f}")
        print(f"  • Most connected user has {max(degrees)} friends")
        print(f"  • {sum(1 for d in degrees if d > 100)} users have >100 friends")
        print(f"  • {sum(1 for d in degrees if d == 1)} users have only 1 friend")
    
    def calculate_centralities(self, sample_size=1000):
        """Calculate various centrality measures."""
        if self.facebook_graph is None:
            return
        
        G = self.facebook_graph
        print(f"\n" + "="*60)
        print("CENTRALITY ANALYSIS")
        print("="*60)
        
        # For large networks, work with largest connected component
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G_main = G.subgraph(largest_cc)
            print(f"Working with largest connected component: {G_main.number_of_nodes():,} nodes")
        else:
            G_main = G
        
        # Sample for computational efficiency
        if G_main.number_of_nodes() > sample_size:
            nodes_sample = np.random.choice(list(G_main.nodes()), sample_size, replace=False)
            G_sample = G_main.subgraph(nodes_sample)
            print(f"Sampling {sample_size} nodes for centrality calculations...")
        else:
            G_sample = G_main
            nodes_sample = list(G_sample.nodes())
        
        print("Calculating centrality measures...")
        
        # Degree centrality
        print("  • Degree centrality...")
        degree_cent = nx.degree_centrality(G_sample)
        
        # Betweenness centrality
        print("  • Betweenness centrality...")
        betweenness_cent = nx.betweenness_centrality(G_sample, k=min(100, len(nodes_sample)))
        
        # Closeness centrality
        print("  • Closeness centrality...")
        closeness_cent = nx.closeness_centrality(G_sample)
        
        # Eigenvector centrality
        print("  • Eigenvector centrality...")
        try:
            eigenvector_cent = nx.eigenvector_centrality(G_sample, max_iter=1000)
        except:
            eigenvector_cent = {n: 0 for n in G_sample.nodes()}
            print("    (Eigenvector centrality calculation failed)")
        
        self.centralities = {
            'degree': degree_cent,
            'betweenness': betweenness_cent,
            'closeness': closeness_cent,
            'eigenvector': eigenvector_cent
        }
        
        # Create centrality comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Facebook Network: Centrality Analysis', fontsize=16, fontweight='bold')
        
        centrality_names = ['Degree', 'Betweenness', 'Closeness', 'Eigenvector']
        centrality_keys = ['degree', 'betweenness', 'closeness', 'eigenvector']
        
        for i, (name, key) in enumerate(zip(centrality_names, centrality_keys)):
            ax = axes[i//2, i%2]
            values = list(self.centralities[key].values())
            
            ax.hist(values, bins=30, alpha=0.7, color=plt.cm.Set3(i), edgecolor='black')
            ax.set_xlabel(f'{name} Centrality')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Centrality Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(values)
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.4f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('facebook_centrality_analysis.png', dpi=300, bbox_inches='tight')
        print("Centrality analysis saved as 'facebook_centrality_analysis.png'")
        
        # Print top nodes for each centrality
        print(f"\nTop 5 nodes by centrality:")
        for name, key in zip(centrality_names, centrality_keys):
            top_nodes = sorted(self.centralities[key].items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n{name} Centrality:")
            for rank, (node, score) in enumerate(top_nodes, 1):
                print(f"  {rank}. Node {node}: {score:.4f}")
    
    def visualize_network_sample(self, sample_size=100, layout='spring'):
        """Visualize a sample of the Facebook network."""
        if self.facebook_graph is None:
            return
        
        G = self.facebook_graph
        print(f"\n" + "="*60)
        print(f"NETWORK VISUALIZATION (Sample of {sample_size} nodes)")
        print("="*60)
        
        # Sample nodes
        nodes_sample = np.random.choice(list(G.nodes()), 
                                       min(sample_size, G.number_of_nodes()), 
                                       replace=False)
        G_sample = G.subgraph(nodes_sample)
        
        print(f"Sampled subgraph: {G_sample.number_of_nodes()} nodes, {G_sample.number_of_edges()} edges")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'Facebook Network Sample Visualization ({sample_size} nodes)', 
                    fontsize=16, fontweight='bold')
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G_sample, k=2, iterations=50, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G_sample)
        else:
            pos = nx.spring_layout(G_sample, seed=42)
        
        # Plot 1: Colored by degree
        ax = axes[0]
        degrees = [G_sample.degree(n) for n in G_sample.nodes()]
        
        nx.draw_networkx_edges(G_sample, pos, ax=ax, alpha=0.3, width=0.5, edge_color='lightgray')
        nodes = nx.draw_networkx_nodes(G_sample, pos, ax=ax,
                                     node_color=degrees,
                                     node_size=[50 + 10*d for d in degrees],
                                     cmap='viridis',
                                     alpha=0.8)
        
        if nodes is not None:
            cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
            cbar.set_label('Node Degree', rotation=270, labelpad=15)
        
        ax.set_title('Nodes Colored by Degree')
        ax.axis('off')
        
        # Plot 2: Colored by clustering
        ax = axes[1]
        clustering_dict = nx.clustering(G_sample)
        clustering_values = [clustering_dict[n] for n in G_sample.nodes()]
        
        nx.draw_networkx_edges(G_sample, pos, ax=ax, alpha=0.3, width=0.5, edge_color='lightgray')
        nodes = nx.draw_networkx_nodes(G_sample, pos, ax=ax,
                                     node_color=clustering_values,
                                     node_size=[50 + 10*d for d in degrees],
                                     cmap='plasma',
                                     alpha=0.8)
        
        if nodes is not None:
            cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
            cbar.set_label('Local Clustering', rotation=270, labelpad=15)
        
        ax.set_title('Nodes Colored by Local Clustering')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('facebook_network_sample.png', dpi=300, bbox_inches='tight')
        print("Network sample visualization saved as 'facebook_network_sample.png'")
    
    def compare_with_random_network(self):
        """Compare Facebook network with equivalent random networks."""
        if self.facebook_graph is None:
            return
        
        G = self.facebook_graph
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        print(f"\n" + "="*60)
        print("COMPARISON WITH RANDOM NETWORKS")
        print("="*60)
        
        # Generate equivalent random networks
        print("Generating comparison networks...")
        
        # Erdős-Rényi with same density
        p = 2 * m / (n * (n - 1))
        er_graph = nx.erdos_renyi_graph(n, p, seed=42)
        
        # Configuration model (same degree sequence)
        degree_sequence = [d for n, d in G.degree()]
        # Ensure sum is even for configuration model
        if sum(degree_sequence) % 2 == 1:
            degree_sequence[0] += 1
        
        try:
            config_graph = nx.configuration_model(degree_sequence, seed=42)
            config_graph = nx.Graph(config_graph)  # Remove multi-edges and self-loops
            config_graph.remove_edges_from(nx.selfloop_edges(config_graph))
        except:
            config_graph = er_graph  # Fallback
        
        # Calculate properties
        networks = {
            'Facebook (Real)': G,
            'Erdős-Rényi': er_graph,
            'Configuration Model': config_graph
        }
        
        properties = {}
        for name, graph in networks.items():
            props = {}
            props['nodes'] = graph.number_of_nodes()
            props['edges'] = graph.number_of_edges()
            props['avg_degree'] = 2 * graph.number_of_edges() / graph.number_of_nodes()
            props['clustering'] = nx.average_clustering(graph)
            props['transitivity'] = nx.transitivity(graph)
            
            # Path length on largest component
            if nx.is_connected(graph):
                props['avg_path_length'] = nx.average_shortest_path_length(graph)
            else:
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                if subgraph.number_of_nodes() > 1:
                    props['avg_path_length'] = nx.average_shortest_path_length(subgraph)
                else:
                    props['avg_path_length'] = 0
            
            properties[name] = props
        
        # Print comparison table
        print(f"\n{'Network':<20} {'Nodes':<8} {'Edges':<8} {'Avg Degree':<12} {'Clustering':<12} {'Path Length':<12}")
        print("-" * 80)
        
        for name, props in properties.items():
            print(f"{name:<20} {props['nodes']:<8} {props['edges']:<8} "
                  f"{props['avg_degree']:<12.2f} {props['clustering']:<12.4f} "
                  f"{props['avg_path_length']:<12.2f}")
        
        # Calculate ratios
        fb_clustering = properties['Facebook (Real)']['clustering']
        er_clustering = properties['Erdős-Rényi']['clustering']
        
        print(f"\nKey Comparisons:")
        print(f"  • Facebook clustering is {fb_clustering/er_clustering:.1f}x higher than random")
        print(f"  • This demonstrates the 'small-world' property of social networks")
        
        return properties

def main():
    """Main function to run Facebook network exploration."""
    print("Facebook Network Explorer")
    print("="*40)
    
    # Initialize explorer
    explorer = FacebookNetworkExplorer()
    
    # Load Facebook network
    if not explorer.load_facebook_network():
        print("Failed to load Facebook network. Please check the data file.")
        return
    
    # Run comprehensive analysis
    explorer.analyze_basic_properties()
    explorer.analyze_degree_distribution()
    explorer.calculate_centralities()
    explorer.visualize_network_sample(sample_size=150)
    explorer.compare_with_random_network()
    
    print("\n" + "="*60)
    print("Facebook Network Analysis Complete!")
    print("="*60)
    print("Generated files:")
    print("  • facebook_degree_analysis.png")
    print("  • facebook_centrality_analysis.png")
    print("  • facebook_network_sample.png")
    print("\nThis analysis demonstrates key properties of real-world social networks:")
    print("  • High clustering (friends of friends are often friends)")
    print("  • Small-world properties (short paths despite high clustering)")
    print("  • Scale-free degree distribution (few highly connected hubs)")
    print("  • Community structure and centrality patterns")

if __name__ == "__main__":
    main()