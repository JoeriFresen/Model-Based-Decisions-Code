#!/usr/bin/env python3
"""
Facebook vs Erdős-Rényi Network Comparison
==========================================

Direct comparison between the full Facebook social network and an equivalent 
Erdős-Rényi random graph with the same number of nodes and edges.

This analysis demonstrates why random graphs fail to capture the structure
of real-world social networks.

Author: Network Analysis Course
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
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FacebookERComparison:
    """Compare Facebook network with equivalent Erdős-Rényi graph."""
    
    def __init__(self, data_dir="datasets"):
        """Initialize the comparison."""
        self.data_dir = data_dir
        self.facebook_graph = None
        self.er_graph = None
        self.analysis_results = {}
        
    def load_facebook_network(self, filename="facebook_combined.txt.gz"):
        """Load the full Facebook social network."""
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
        print(f"  • {self.facebook_graph.number_of_nodes():,} nodes")
        print(f"  • {self.facebook_graph.number_of_edges():,} edges")
        
        return True
    
    def generate_equivalent_er_graph(self):
        """Generate Erdős-Rényi graph with same nodes and expected edges."""
        if self.facebook_graph is None:
            print("Please load Facebook network first!")
            return False
        
        n = self.facebook_graph.number_of_nodes()
        m = self.facebook_graph.number_of_edges()
        
        # Calculate probability for same expected number of edges
        p = (2 * m) / (n * (n - 1))
        
        print(f"\nGenerating equivalent Erdős-Rényi graph...")
        print(f"  • Nodes: {n:,}")
        print(f"  • Target edges: {m:,}")
        print(f"  • Connection probability: {p:.6f}")
        
        # Generate ER graph
        self.er_graph = nx.erdos_renyi_graph(n, p, seed=42)
        
        print(f"✓ Generated ER graph:")
        print(f"  • {self.er_graph.number_of_nodes():,} nodes")
        print(f"  • {self.er_graph.number_of_edges():,} edges")
        
        return True
    
    def analyze_network_properties(self):
        """Analyze and compare key network properties."""
        if self.facebook_graph is None or self.er_graph is None:
            print("Please load networks first!")
            return
        
        print(f"\n" + "="*70)
        print("COMPREHENSIVE NETWORK ANALYSIS")
        print("="*70)
        
        networks = {
            'Facebook': self.facebook_graph,
            'Erdős-Rényi': self.er_graph
        }
        
        results = {}
        
        for name, G in networks.items():
            print(f"\nAnalyzing {name} network...")
            
            stats = {}
            
            # Basic properties
            stats['nodes'] = G.number_of_nodes()
            stats['edges'] = G.number_of_edges()
            stats['density'] = nx.density(G)
            
            # Degree statistics
            degrees = [d for n, d in G.degree()]
            stats['avg_degree'] = np.mean(degrees)
            stats['median_degree'] = np.median(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)
            stats['degree_std'] = np.std(degrees)
            stats['degree_variance'] = np.var(degrees)
            
            # Connectivity analysis
            stats['is_connected'] = nx.is_connected(G)
            stats['num_components'] = nx.number_connected_components(G)
            
            # Work with largest component for path analysis
            if stats['is_connected']:
                largest_cc = G
                stats['largest_component_size'] = stats['nodes']
                stats['largest_component_fraction'] = 1.0
            else:
                largest_cc_nodes = max(nx.connected_components(G), key=len)
                largest_cc = G.subgraph(largest_cc_nodes)
                stats['largest_component_size'] = largest_cc.number_of_nodes()
                stats['largest_component_fraction'] = stats['largest_component_size'] / stats['nodes']
            
            # Path lengths (on largest component)
            if largest_cc.number_of_nodes() > 1:
                print(f"  Computing path lengths on component with {largest_cc.number_of_nodes():,} nodes...")
                stats['avg_path_length'] = nx.average_shortest_path_length(largest_cc)
                stats['diameter'] = nx.diameter(largest_cc)
            else:
                stats['avg_path_length'] = 0
                stats['diameter'] = 0
            
            # Clustering analysis
            print(f"  Computing clustering coefficients...")
            stats['avg_clustering'] = nx.average_clustering(G)
            stats['global_clustering'] = nx.transitivity(G)
            
            # Degree distribution analysis
            degree_counts = Counter(degrees)
            stats['degree_distribution'] = degree_counts
            stats['unique_degrees'] = len(degree_counts)
            
            results[name] = stats
        
        self.analysis_results = results
        
        # Print comparison table
        self.print_comparison_table()
        
        return results
    
    def print_comparison_table(self):
        """Print detailed comparison table."""
        if not self.analysis_results:
            return
        
        fb_stats = self.analysis_results['Facebook']
        er_stats = self.analysis_results['Erdős-Rényi']
        
        print(f"\n" + "="*70)
        print("DETAILED COMPARISON TABLE")
        print("="*70)
        
        print(f"{'Property':<25} {'Facebook':<15} {'Erdős-Rényi':<15} {'Ratio (FB/ER)':<15}")
        print("-" * 70)
        
        # Basic properties
        print(f"{'Nodes':<25} {fb_stats['nodes']:<15,} {er_stats['nodes']:<15,} {fb_stats['nodes']/er_stats['nodes']:<15.2f}")
        print(f"{'Edges':<25} {fb_stats['edges']:<15,} {er_stats['edges']:<15,} {fb_stats['edges']/er_stats['edges']:<15.2f}")
        print(f"{'Density':<25} {fb_stats['density']:<15.6f} {er_stats['density']:<15.6f} {fb_stats['density']/er_stats['density']:<15.2f}")
        
        print()
        # Degree statistics
        print(f"{'Average Degree':<25} {fb_stats['avg_degree']:<15.2f} {er_stats['avg_degree']:<15.2f} {fb_stats['avg_degree']/er_stats['avg_degree']:<15.2f}")
        print(f"{'Median Degree':<25} {fb_stats['median_degree']:<15.2f} {er_stats['median_degree']:<15.2f} {fb_stats['median_degree']/er_stats['median_degree']:<15.2f}")
        print(f"{'Max Degree':<25} {fb_stats['max_degree']:<15} {er_stats['max_degree']:<15} {fb_stats['max_degree']/er_stats['max_degree']:<15.2f}")
        print(f"{'Degree Std Dev':<25} {fb_stats['degree_std']:<15.2f} {er_stats['degree_std']:<15.2f} {fb_stats['degree_std']/er_stats['degree_std']:<15.2f}")
        
        print()
        # Connectivity
        print(f"{'Is Connected':<25} {str(fb_stats['is_connected']):<15} {str(er_stats['is_connected']):<15} {'-':<15}")
        print(f"{'Num Components':<25} {fb_stats['num_components']:<15} {er_stats['num_components']:<15} {fb_stats['num_components']/er_stats['num_components']:<15.2f}")
        print(f"{'Largest Comp %':<25} {fb_stats['largest_component_fraction']*100:<15.1f} {er_stats['largest_component_fraction']*100:<15.1f} {fb_stats['largest_component_fraction']/er_stats['largest_component_fraction']:<15.2f}")
        
        print()
        # Path lengths
        if fb_stats['avg_path_length'] > 0 and er_stats['avg_path_length'] > 0:
            print(f"{'Avg Path Length':<25} {fb_stats['avg_path_length']:<15.2f} {er_stats['avg_path_length']:<15.2f} {fb_stats['avg_path_length']/er_stats['avg_path_length']:<15.2f}")
            print(f"{'Diameter':<25} {fb_stats['diameter']:<15} {er_stats['diameter']:<15} {fb_stats['diameter']/er_stats['diameter']:<15.2f}")
        
        print()
        # Clustering
        print(f"{'Avg Clustering':<25} {fb_stats['avg_clustering']:<15.4f} {er_stats['avg_clustering']:<15.4f} {fb_stats['avg_clustering']/er_stats['avg_clustering']:<15.1f}")
        print(f"{'Global Clustering':<25} {fb_stats['global_clustering']:<15.4f} {er_stats['global_clustering']:<15.4f} {fb_stats['global_clustering']/er_stats['global_clustering']:<15.1f}")
        
        print(f"\n" + "="*70)
        print("KEY INSIGHTS")
        print("="*70)
        
        clustering_ratio = fb_stats['avg_clustering'] / er_stats['avg_clustering']
        degree_var_ratio = fb_stats['degree_variance'] / er_stats['degree_variance']
        
        print(f"• Facebook clustering is {clustering_ratio:.1f}x higher than ER")
        print(f"• Facebook degree variance is {degree_var_ratio:.1f}x higher than ER")
        print(f"• Facebook has {fb_stats['max_degree']} max degree vs {er_stats['max_degree']} for ER")
        print(f"• This demonstrates the limitations of random graph models for social networks")
        
        if not fb_stats['is_connected'] and er_stats['is_connected']:
            print(f"• Facebook is fragmented ({fb_stats['num_components']} components) while ER is connected")
        elif fb_stats['is_connected'] and not er_stats['is_connected']:
            print(f"• Facebook is connected while ER is fragmented ({er_stats['num_components']} components)")
    
    def create_comprehensive_comparison_plot(self):
        """Create comprehensive comparison visualizations."""
        if not self.analysis_results:
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Facebook vs Erdős-Rényi: Comprehensive Comparison', 
                    fontsize=16, fontweight='bold')
        
        fb_graph = self.facebook_graph
        er_graph = self.er_graph
        
        # 1. Degree Distribution Comparison
        ax = axes[0, 0]
        
        fb_degrees = [d for n, d in fb_graph.degree()]
        er_degrees = [d for n, d in er_graph.degree()]
        
        ax.hist(fb_degrees, bins=50, alpha=0.7, label='Facebook', color='blue', density=True)
        ax.hist(er_degrees, bins=50, alpha=0.7, label='Erdős-Rényi', color='red', density=True)
        
        ax.set_xlabel('Degree')
        ax.set_ylabel('Density')
        ax.set_title('Degree Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.axvline(np.mean(fb_degrees), color='blue', linestyle='--', alpha=0.8)
        ax.axvline(np.mean(er_degrees), color='red', linestyle='--', alpha=0.8)
        
        # 2. Log-Log Degree Distribution
        ax = axes[0, 1]
        
        fb_degree_counts = Counter(fb_degrees)
        er_degree_counts = Counter(er_degrees)
        
        fb_degrees_unique = sorted(fb_degree_counts.keys())
        fb_counts = [fb_degree_counts[d] for d in fb_degrees_unique]
        
        er_degrees_unique = sorted(er_degree_counts.keys())
        er_counts = [er_degree_counts[d] for d in er_degrees_unique]
        
        ax.loglog(fb_degrees_unique, fb_counts, 'bo-', alpha=0.7, markersize=3, label='Facebook')
        ax.loglog(er_degrees_unique, er_counts, 'ro-', alpha=0.7, markersize=3, label='Erdős-Rényi')
        
        ax.set_xlabel('Degree (log scale)')
        ax.set_ylabel('Frequency (log scale)')
        ax.set_title('Degree Distribution (Log-Log)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Clustering Coefficient Distribution
        ax = axes[1, 0]
        
        fb_clustering = list(nx.clustering(fb_graph).values())
        er_clustering = list(nx.clustering(er_graph).values())
        
        ax.hist(fb_clustering, bins=30, alpha=0.7, label='Facebook', color='blue', density=True)
        ax.hist(er_clustering, bins=30, alpha=0.7, label='Erdős-Rényi', color='red', density=True)
        
        ax.set_xlabel('Local Clustering Coefficient')
        ax.set_ylabel('Density')
        ax.set_title('Clustering Coefficient Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Degree vs Clustering
        ax = axes[1, 1]
        
        # Sample for visualization if needed
        sample_size = min(2000, fb_graph.number_of_nodes())
        fb_nodes_sample = np.random.choice(list(fb_graph.nodes()), sample_size, replace=False)
        er_nodes_sample = np.random.choice(list(er_graph.nodes()), sample_size, replace=False)
        
        fb_degrees_sample = [fb_graph.degree(n) for n in fb_nodes_sample]
        fb_clustering_sample = [nx.clustering(fb_graph, n) for n in fb_nodes_sample]
        
        er_degrees_sample = [er_graph.degree(n) for n in er_nodes_sample]
        er_clustering_sample = [nx.clustering(er_graph, n) for n in er_nodes_sample]
        
        ax.scatter(fb_degrees_sample, fb_clustering_sample, alpha=0.5, s=10, 
                  color='blue', label='Facebook')
        ax.scatter(er_degrees_sample, er_clustering_sample, alpha=0.5, s=10, 
                  color='red', label='Erdős-Rényi')
        
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Local Clustering Coefficient')
        ax.set_title('Degree vs Local Clustering')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Network Properties Bar Chart
        ax = axes[2, 0]
        
        properties = ['Avg Clustering', 'Global Clustering', 'Avg Degree', 'Max Degree/100']
        fb_values = [
            self.analysis_results['Facebook']['avg_clustering'],
            self.analysis_results['Facebook']['global_clustering'],
            self.analysis_results['Facebook']['avg_degree'],
            self.analysis_results['Facebook']['max_degree'] / 100
        ]
        er_values = [
            self.analysis_results['Erdős-Rényi']['avg_clustering'],
            self.analysis_results['Erdős-Rényi']['global_clustering'],
            self.analysis_results['Erdős-Rényi']['avg_degree'],
            self.analysis_results['Erdős-Rényi']['max_degree'] / 100
        ]
        
        x = np.arange(len(properties))
        width = 0.35
        
        ax.bar(x - width/2, fb_values, width, label='Facebook', color='blue', alpha=0.7)
        ax.bar(x + width/2, er_values, width, label='Erdős-Rényi', color='red', alpha=0.7)
        
        ax.set_xlabel('Network Properties')
        ax.set_ylabel('Value')
        ax.set_title('Network Properties Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(properties, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Component Size Distribution
        ax = axes[2, 1]
        
        fb_components = [len(c) for c in nx.connected_components(fb_graph)]
        er_components = [len(c) for c in nx.connected_components(er_graph)]
        
        fb_components.sort(reverse=True)
        er_components.sort(reverse=True)
        
        # Show top 20 components
        fb_top = fb_components[:20] if len(fb_components) > 20 else fb_components
        er_top = er_components[:20] if len(er_components) > 20 else er_components
        
        ax.semilogy(range(1, len(fb_top) + 1), fb_top, 'bo-', label='Facebook', markersize=4)
        ax.semilogy(range(1, len(er_top) + 1), er_top, 'ro-', label='Erdős-Rényi', markersize=4)
        
        ax.set_xlabel('Component Rank')
        ax.set_ylabel('Component Size (log scale)')
        ax.set_title('Component Size Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('facebook_vs_erdos_renyi_comparison.png', dpi=300, bbox_inches='tight')
        print("\nComprehensive comparison plot saved as 'facebook_vs_erdos_renyi_comparison.png'")
        
        return fig

def main():
    """Main function to run Facebook vs ER comparison."""
    print("Facebook vs Erdős-Rényi Network Comparison")
    print("="*50)
    
    # Initialize comparison
    comparison = FacebookERComparison()
    
    # Load Facebook network
    if not comparison.load_facebook_network():
        print("Failed to load Facebook network. Please check the data file.")
        return
    
    # Generate equivalent ER graph
    if not comparison.generate_equivalent_er_graph():
        print("Failed to generate ER graph.")
        return
    
    # Analyze and compare properties
    comparison.analyze_network_properties()
    
    # Create comprehensive visualization
    comparison.create_comprehensive_comparison_plot()
    
    print(f"\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print("Generated file: facebook_vs_erdos_renyi_comparison.png")
    print("\nThis analysis clearly demonstrates why Erdős-Rényi graphs")
    print("fail to capture the structural properties of real social networks:")
    print("  • Much lower clustering in ER graphs")
    print("  • Different degree distributions")
    print("  • Different connectivity patterns")
    print("  • Missing community structure")

if __name__ == "__main__":
    main()