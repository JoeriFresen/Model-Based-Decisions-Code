#!/usr/bin/env python3
"""
Network Visual Comparison: Facebook vs Erdős-Rényi
==================================================

Side-by-side visualization of Facebook and Erdős-Rényi networks,
plus detailed clustering coefficient distribution comparison.

Author: Network Analysis Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import gzip
import os
from collections import Counter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NetworkVisualComparison:
    """Visual comparison of Facebook and Erdős-Rényi networks."""
    
    def __init__(self, data_dir="../datasets"):
        """Initialize the comparison."""
        self.data_dir = data_dir
        self.facebook_graph = None
        self.er_graph = None
        
    def load_facebook_network(self, filename="facebook_combined.txt.gz"):
        """Load the Facebook social network."""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found!")
            return False
            
        print(f"Loading Facebook network from {filename}...")
        
        self.facebook_graph = nx.Graph()
        
        with gzip.open(filepath, 'rt') as f:
            for line in f:
                if line.strip():
                    u, v = map(int, line.strip().split())
                    self.facebook_graph.add_edge(u, v)
        
        print(f"✓ Loaded Facebook network: {self.facebook_graph.number_of_nodes():,} nodes, {self.facebook_graph.number_of_edges():,} edges")
        return True
    
    def generate_equivalent_er_graph(self):
        """Generate Erdős-Rényi graph with same nodes and expected edges."""
        if self.facebook_graph is None:
            print("Please load Facebook network first!")
            return False
        
        n = self.facebook_graph.number_of_nodes()
        m = self.facebook_graph.number_of_edges()
        p = (2 * m) / (n * (n - 1))
        
        print(f"Generating equivalent Erdős-Rényi graph...")
        self.er_graph = nx.erdos_renyi_graph(n, p, seed=42)
        
        print(f"✓ Generated ER graph: {self.er_graph.number_of_nodes():,} nodes, {self.er_graph.number_of_edges():,} edges")
        return True
    
    def create_vertical_network_plot(self):
        """Create vertical visualization of both networks (one above the other)."""
        if self.facebook_graph is None or self.er_graph is None:
            print("Please load networks first!")
            return
        
        print("Creating vertical network visualization...")
        
        # Create figure with two subplots (one above the other)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        fig.suptitle('Network Structure Comparison: Facebook vs Erdős-Rényi', 
                    fontsize=16, fontweight='bold')
        
        # For visualization, we'll sample nodes to make it manageable
        sample_size = 500  # Adjust based on what looks good
        
        # Sample Facebook network
        fb_nodes_sample = np.random.choice(list(self.facebook_graph.nodes()), 
                                          min(sample_size, self.facebook_graph.number_of_nodes()), 
                                          replace=False)
        fb_subgraph = self.facebook_graph.subgraph(fb_nodes_sample)
        
        # Sample ER network (same nodes for fair comparison)
        er_subgraph = self.er_graph.subgraph(fb_nodes_sample)
        
        # Calculate node sizes based on degree
        fb_degrees = dict(fb_subgraph.degree())
        er_degrees = dict(er_subgraph.degree())
        
        fb_node_sizes = [max(20, fb_degrees[node] * 3) for node in fb_subgraph.nodes()]
        er_node_sizes = [max(20, er_degrees[node] * 3) for node in er_subgraph.nodes()]
        
        # Calculate node colors based on clustering coefficient
        fb_clustering = nx.clustering(fb_subgraph)
        er_clustering = nx.clustering(er_subgraph)
        
        fb_node_colors = [fb_clustering[node] for node in fb_subgraph.nodes()]
        er_node_colors = [er_clustering[node] for node in er_subgraph.nodes()]
        
        # Use spring layout for both (with same seed for consistency)
        print("  Computing layouts...")
        fb_pos = nx.spring_layout(fb_subgraph, k=1, iterations=50, seed=42)
        er_pos = nx.spring_layout(er_subgraph, k=1, iterations=50, seed=42)
        
        # Plot Facebook network (top subplot)
        print("  Drawing Facebook network...")
        nx.draw_networkx_edges(fb_subgraph, fb_pos, ax=ax1, alpha=0.3, width=0.5, edge_color='gray')
        nodes1 = nx.draw_networkx_nodes(fb_subgraph, fb_pos, ax=ax1,
                                       node_size=fb_node_sizes,
                                       node_color=fb_node_colors,
                                       cmap='viridis',
                                       alpha=0.8)
        
        ax1.set_title(f'Facebook Social Network\n({fb_subgraph.number_of_nodes()} nodes, {fb_subgraph.number_of_edges()} edges)', 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Add colorbar for Facebook
        cbar1 = plt.colorbar(nodes1, ax=ax1, shrink=0.8)
        cbar1.set_label('Local Clustering Coefficient', rotation=270, labelpad=20)
        
        # Plot ER network (bottom subplot)
        print("  Drawing Erdős-Rényi network...")
        nx.draw_networkx_edges(er_subgraph, er_pos, ax=ax2, alpha=0.3, width=0.5, edge_color='gray')
        nodes2 = nx.draw_networkx_nodes(er_subgraph, er_pos, ax=ax2,
                                       node_size=er_node_sizes,
                                       node_color=er_node_colors,
                                       cmap='viridis',
                                       alpha=0.8)
        
        ax2.set_title(f'Erdős-Rényi Random Graph\n({er_subgraph.number_of_nodes()} nodes, {er_subgraph.number_of_edges()} edges)', 
                     fontsize=14, fontweight='bold')
        ax2.axis('off')
        # Add colorbar for ER
        cbar2 = plt.colorbar(nodes2, ax=ax2, shrink=0.8)
        cbar2.set_label('Local Clustering Coefficient', rotation=270, labelpad=20)
        
        # Add network statistics as text
        fb_avg_clustering = nx.average_clustering(fb_subgraph)
        er_avg_clustering = nx.average_clustering(er_subgraph)
        fb_avg_degree = np.mean([d for n, d in fb_subgraph.degree()])
        er_avg_degree = np.mean([d for n, d in er_subgraph.degree()])
        
        ax1.text(0.02, 0.98, f'Avg Clustering: {fb_avg_clustering:.3f}\nAvg Degree: {fb_avg_degree:.1f}', 
                transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.text(0.02, 0.98, f'Avg Clustering: {er_avg_clustering:.3f}\nAvg Degree: {er_avg_degree:.1f}', 
                transform=ax2.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('networks_vertical.png', dpi=300, bbox_inches='tight')
        print("✓ Vertical network plot saved as 'networks_vertical.png'")
        
        return fig
    
    def create_clustering_distribution_comparison(self):
        """Create detailed clustering coefficient distribution comparison."""
        if self.facebook_graph is None or self.er_graph is None:
            print("Please load networks first!")
            return
        
        print("Creating clustering coefficient distribution comparison...")
        
        # Calculate clustering coefficients for all nodes
        print("  Computing clustering coefficients...")
        fb_clustering = list(nx.clustering(self.facebook_graph).values())
        er_clustering = list(nx.clustering(self.er_graph).values())
        
        # Create comprehensive clustering comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clustering Coefficient Distribution Comparison', 
                    fontsize=16, fontweight='bold')
        
        # 1. Histogram comparison
        ax = axes[0, 0]
        
        bins = np.linspace(0, 1, 50)
        ax.hist(fb_clustering, bins=bins, alpha=0.7, label='Facebook', 
               color='blue', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(er_clustering, bins=bins, alpha=0.7, label='Erdős-Rényi', 
               color='red', density=True, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Local Clustering Coefficient')
        ax.set_ylabel('Density')
        ax.set_title('Clustering Coefficient Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for means
        fb_mean = np.mean(fb_clustering)
        er_mean = np.mean(er_clustering)
        ax.axvline(fb_mean, color='blue', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(er_mean, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        # Add text annotations
        ax.text(0.6, 0.8, f'Facebook Mean: {fb_mean:.3f}', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.text(0.6, 0.7, f'ER Mean: {er_mean:.3f}', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # 2. Cumulative distribution
        ax = axes[0, 1]
        
        fb_sorted = np.sort(fb_clustering)
        er_sorted = np.sort(er_clustering)
        fb_cumulative = np.arange(1, len(fb_sorted) + 1) / len(fb_sorted)
        er_cumulative = np.arange(1, len(er_sorted) + 1) / len(er_sorted)
        
        ax.plot(fb_sorted, fb_cumulative, label='Facebook', color='blue', linewidth=2)
        ax.plot(er_sorted, er_cumulative, label='Erdős-Rényi', color='red', linewidth=2)
        
        ax.set_xlabel('Local Clustering Coefficient')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Box plot comparison
        ax = axes[1, 0]
        
        data_to_plot = [fb_clustering, er_clustering]
        labels = ['Facebook', 'Erdős-Rényi']
        colors = ['lightblue', 'lightcoral']
        
        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Local Clustering Coefficient')
        ax.set_title('Box Plot Comparison')
        ax.grid(True, alpha=0.3)
        
        # 4. Statistical summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate statistics
        fb_stats = {
            'Mean': np.mean(fb_clustering),
            'Median': np.median(fb_clustering),
            'Std Dev': np.std(fb_clustering),
            'Min': np.min(fb_clustering),
            'Max': np.max(fb_clustering),
            'Q1': np.percentile(fb_clustering, 25),
            'Q3': np.percentile(fb_clustering, 75)
        }
        
        er_stats = {
            'Mean': np.mean(er_clustering),
            'Median': np.median(er_clustering),
            'Std Dev': np.std(er_clustering),
            'Min': np.min(er_clustering),
            'Max': np.max(er_clustering),
            'Q1': np.percentile(er_clustering, 25),
            'Q3': np.percentile(er_clustering, 75)
        }
        
        # Create statistics table
        stats_text = "Statistical Summary\n" + "="*40 + "\n\n"
        stats_text += f"{'Statistic':<12} {'Facebook':<12} {'Erdős-Rényi':<12} {'Ratio':<8}\n"
        stats_text += "-" * 50 + "\n"
        
        for stat in ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3']:
            fb_val = fb_stats[stat]
            er_val = er_stats[stat]
            ratio = fb_val / er_val if er_val > 0 else float('inf')
            stats_text += f"{stat:<12} {fb_val:<12.4f} {er_val:<12.4f} {ratio:<8.1f}\n"
        
        # Add key insights
        stats_text += "\n" + "Key Insights:\n" + "-" * 20 + "\n"
        stats_text += f"• Facebook clustering is {fb_stats['Mean']/er_stats['Mean']:.1f}× higher\n"
        stats_text += f"• Facebook has much wider distribution\n"
        stats_text += f"• ER clustering is nearly uniform and low\n"
        stats_text += f"• Facebook shows strong community structure\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('clustering_distribution_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Clustering distribution comparison saved as 'clustering_distribution_comparison.png'")
        
        return fig

def main():
    """Main function to create network visualizations."""
    print("Network Visual Comparison: Facebook vs Erdős-Rényi")
    print("="*55)
    
    # Initialize comparison
    comparison = NetworkVisualComparison()
    
    # Load Facebook network
    if not comparison.load_facebook_network():
        print("Failed to load Facebook network. Please check the data file.")
        return
    
    # Generate equivalent ER graph
    if not comparison.generate_equivalent_er_graph():
        print("Failed to generate ER graph.")
        return
    
    # Create vertical network visualization
    comparison.create_vertical_network_plot()
    
    # Create clustering distribution comparison
    comparison.create_clustering_distribution_comparison()
    
    print(f"\n" + "="*55)
    print("VISUALIZATION COMPLETE!")
    print("="*55)
    print("Generated files:")
    print("  • networks_side_by_side.png")
    print("  • clustering_distribution_comparison.png")
    print("\nThese visualizations clearly show:")
    print("  • Structural differences between the networks")
    print("  • Dramatic clustering coefficient differences")
    print("  • Why ER graphs fail to model social networks")

if __name__ == "__main__":
    main()