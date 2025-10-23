#!/usr/bin/env python3
"""
Watts-Strogatz Network Degree Distribution Analysis
==================================================

This script analyzes the degree distribution of Watts-Strogatz networks
across different β (beta) values and compares them to power law distributions
commonly observed in real-world networks.

The Watts-Strogatz model interpolates between:
- β = 0: Regular ring lattice (all nodes have same degree)
- β = 1: Random network (Erdős-Rényi-like)
- 0 < β < 1: Small-world networks

Date: 2025
"""

# Import necessary libraries
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class WattsStrogatzDegreeAnalyzer:
    """
    A class to analyze degree distributions of Watts-Strogatz networks
    across different β values and compare with power law distributions.
    """
    
    def __init__(self, n=1000, k=6, num_graphs=100):
        """
        Initialize the analyzer with network parameters.
        
        Parameters:
        -----------
        n : int
            Number of nodes in each network (default: 1000)
        k : int
            Each node is connected to k nearest neighbors in ring topology
            (default: 6, so each node connects to 3 neighbors on each side)
        num_graphs : int
            Number of graphs to generate for each β value (default: 100)
        """
        self.n = n
        self.k = k
        self.num_graphs = num_graphs
        self.beta_values = [0, 0.05, 0.5, 1.0]
        self.degree_distributions = {}
        
        print(f"Watts-Strogatz Degree Distribution Analyzer")
        print(f"=" * 50)
        print(f"Network parameters:")
        print(f"  • Nodes (n): {self.n}")
        print(f"  • Nearest neighbors (k): {self.k}")
        print(f"  • Graphs per β: {self.num_graphs}")
        print(f"  • β values: {self.beta_values}")
        print(f"=" * 50)
    
    def generate_ws_graphs(self, beta):
        """
        Generate multiple Watts-Strogatz graphs for a given β value.
        
        Parameters:
        -----------
        beta : float
            Rewiring probability (0 ≤ β ≤ 1)
            
        Returns:
        --------
        list
            List of NetworkX graphs
        """
        graphs = []
        print(f"Generating {self.num_graphs} graphs for β = {beta}...")
        
        for i in range(self.num_graphs):
            # Generate Watts-Strogatz graph
            G = nx.watts_strogatz_graph(self.n, self.k, beta)
            graphs.append(G)
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{self.num_graphs} graphs")
        
        return graphs
    
    def calculate_degree_distribution(self, graphs):
        """
        Calculate the aggregated degree distribution from multiple graphs.
        
        Parameters:
        -----------
        graphs : list
            List of NetworkX graphs
            
        Returns:
        --------
        tuple
            (degrees, counts, probabilities)
        """
        all_degrees = []
        
        # Collect degrees from all graphs
        for graph in graphs:
            degrees = [degree for node, degree in graph.degree()]
            all_degrees.extend(degrees)
        
        # Count degree frequencies
        degree_counts = Counter(all_degrees)
        
        # Convert to arrays for plotting
        degrees = np.array(sorted(degree_counts.keys()))
        counts = np.array([degree_counts[d] for d in degrees])
        
        # Calculate probabilities (normalize by total number of nodes)
        total_nodes = len(all_degrees)
        probabilities = counts / total_nodes
        
        return degrees, counts, probabilities
    
    def generate_power_law_distribution(self, x_min=1, x_max=50, gamma=2.5):
        """
        Generate a power law distribution for comparison.
        
        Power law: P(k) ∝ k^(-γ)
        Typical values for real networks: γ ∈ [2, 3]
        
        Parameters:
        -----------
        x_min : int
            Minimum degree value
        x_max : int
            Maximum degree value
        gamma : float
            Power law exponent (default: 2.5, typical for real networks)
            
        Returns:
        --------
        tuple
            (degrees, probabilities)
        """
        degrees = np.arange(x_min, x_max + 1)
        
        # Power law: P(k) = C * k^(-gamma)
        # where C is normalization constant
        probabilities = degrees ** (-gamma)
        
        # Normalize to make it a proper probability distribution
        probabilities = probabilities / np.sum(probabilities)
        
        return degrees, probabilities
    
    def analyze_all_beta_values(self):
        """
        Analyze degree distributions for all β values.
        """
        print(f"\nStarting analysis for all β values...")
        print(f"This may take a few minutes...\n")
        
        for beta in self.beta_values:
            print(f"Analyzing β = {beta}")
            
            # Generate graphs for this β value
            graphs = self.generate_ws_graphs(beta)
            
            # Calculate degree distribution
            degrees, counts, probabilities = self.calculate_degree_distribution(graphs)
            
            # Store results
            self.degree_distributions[beta] = {
                'degrees': degrees,
                'counts': counts,
                'probabilities': probabilities
            }
            
            # Print basic statistics
            all_degrees = []
            for graph in graphs:
                all_degrees.extend([degree for node, degree in graph.degree()])
            
            mean_degree = np.mean(all_degrees)
            std_degree = np.std(all_degrees)
            min_degree = np.min(all_degrees)
            max_degree = np.max(all_degrees)
            
            print(f"  ✓ Degree statistics:")
            print(f"    Mean: {mean_degree:.2f}")
            print(f"    Std:  {std_degree:.2f}")
            print(f"    Range: [{min_degree}, {max_degree}]")
            print()
    
    def plot_degree_distributions(self, figsize=(15, 10)):
        """
        Plot degree distributions for all β values with power law comparison.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        axes = [ax1, ax2, ax3, ax4]
        
        # Color scheme for different β values
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Plot each β value in its own subplot
        for i, beta in enumerate(self.beta_values):
            ax = axes[i]
            data = self.degree_distributions[beta]
            
            # Plot degree distribution
            ax.bar(data['degrees'], data['probabilities'], 
                   alpha=0.7, color=colors[i], 
                   label=f'WS β={beta}', width=0.8)
            
            # Add power law for comparison (only for β > 0)
            if beta > 0:
                max_degree = np.max(data['degrees'])
                power_degrees, power_probs = self.generate_power_law_distribution(
                    x_min=1, x_max=max_degree, gamma=2.5
                )
                
                # Scale power law to match the data scale
                scale_factor = np.max(data['probabilities']) / np.max(power_probs) * 0.8
                scaled_power_probs = power_probs * scale_factor
                
                ax.plot(power_degrees, scaled_power_probs, 
                       'r--', linewidth=2, alpha=0.8,
                       label='Power Law (γ=2.5)')
            
            # Formatting
            ax.set_xlabel('Degree (k)', fontsize=12)
            ax.set_ylabel('Probability P(k)', fontsize=12)
            ax.set_title(f'β = {beta}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set reasonable axis limits
            ax.set_xlim(0, np.max(data['degrees']) + 1)
        
        # Overall title
        fig.suptitle('Watts-Strogatz Degree Distributions vs Power Law\n' + 
                    f'(n={self.n}, k={self.k}, {self.num_graphs} graphs per β)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('watts_strogatz_degree_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Degree distribution plots saved as 'watts_strogatz_degree_distributions.png'")
        plt.close()
    
    def plot_combined_comparison(self, figsize=(12, 8)):
        """
        Plot all degree distributions on the same figure for direct comparison.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Linear scale plot
        for i, beta in enumerate(self.beta_values):
            data = self.degree_distributions[beta]
            ax1.plot(data['degrees'], data['probabilities'], 
                    'o-', color=colors[i], alpha=0.8, 
                    label=f'β = {beta}', markersize=4)
        
        # Add power law reference
        max_degree = max([np.max(self.degree_distributions[beta]['degrees']) 
                         for beta in self.beta_values])
        power_degrees, power_probs = self.generate_power_law_distribution(
            x_min=1, x_max=max_degree, gamma=2.5
        )
        
        # Scale power law
        max_prob = max([np.max(self.degree_distributions[beta]['probabilities']) 
                       for beta in self.beta_values])
        scale_factor = max_prob / np.max(power_probs) * 0.5
        scaled_power_probs = power_probs * scale_factor
        
        ax1.plot(power_degrees, scaled_power_probs, 
                'k--', linewidth=2, alpha=0.8,
                label='Power Law (γ=2.5)')
        
        ax1.set_xlabel('Degree (k)', fontsize=12)
        ax1.set_ylabel('Probability P(k)', fontsize=12)
        ax1.set_title('Linear Scale', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Log-log scale plot
        for i, beta in enumerate(self.beta_values):
            data = self.degree_distributions[beta]
            # Filter out zero probabilities for log scale
            mask = data['probabilities'] > 0
            ax2.loglog(data['degrees'][mask], data['probabilities'][mask], 
                      'o-', color=colors[i], alpha=0.8, 
                      label=f'β = {beta}', markersize=4)
        
        # Power law on log-log scale
        mask = scaled_power_probs > 0
        ax2.loglog(power_degrees[mask], scaled_power_probs[mask], 
                  'k--', linewidth=2, alpha=0.8,
                  label='Power Law (γ=2.5)')
        
        ax2.set_xlabel('Degree (k)', fontsize=12)
        ax2.set_ylabel('Probability P(k)', fontsize=12)
        ax2.set_title('Log-Log Scale', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Overall title
        fig.suptitle('Watts-Strogatz Degree Distributions: Transition Analysis\n' + 
                    f'(n={self.n}, k={self.k}, {self.num_graphs} graphs per β)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('watts_strogatz_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Comparison plot saved as 'watts_strogatz_comparison.png'")
        plt.close()
    
    def print_analysis_summary(self):
        """
        Print a summary of the analysis results.
        """
        print(f"\n" + "=" * 60)
        print(f"ANALYSIS SUMMARY")
        print(f"=" * 60)
        
        for beta in self.beta_values:
            data = self.degree_distributions[beta]
            degrees = data['degrees']
            probs = data['probabilities']
            
            # Calculate statistics
            mean_degree = np.sum(degrees * probs)
            variance = np.sum((degrees - mean_degree)**2 * probs)
            std_degree = np.sqrt(variance)
            
            print(f"\nβ = {beta}:")
            print(f"  • Degree range: [{np.min(degrees)}, {np.max(degrees)}]")
            print(f"  • Mean degree: {mean_degree:.2f}")
            print(f"  • Std deviation: {std_degree:.2f}")
            print(f"  • Most common degree: {degrees[np.argmax(probs)]}")
            
            # Network type description
            if beta == 0:
                print(f"  • Network type: Regular ring lattice")
            elif beta == 1:
                print(f"  • Network type: Random network")
            else:
                print(f"  • Network type: Small-world network")
        
        print(f"\n" + "=" * 60)
        print(f"Key Observations:")
        print(f"• β = 0: All nodes have exactly degree {self.k} (regular)")
        print(f"• β > 0: Degree distribution broadens with increasing β")
        print(f"• β = 1: Approaches Poisson-like distribution (random)")
        print(f"• Power law: Typical of scale-free real networks")
        print(f"=" * 60)

def main():
    """
    Main function to run the complete Watts-Strogatz degree distribution analysis.
    """
    # Initialize analyzer
    # Using smaller network for faster computation, but still meaningful results
    analyzer = WattsStrogatzDegreeAnalyzer(n=1000, k=6, num_graphs=100)
    
    # Run analysis for all β values
    analyzer.analyze_all_beta_values()
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    analyzer.plot_degree_distributions()
    analyzer.plot_combined_comparison()
    
    # Print summary
    analyzer.print_analysis_summary()
    
    print(f"\n✓ Analysis complete! Check the generated PNG files.")

if __name__ == "__main__":
    main()