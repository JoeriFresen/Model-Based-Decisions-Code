#!/usr/bin/env python3
"""
Airline Routes Network Analysis

This script analyzes the airline routes network from the OpenFlights dataset.
It creates a network graph and computes various network statistics including:
- Degree distribution
- Top 10 airports by different centrality measures

Data format (CSV):
Airline, Airline ID, Source airport, Source airport ID, Destination airport, 
Destination airport ID, Codeshare, Stops, Equipment

Each entry contains the following information:

Airline	2-letter (IATA) or 3-letter (ICAO) code of the airline.
Airline ID	Unique OpenFlights identifier for airline (see Airline).
Source airport	3-letter (IATA) or 4-letter (ICAO) code of the source airport.
Source airport ID	Unique OpenFlights identifier for source airport (see Airport)
Destination airport	3-letter (IATA) or 4-letter (ICAO) code of the destination airport.
Destination airport ID	Unique OpenFlights identifier for destination airport (see Airport)
Codeshare	"Y" if this flight is a codeshare (that is, not operated by Airline, but another carrier), empty otherwise.
Stops	Number of stops on this flight ("0" for direct)
Equipment	3-letter codes for plane type(s) generally used on this flight, separated by spaces
The data is UTF-8 encoded. The special value \N is used for "NULL" to indicate that no value is available, and is understood automatically by MySQL if imported.

Notes:
Routes are directional: if an airline operates services from A to B and from B to A, both A-B and B-A are listed separately.
Routes where one carrier operates both its own and codeshare flights are listed only once.
Sample entries
BA,1355,SIN,3316,LHR,507,,0,744 777
BA,1355,SIN,3316,MEL,3339,Y,0,744
TOM,5013,ACE,1055,BFS,465,,0,320

Taken from https://openflights.org/data

"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns

def load_airline_data(filepath):
    """
    Load airline routes data from CSV file.
    
    Args:
        filepath (str): Path to the airline routes data file
        
    Returns:
        pandas.DataFrame: Loaded and cleaned data
    """
    # Define column names based on the data description
    columns = [
        'airline_code', 'airline_id', 'source_airport', 'source_airport_id',
        'dest_airport', 'dest_airport_id', 'codeshare', 'stops', 'equipment'
    ]
    
    # Load data
    print("Loading airline routes data...")
    df = pd.read_csv(filepath, names=columns, na_values=['\\N'])
    
    print(f"Loaded {len(df)} routes")
    print(f"Data shape: {df.shape}")
    
    # Remove rows with missing airport codes
    df_clean = df.dropna(subset=['source_airport', 'dest_airport'])
    print(f"After removing missing airport codes: {len(df_clean)} routes")
    
    return df_clean

def create_airport_network(df):
    """
    Create a network graph from airline routes data.
    
    Args:
        df (pandas.DataFrame): Airline routes data
        
    Returns:
        networkx.Graph: Undirected graph of airport connections
    """
    print("Creating airport network...")
    
    # Create undirected graph (treating routes as bidirectional connections)
    G = nx.Graph()
    
    # Add edges for each route
    for _, row in df.iterrows():
        source = row['source_airport']
        dest = row['dest_airport']
        
        # Skip self-loops
        if source != dest:
            if G.has_edge(source, dest):
                # Increment weight if edge already exists
                G[source][dest]['weight'] += 1
            else:
                # Add new edge with weight 1
                G.add_edge(source, dest, weight=1)
    
    print(f"Network created with {G.number_of_nodes()} airports and {G.number_of_edges()} connections")
    
    return G

def plot_degree_distribution(G):
    """
    Plot the degree distribution of the network.
    
    Args:
        G (networkx.Graph): Network graph
    """
    print("Plotting degree distribution...")
    
    # Get degree sequence
    degrees = [G.degree(n) for n in G.nodes()]
    degree_counts = Counter(degrees)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale plot
    degrees_list = list(degree_counts.keys())
    counts_list = list(degree_counts.values())
    
    ax1.bar(degrees_list, counts_list, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Number of Airports')
    ax1.set_title('Degree Distribution (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    
    # Log-log scale plot
    ax2.loglog(degrees_list, counts_list, 'bo', alpha=0.7, markersize=4)
    ax2.set_xlabel('Degree (log scale)')
    ax2.set_ylabel('Number of Airports (log scale)')
    ax2.set_title('Degree Distribution (Log-Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('degree_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print basic statistics
    print(f"\nDegree Distribution Statistics:")
    print(f"Mean degree: {np.mean(degrees):.2f}")
    print(f"Median degree: {np.median(degrees):.2f}")
    print(f"Max degree: {max(degrees)}")
    print(f"Min degree: {min(degrees)}")
    print(f"Standard deviation: {np.std(degrees):.2f}")

def calculate_centrality_measures(G):
    """
    Calculate various centrality measures for the network.
    
    Args:
        G (networkx.Graph): Network graph
        
    Returns:
        dict: Dictionary containing centrality measures
    """
    print("Calculating centrality measures...")
    
    centralities = {}
    
    # Degree centrality
    print("  - Degree centrality...")
    centralities['degree'] = nx.degree_centrality(G)
    
    # Betweenness centrality (using sampling for large networks)
    print("  - Betweenness centrality...")
    if G.number_of_nodes() > 1000:
        # Use sampling for large networks to speed up computation
        k = min(1000, G.number_of_nodes())
        centralities['betweenness'] = nx.betweenness_centrality(G, k=k)
        print(f"    (using sampling with k={k} nodes)")
    else:
        centralities['betweenness'] = nx.betweenness_centrality(G)
    
    # Closeness centrality
    print("  - Closeness centrality...")
    # Only calculate for the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc)
    closeness_largest = nx.closeness_centrality(G_largest)
    
    # Extend to full graph with 0 for disconnected nodes
    centralities['closeness'] = {node: closeness_largest.get(node, 0) for node in G.nodes()}
    
    # Eigenvector centrality
    print("  - Eigenvector centrality...")
    try:
        centralities['eigenvector'] = nx.eigenvector_centrality(G_largest, max_iter=1000)
        # Extend to full graph with 0 for disconnected nodes
        centralities['eigenvector'] = {node: centralities['eigenvector'].get(node, 0) for node in G.nodes()}
    except nx.PowerIterationFailedConvergence:
        print("    Warning: Eigenvector centrality failed to converge, using PageRank instead")
        centralities['eigenvector'] = nx.pagerank(G)
    
    return centralities

def display_top_airports(centralities, top_n=10):
    """
    Display top N airports for each centrality measure.
    
    Args:
        centralities (dict): Dictionary of centrality measures
        top_n (int): Number of top airports to display
    """
    print(f"\n{'='*60}")
    print(f"TOP {top_n} AIRPORTS BY CENTRALITY MEASURES")
    print(f"{'='*60}")
    
    for measure_name, measure_values in centralities.items():
        print(f"\n{measure_name.upper()} CENTRALITY:")
        print("-" * 40)
        
        # Sort airports by centrality value
        sorted_airports = sorted(measure_values.items(), key=lambda x: x[1], reverse=True)
        
        for i, (airport, value) in enumerate(sorted_airports[:top_n], 1):
            print(f"{i:2d}. {airport:4s} - {value:.6f}")

def network_summary(G, df):
    """
    Print a summary of the network properties.
    
    Args:
        G (networkx.Graph): Network graph
        df (pandas.DataFrame): Original data
    """
    print(f"\n{'='*60}")
    print("NETWORK SUMMARY")
    print(f"{'='*60}")
    
    print(f"Number of airports (nodes): {G.number_of_nodes()}")
    print(f"Number of connections (edges): {G.number_of_edges()}")
    print(f"Number of routes in dataset: {len(df)}")
    
    # Connected components
    num_components = nx.number_connected_components(G)
    largest_cc_size = len(max(nx.connected_components(G), key=len))
    
    print(f"Number of connected components: {num_components}")
    print(f"Size of largest connected component: {largest_cc_size}")
    print(f"Percentage in largest component: {largest_cc_size/G.number_of_nodes()*100:.1f}%")
    
    # Density
    density = nx.density(G)
    print(f"Network density: {density:.6f}")
    
    # Average clustering coefficient
    avg_clustering = nx.average_clustering(G)
    print(f"Average clustering coefficient: {avg_clustering:.4f}")

def main():
    """Main function to run the airline network analysis."""
    
    # File path
    data_file = "datasets/airline_routes.txt"
    
    try:
        # Load data
        df = load_airline_data(data_file)
        
        # Create network
        G = create_airport_network(df)
        
        # Network summary
        network_summary(G, df)
        
        # Plot degree distribution
        plot_degree_distribution(G)
        
        # Calculate centrality measures
        centralities = calculate_centrality_measures(G)
        
        # Display top airports
        display_top_airports(centralities, top_n=10)
        
        print(f"\n{'='*60}")
        print("Analysis complete! Degree distribution plot saved as 'degree_distribution.png'")
        print(f"{'='*60}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the data file '{data_file}'")
        print("Please make sure the file exists in the correct location.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()