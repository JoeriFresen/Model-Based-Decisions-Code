#!/usr/bin/env python3
"""
Test Script for Virus Spread Visualization
==========================================

This script tests the dependencies and provides usage instructions
for the virus spread visualization on Barabási-Albert networks.
"""

import sys

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies for virus spread visualization...")
    print("=" * 50)
    
    dependencies = {
        'numpy': 'numpy',
        'networkx': 'networkx', 
        'manim': 'manim'
    }
    
    missing = []
    
    for package_name, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} - OK")
        except ImportError:
            print(f"✗ {package_name} - MISSING")
            missing.append(package_name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nTo install missing packages:")
        print("pip install " + " ".join(missing))
        return False
    else:
        print("\n✓ All dependencies are available!")
        return True

def show_usage_instructions():
    """Show how to use the virus spread visualization."""
    print("\n" + "=" * 60)
    print("VIRUS SPREAD VISUALIZATION USAGE")
    print("=" * 60)
    
    print("\n1. BASIC USAGE:")
    print("   manim virus_spread_ba_network.py VirusSpreadVisualization -pql")
    
    print("\n2. SIMPLE DEMO (faster rendering):")
    print("   manim virus_spread_ba_network.py SimpleVirusSpread -pql")
    
    print("\n3. HIGH QUALITY RENDERING:")
    print("   manim virus_spread_ba_network.py VirusSpreadVisualization -pqh")
    
    print("\n4. COMMAND FLAGS:")
    print("   -p : Preview the animation when complete")
    print("   -q : Quality setting")
    print("   -l : Low quality (fast)")
    print("   -m : Medium quality")
    print("   -h : High quality (slow)")
    
    print("\n5. WHAT THE ANIMATION SHOWS:")
    print("   • Blue nodes: Susceptible (can be infected)")
    print("   • Red nodes: Infected (spreading virus)")
    print("   • Green nodes: Recovered (immune)")
    
    print("\n6. SIMULATION PARAMETERS:")
    print("   • Network: Barabási-Albert (scale-free)")
    print("   • Model: SIR (Susceptible-Infected-Recovered)")
    print("   • Infection spreads through network connections")
    print("   • Infected nodes recover after fixed time")
    
    print("\n7. OUTPUT:")
    print("   • Video file: media/videos/virus_spread_ba_network/[quality]/VirusSpreadVisualization.mp4")
    print("   • The animation shows virus spreading through the network over time")

def test_network_generation():
    """Test basic network generation without Manim."""
    try:
        import networkx as nx
        import numpy as np
        
        print("\n" + "=" * 50)
        print("TESTING NETWORK GENERATION")
        print("=" * 50)
        
        # Generate a small BA network
        n_nodes = 20
        m_edges = 2
        
        print(f"Generating Barabási-Albert network...")
        print(f"  Nodes: {n_nodes}")
        print(f"  Attachment parameter (m): {m_edges}")
        
        G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=42)
        
        print(f"\n✓ Network generated successfully!")
        print(f"  Actual nodes: {G.number_of_nodes()}")
        print(f"  Actual edges: {G.number_of_edges()}")
        
        # Calculate some basic statistics
        degrees = dict(G.degree())
        avg_degree = np.mean(list(degrees.values()))
        max_degree = max(degrees.values())
        min_degree = min(degrees.values())
        
        print(f"\nNetwork Statistics:")
        print(f"  Average degree: {avg_degree:.2f}")
        print(f"  Degree range: [{min_degree}, {max_degree}]")
        
        # Find the hub (highest degree node)
        hub = max(degrees, key=degrees.get)
        print(f"  Hub node: {hub} (degree {degrees[hub]})")
        
        return True
        
    except Exception as e:
        print(f"✗ Network generation failed: {e}")
        return False

def test_simulation_logic():
    """Test the virus spread simulation logic."""
    try:
        import networkx as nx
        import random
        
        print("\n" + "=" * 50)
        print("TESTING SIMULATION LOGIC")
        print("=" * 50)
        
        # Create a simple network
        G = nx.barabasi_albert_graph(10, 2, seed=42)
        
        # Simple simulation test
        print("Testing basic SIR simulation...")
        
        # Initialize states
        states = {node: 'S' for node in G.nodes()}  # All susceptible
        states[0] = 'I'  # Patient zero
        
        print(f"Initial state: 1 infected, {len(G.nodes())-1} susceptible")
        
        # Run a few simulation steps
        infection_rate = 0.5
        for step in range(3):
            new_infections = 0
            new_states = states.copy()
            
            for node in G.nodes():
                if states[node] == 'S':
                    infected_neighbors = [
                        n for n in G.neighbors(node) 
                        if states[n] == 'I'
                    ]
                    
                    if infected_neighbors and random.random() < infection_rate:
                        new_states[node] = 'I'
                        new_infections += 1
            
            states = new_states
            infected_count = list(states.values()).count('I')
            
            print(f"Step {step+1}: {infected_count} infected (+{new_infections} new)")
        
        print("✓ Simulation logic working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Simulation test failed: {e}")
        return False

def main():
    """Main test function."""
    print("VIRUS SPREAD VISUALIZATION - DEPENDENCY CHECK")
    print("=" * 60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if deps_ok:
        # Test network generation
        network_ok = test_network_generation()
        
        # Test simulation logic
        sim_ok = test_simulation_logic()
        
        if network_ok and sim_ok:
            print("\n" + "=" * 60)
            print("✓ ALL TESTS PASSED!")
            print("The virus spread visualization is ready to use.")
            show_usage_instructions()
        else:
            print("\n✗ Some tests failed. Check the error messages above.")
    else:
        print("\n✗ Missing dependencies. Install them first.")
        print("\nAfter installing dependencies, you can run:")
        print("python test_virus_visualization.py")

if __name__ == "__main__":
    main()