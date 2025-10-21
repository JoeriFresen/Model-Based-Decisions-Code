#!/usr/bin/env python3
"""
Virus Spread Visualization on Barabási-Albert Networks
======================================================

This script creates an animated visualization of virus spread across a 
Barabási-Albert (BA) preferential attachment network using NetworkX for 
network generation and Manim for animation.

The simulation implements a simple SIR (Susceptible-Infected-Recovered) model:
- Susceptible nodes (blue) can become infected
- Infected nodes (red) can spread the virus to neighbors
- Infected nodes recover after a certain time (green)

The BA network represents scale-free networks commonly found in:
- Social networks
- Internet topology
- Biological networks
- Citation networks

Date: 2025
"""

import numpy as np
import networkx as nx
import random
from manim import *
from collections import defaultdict
import math

class VirusSpreadSimulation:
    """
    Handles the virus spread simulation logic on a network.
    Implements a discrete-time SIR (Susceptible-Infected-Recovered) model.
    """
    
    def __init__(self, graph, infection_rate=0.3, recovery_time=5, initial_infected=1):
        """
        Initialize the virus spread simulation.
        
        Parameters:
        -----------
        graph : networkx.Graph
            The network on which to simulate virus spread
        infection_rate : float
            Probability that an infected node infects a susceptible neighbor (0-1)
        recovery_time : int
            Number of time steps before an infected node recovers
        initial_infected : int
            Number of initially infected nodes
        """
        self.graph = graph
        self.infection_rate = infection_rate
        self.recovery_time = recovery_time
        self.initial_infected = initial_infected
        
        # Node states: 'S' = Susceptible, 'I' = Infected, 'R' = Recovered
        self.node_states = {}
        self.infection_times = {}  # Track how long nodes have been infected
        self.history = []  # Store state history for animation
        
        self.reset_simulation()
    
    def reset_simulation(self):
        """Reset the simulation to initial conditions."""
        # Initialize all nodes as susceptible
        for node in self.graph.nodes():
            self.node_states[node] = 'S'
            self.infection_times[node] = 0
        
        # Randomly select initial infected nodes (prefer high-degree nodes for realism)
        degrees = dict(self.graph.degree())
        # Weight selection by degree (hubs more likely to be patient zero)
        weights = [degrees[node] + 1 for node in self.graph.nodes()]
        initial_nodes = np.random.choice(
            list(self.graph.nodes()), 
            size=self.initial_infected, 
            replace=False,
            p=np.array(weights) / sum(weights)
        )
        
        for node in initial_nodes:
            self.node_states[node] = 'I'
            self.infection_times[node] = 1
        
        # Store initial state
        self.history = [self.node_states.copy()]
    
    def step(self):
        """Execute one time step of the simulation."""
        new_states = self.node_states.copy()
        new_infection_times = self.infection_times.copy()
        
        # Process each node
        for node in self.graph.nodes():
            current_state = self.node_states[node]
            
            if current_state == 'S':
                # Check if susceptible node gets infected
                infected_neighbors = [
                    neighbor for neighbor in self.graph.neighbors(node)
                    if self.node_states[neighbor] == 'I'
                ]
                
                # Probability of infection increases with number of infected neighbors
                infection_prob = 1 - (1 - self.infection_rate) ** len(infected_neighbors)
                
                if random.random() < infection_prob:
                    new_states[node] = 'I'
                    new_infection_times[node] = 1
            
            elif current_state == 'I':
                # Update infection time
                new_infection_times[node] = self.infection_times[node] + 1
                
                # Check if node recovers
                if self.infection_times[node] >= self.recovery_time:
                    new_states[node] = 'R'
                    new_infection_times[node] = 0
        
        # Update states
        self.node_states = new_states
        self.infection_times = new_infection_times
        
        # Store state in history
        self.history.append(self.node_states.copy())
        
        # Return True if simulation should continue (there are still infected nodes)
        return 'I' in self.node_states.values()
    
    def run_simulation(self, max_steps=50):
        """Run the complete simulation."""
        self.reset_simulation()
        
        for step in range(max_steps):
            if not self.step():
                break  # No more infected nodes
        
        return self.history
    
    def get_statistics(self):
        """Get current simulation statistics."""
        states = list(self.node_states.values())
        return {
            'susceptible': states.count('S'),
            'infected': states.count('I'),
            'recovered': states.count('R'),
            'total': len(states)
        }

class VirusSpreadVisualization(Scene):
    """
    Manim scene for visualizing virus spread on a Barabási-Albert network.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Network parameters
        self.n_nodes = 50  # Number of nodes
        self.m_edges = 2   # Number of edges to attach from new node (BA parameter)
        
        # Simulation parameters
        self.infection_rate = 0.4
        self.recovery_time = 4
        self.initial_infected = 2
        
        # Animation parameters
        self.step_duration = 0.8  # Duration of each time step
        self.node_radius = 0.15
        
        # Colors for different states
        self.colors = {
            'S': BLUE,      # Susceptible
            'I': RED,       # Infected
            'R': GREEN      # Recovered
        }
        
        # Generate network and simulation
        self.generate_network()
        self.setup_simulation()
    
    def generate_network(self):
        """Generate a Barabási-Albert network."""
        # Create BA network
        self.graph = nx.barabasi_albert_graph(self.n_nodes, self.m_edges, seed=42)
        
        # Calculate layout using spring layout for better visualization
        self.pos = nx.spring_layout(self.graph, k=3, iterations=50, seed=42)
        
        # Scale positions to fit Manim coordinate system
        scale = 3.5
        for node in self.pos:
            self.pos[node] = np.array([
                self.pos[node][0] * scale,
                self.pos[node][1] * scale,
                0
            ])
    
    def setup_simulation(self):
        """Setup the virus spread simulation."""
        self.simulation = VirusSpreadSimulation(
            self.graph,
            infection_rate=self.infection_rate,
            recovery_time=self.recovery_time,
            initial_infected=self.initial_infected
        )
        
        # Run simulation to get complete history
        self.history = self.simulation.run_simulation(max_steps=30)
    
    def create_network_mobjects(self):
        """Create Manim objects for the network."""
        # Create edges
        self.edge_mobjects = VGroup()
        for edge in self.graph.edges():
            start_pos = self.pos[edge[0]]
            end_pos = self.pos[edge[1]]
            line = Line(start_pos, end_pos, stroke_width=1, color=GRAY)
            self.edge_mobjects.add(line)
        
        # Create nodes
        self.node_mobjects = {}
        for node in self.graph.nodes():
            circle = Circle(
                radius=self.node_radius,
                color=WHITE,
                fill_opacity=1,
                stroke_width=2
            ).move_to(self.pos[node])
            
            # Add node label
            label = Text(
                str(node),
                font_size=16,
                color=BLACK
            ).move_to(self.pos[node])
            
            node_group = VGroup(circle, label)
            self.node_mobjects[node] = node_group
        
        # Group all nodes
        self.all_nodes = VGroup(*self.node_mobjects.values())
    
    def update_node_colors(self, states):
        """Update node colors based on current states."""
        animations = []
        
        for node, state in states.items():
            circle = self.node_mobjects[node][0]  # Circle is first element
            target_color = self.colors[state]
            
            if circle.fill_color != target_color:
                animations.append(
                    circle.animate.set_fill(target_color)
                )
        
        return animations
    
    def create_legend(self):
        """Create a legend explaining the node colors."""
        legend_items = []
        
        # Title
        title = Text("Node States", font_size=24, color=WHITE)
        legend_items.append(title)
        
        # State explanations
        states_info = [
            ("Susceptible", BLUE),
            ("Infected", RED),
            ("Recovered", GREEN)
        ]
        
        for i, (state_name, color) in enumerate(states_info):
            # Create colored circle
            circle = Circle(radius=0.15, color=color, fill_opacity=1)
            
            # Create text
            text = Text(state_name, font_size=18, color=WHITE)
            
            # Position items
            item = VGroup(circle, text.next_to(circle, RIGHT, buff=0.2))
            legend_items.append(item)
        
        # Arrange legend vertically
        legend = VGroup(*legend_items).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        legend.to_corner(UL, buff=0.5)
        
        return legend
    
    def create_statistics_display(self):
        """Create a display for simulation statistics."""
        self.stats_text = Text("", font_size=16, color=WHITE)
        self.stats_text.to_corner(UR, buff=0.5)
        return self.stats_text
    
    def update_statistics(self, states):
        """Update the statistics display."""
        state_counts = {'S': 0, 'I': 0, 'R': 0}
        for state in states.values():
            state_counts[state] += 1
        
        stats_str = f"Susceptible: {state_counts['S']}\n"
        stats_str += f"Infected: {state_counts['I']}\n"
        stats_str += f"Recovered: {state_counts['R']}"
        
        return self.stats_text.animate.become(
            Text(stats_str, font_size=16, color=WHITE).to_corner(UR, buff=0.5)
        )
    
    def construct(self):
        """Main animation construction method."""
        # Set background color
        self.camera.background_color = "#1e1e1e"
        
        # Create title
        title = Text(
            "Virus Spread on Barabási-Albert Network",
            font_size=32,
            color=WHITE
        ).to_edge(UP, buff=0.5)
        
        # Create subtitle with parameters
        subtitle = Text(
            f"SIR Model: Infection Rate = {self.infection_rate}, Recovery Time = {self.recovery_time} steps",
            font_size=18,
            color=GRAY
        ).next_to(title, DOWN, buff=0.2)
        
        # Create network visualization
        self.create_network_mobjects()
        
        # Create legend and statistics
        legend = self.create_legend()
        stats_display = self.create_statistics_display()
        
        # Initial scene setup
        self.play(
            Write(title),
            Write(subtitle),
            run_time=2
        )
        
        self.play(
            Create(self.edge_mobjects),
            run_time=2
        )
        
        self.play(
            Create(self.all_nodes),
            Create(legend),
            Create(stats_display),
            run_time=2
        )
        
        # Set initial colors
        initial_states = self.history[0]
        initial_animations = self.update_node_colors(initial_states)
        stats_animation = self.update_statistics(initial_states)
        
        if initial_animations:
            self.play(*initial_animations, stats_animation, run_time=1)
        
        # Animate through simulation steps
        for step, states in enumerate(self.history[1:], 1):
            # Create step indicator
            step_text = Text(
                f"Time Step: {step}",
                font_size=20,
                color=YELLOW
            ).to_corner(DL, buff=0.5)
            
            # Update node colors and statistics
            color_animations = self.update_node_colors(states)
            stats_animation = self.update_statistics(states)
            
            animations = color_animations + [stats_animation]
            
            # Add step indicator for first few steps
            if step <= 3:
                if hasattr(self, 'current_step_text'):
                    animations.append(
                        Transform(self.current_step_text, step_text)
                    )
                else:
                    self.current_step_text = step_text
                    animations.append(Write(self.current_step_text))
            elif hasattr(self, 'current_step_text'):
                animations.append(
                    Transform(self.current_step_text, step_text)
                )
            
            if animations:
                self.play(*animations, run_time=self.step_duration)
            else:
                self.wait(self.step_duration)
        
        # Final summary
        final_states = self.history[-1]
        final_stats = {'S': 0, 'I': 0, 'R': 0}
        for state in final_states.values():
            final_stats[state] += 1
        
        summary_text = Text(
            f"Simulation Complete!\n"
            f"Final State: {final_stats['R']} Recovered, "
            f"{final_stats['S']} Never Infected",
            font_size=20,
            color=YELLOW
        ).to_edge(DOWN, buff=0.5)
        
        self.play(Write(summary_text), run_time=2)
        self.wait(3)

# Additional utility functions for running the visualization

def create_simple_demo():
    """Create a simpler demo version with fewer nodes for faster rendering."""
    
    class SimpleVirusSpread(VirusSpreadVisualization):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.n_nodes = 25  # Fewer nodes for faster rendering
            self.m_edges = 2
            self.step_duration = 1.0  # Slower for better visibility
            
            # Regenerate with new parameters
            self.generate_network()
            self.setup_simulation()
    
    return SimpleVirusSpread

def main():
    """
    Main function to run the virus spread visualization.
    
    To render the animation, run:
    manim virus_spread_ba_network.py VirusSpreadVisualization -pql
    
    For a simpler demo:
    manim virus_spread_ba_network.py SimpleVirusSpread -pql
    """
    print("Virus Spread Visualization on Barabási-Albert Networks")
    print("=" * 60)
    print("This script creates an animated visualization of virus spread")
    print("using the SIR (Susceptible-Infected-Recovered) model.")
    print()
    print("To render the animation, use one of these commands:")
    print("  manim virus_spread_ba_network.py VirusSpreadVisualization -pql")
    print("  manim virus_spread_ba_network.py SimpleVirusSpread -pql")
    print()
    print("Flags:")
    print("  -p: Preview the animation")
    print("  -q: Quality (l=low, m=medium, h=high)")
    print("  -l: Low quality for faster rendering")
    print()
    print("The animation shows:")
    print("  • Blue nodes: Susceptible")
    print("  • Red nodes: Infected") 
    print("  • Green nodes: Recovered")
    print()
    print("The Barabási-Albert network represents scale-free networks")
    print("commonly found in social networks, internet topology, etc.")

# Create the simple demo class
SimpleVirusSpread = create_simple_demo()

if __name__ == "__main__":
    main()