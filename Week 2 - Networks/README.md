# Week Two - Networks: Visualization Suite

This directory contains comprehensive network analysis and visualization tools for studying both real-world and theoretical networks.

## 📁 Files Overview

### 🔧 Analysis Programs
- **`network_visualization_suite.py`** - Comprehensive comparison of Facebook network with theoretical models (ER, WS, BA)
- **`facebook_network_explorer.py`** - Detailed analysis of the Facebook social network dataset

### 📊 Generated Visualizations
- **`network_properties_comparison.png`** - Side-by-side comparison of network properties
- **`network_visualizations_spring.png`** - Network layouts using spring-force algorithm
- **`network_visualizations_kamada_kawai.png`** - Network layouts using Kamada-Kawai algorithm
- **`facebook_degree_analysis.png`** - Detailed degree distribution analysis
- **`facebook_centrality_analysis.png`** - Centrality measures comparison
- **`facebook_network_sample.png`** - Sample network visualization with node properties

### 📂 Data
- **`datasets/facebook_combined.txt.gz`** - Facebook social network dataset (4,039 nodes, 88,234 edges)

## 🚀 How to Run

### Quick Start - Network Comparison
```bash
python network_visualization_suite.py
```
This will:
- Load the Facebook dataset
- Generate equivalent theoretical networks (ER, WS, BA)
- Compare their properties
- Create comprehensive visualizations

### Detailed Facebook Analysis
```bash
python facebook_network_explorer.py
```
This provides:
- Detailed network statistics
- Degree distribution analysis (including power-law fitting)
- Centrality analysis (degree, betweenness, closeness, eigenvector)
- Network sample visualizations
- Comparison with random networks

## 📈 Key Findings

### Facebook Network Properties
- **4,039 users** with **88,234 friendships**
- **Average degree**: 43.7 friends per user
- **High clustering**: 0.6055 (56x higher than random!)
- **Short paths**: Average path length of 3.69
- **Scale-free**: Power-law degree distribution with γ ≈ 1.5

### Network Model Comparison
| Network Type | Clustering | Path Length | Key Property |
|--------------|------------|-------------|--------------|
| **Facebook** | 0.6055 | 3.69 | Small-world + Scale-free |
| **Erdős-Rényi** | 0.0108 | 2.61 | Random connections |
| **Watts-Strogatz** | 0.4645 | 5.70 | Small-world |
| **Barabási-Albert** | 0.0475 | 2.95 | Scale-free |

## 🎯 Educational Objectives

This suite demonstrates:

1. **Real-world vs Theoretical Networks**
   - Why simple random graphs (ER) fail to capture social network properties
   - How different models capture different aspects of real networks

2. **Small-World Phenomenon**
   - High clustering (friends of friends are friends)
   - Short path lengths (six degrees of separation)

3. **Scale-Free Properties**
   - Power-law degree distributions
   - Presence of highly connected "hubs"

4. **Network Analysis Techniques**
   - Centrality measures and their interpretations
   - Community detection and visualization
   - Statistical analysis of network properties

## 🔍 Visualization Features

- **Multiple Layout Algorithms**: Spring-force, Kamada-Kawai for different perspectives
- **Node Coloring**: By degree, clustering coefficient, centrality measures
- **Statistical Plots**: Degree distributions, centrality histograms, property comparisons
- **Comparative Analysis**: Side-by-side network property comparisons

## 📚 Dependencies

The programs use standard Python scientific libraries:
- `networkx` - Network analysis
- `matplotlib` - Plotting and visualization
- `numpy` - Numerical computations
- `pandas` - Data handling
- `seaborn` - Statistical plotting

## 💡 Usage Tips

1. **For Large Networks**: The programs automatically sample nodes for visualization to maintain performance
2. **Customization**: Modify `sample_size` parameters to adjust visualization detail
3. **Layout Choice**: Try different layouts (`spring`, `kamada_kawai`) for different insights
4. **Analysis Depth**: Use the Facebook explorer for detailed analysis, the suite for quick comparisons

## 🎓 Learning Outcomes

After running these analyses, students will understand:
- Why real-world networks differ from random graphs
- The importance of clustering in social networks
- How to measure and interpret network centrality
- The trade-offs between different network models
- Practical network analysis and visualization techniques