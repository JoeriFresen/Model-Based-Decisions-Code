# Virus Spread Visualization on Barabási-Albert Networks

This project provides a comprehensive Python implementation for visualizing virus spread on scale-free networks using NetworkX and Manim. The visualization demonstrates how infectious diseases spread through complex networks using the SIR (Susceptible-Infected-Recovered) epidemiological model.

## 🎯 Features

- **Barabási-Albert Network Generation**: Creates scale-free networks with realistic degree distributions
- **SIR Epidemic Model**: Implements discrete-time Susceptible-Infected-Recovered dynamics
- **Interactive Manim Animations**: Beautiful, educational visualizations of virus spread
- **Customizable Parameters**: Adjustable network size, infection rates, and recovery times
- **Multiple Animation Modes**: Simple demo and comprehensive visualization options
- **Educational Focus**: Detailed comments and beginner-friendly code structure

## 📁 Project Files

### Core Scripts
- **`virus_spread_ba_network.py`** - Main visualization script with Manim animations
- **`test_virus_visualization.py`** - Dependency checker and testing utilities

### Generated Content
- **`media/videos/`** - Output directory for rendered animations
- **`SimpleVirusSpread.mp4`** - Demo animation showing virus spread

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install numpy networkx manim
```

### 2. Run Dependency Check

```bash
python test_virus_visualization.py
```

### 3. Generate Simple Demo Animation

```bash
manim virus_spread_ba_network.py SimpleVirusSpread -pql
```

### 4. Generate Full Visualization

```bash
manim virus_spread_ba_network.py VirusSpreadVisualization -pql
```

## 🎬 Animation Options

### Quality Settings
- **`-ql`** - Low quality (480p, fast rendering)
- **`-qm`** - Medium quality (720p, balanced)
- **`-qh`** - High quality (1080p, slow rendering)

### Additional Flags
- **`-p`** - Preview animation when complete
- **`-s`** - Save last frame as image
- **`--write_to_movie`** - Force video output

### Example Commands

```bash
# Quick demo (recommended for testing)
manim virus_spread_ba_network.py SimpleVirusSpread -pql

# High-quality full visualization
manim virus_spread_ba_network.py VirusSpreadVisualization -pqh

# Save final frame as image
manim virus_spread_ba_network.py SimpleVirusSpread -pql -s
```

## 🧬 Simulation Model

### Network Structure
- **Type**: Barabási-Albert (scale-free)
- **Properties**: Power-law degree distribution, small-world characteristics
- **Parameters**: Configurable number of nodes and attachment preference

### Epidemic Model
- **Type**: SIR (Susceptible-Infected-Recovered)
- **States**:
  - 🔵 **Susceptible**: Can become infected
  - 🔴 **Infected**: Spreading virus to neighbors
  - 🟢 **Recovered**: Immune to reinfection

### Dynamics
1. **Infection**: Susceptible nodes become infected based on infected neighbors
2. **Recovery**: Infected nodes recover after a fixed time period
3. **Immunity**: Recovered nodes cannot be reinfected

## 🎨 Visualization Features

### Node Representation
- **Color Coding**: Blue (Susceptible), Red (Infected), Green (Recovered)
- **Size Scaling**: Node size reflects degree (connectivity)
- **Dynamic Updates**: Real-time state changes during simulation

### Animation Elements
- **Network Layout**: Force-directed positioning for clarity
- **Smooth Transitions**: Animated state changes with timing
- **Progress Tracking**: Step counter and statistics display
- **Legend**: Clear state identification

## ⚙️ Customization

### Network Parameters
```python
# In virus_spread_ba_network.py
n_nodes = 30          # Number of nodes
m_edges = 2           # Attachment parameter
seed = 42             # Random seed for reproducibility
```

### Simulation Parameters
```python
infection_rate = 0.3   # Probability of infection per contact
recovery_time = 3      # Steps until recovery
initial_infected = 1   # Number of initial infected nodes
```

### Animation Parameters
```python
animation_speed = 1.0  # Speed multiplier
node_scale = 0.3      # Base node size
edge_opacity = 0.6    # Edge transparency
```

## 📊 Educational Applications

### Network Science Concepts
- **Scale-free networks**: Understanding hub-based connectivity
- **Degree distribution**: Power-law vs. random networks
- **Network resilience**: Impact of hub removal

### Epidemiology Concepts
- **Disease spread**: Contact-based transmission
- **Herd immunity**: Population-level protection
- **Intervention strategies**: Vaccination, isolation

### Computational Methods
- **Graph algorithms**: Network generation and analysis
- **Simulation techniques**: Discrete-time modeling
- **Visualization**: Scientific animation with Manim

## 🔧 Technical Details

### Dependencies
- **NumPy**: Numerical computations and random number generation
- **NetworkX**: Graph creation, manipulation, and analysis
- **Manim**: Mathematical animation and visualization

### Performance Considerations
- **Network Size**: Larger networks require more computation time
- **Animation Quality**: Higher quality increases rendering time
- **Memory Usage**: Complex animations may require significant RAM

### File Structure
```
project/
├── virus_spread_ba_network.py    # Main visualization script
├── test_virus_visualization.py   # Testing and validation
├── README_virus_spread.md        # This documentation
└── media/                        # Generated animations
    └── videos/
        └── virus_spread_ba_network/
            ├── 480p15/           # Low quality videos
            ├── 720p30/           # Medium quality videos
            └── 1080p60/          # High quality videos
```

## 🐛 Troubleshooting

### Common Issues

1. **Manim Installation Problems**
   ```bash
   # Try installing with conda
   conda install -c conda-forge manim
   
   # Or update pip first
   pip install --upgrade pip
   pip install manim
   ```

2. **Animation Not Playing**
   - Check if video player supports MP4
   - Try different quality settings
   - Ensure sufficient disk space

3. **Slow Rendering**
   - Use lower quality settings (`-ql`)
   - Reduce network size
   - Close other applications

4. **Import Errors**
   ```bash
   # Verify installations
   python -c "import numpy, networkx, manim; print('All imports successful')"
   ```

### Getting Help
- Run `python test_virus_visualization.py` for diagnostics
- Check Manim documentation: https://docs.manim.community/
- NetworkX documentation: https://networkx.org/

## 📚 Further Reading

### Network Science
- Barabási, A.L. "Network Science" (2016)
- Newman, M.E.J. "Networks: An Introduction" (2010)

### Epidemiological Modeling
- Keeling, M.J. & Rohani, P. "Modeling Infectious Diseases" (2008)
- Pastor-Satorras, R. et al. "Epidemic processes in complex networks" (2015)

### Computational Tools
- Manim Community Documentation
- NetworkX Tutorial and Reference
- Python Scientific Computing Ecosystem

## 📄 License

This project is created for educational purposes. Feel free to use, modify, and distribute for academic and learning applications.

---

**Created for Model Based Decisions Course - Week Two: Networks**  
*Demonstrating virus spread dynamics on complex networks*