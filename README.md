# Model Based Decisions — Complex Systems & Policy (MSc)

This repository hosts materials for the course "Model Based Decisions" in the Master programme Complex Systems & Policy. It contains scripts, notebooks, and datasets used to explore networks, dynamics on networks, and agent-based modelling. The repo is organized by week and is intended for students to run, extend, and experiment with the provided models.

## Repository Structure
- `Week 1 - Introduction/`
  - Introductory notebooks and materials (cellular automata, Git intro, LaTeX slide sources).
- `Week 2 - Networks/`
  - Python scripts and datasets for network analysis and visualization:
    - `network_visualization_suite.py`, `network_visual_comparison.py`
    - Facebook and Watts–Strogatz analyses, centrality, ER vs. real networks comparisons
    - Virus spread experiments on Barabási–Albert networks
- `Week 3 - Dynamics on Networks/`
  - Notebooks exploring threshold models, robustness, and topology-dependent spreading.
- `Week 4 - Agent-based Models & Stochastic Simulation/`
  - Notebooks and models for ABM and stochastic simulation (e.g., Sugarscape, Axelrod).
- `requirements.txt`
  - Consolidated Python dependencies for running scripts and notebooks.

## Learning Objectives
- Build and reason about models to support decisions in complex systems.
- Analyze and visualize network structure and dynamics (random, small-world, scale-free).
- Explore spreading processes, thresholds, and robustness on networks.
- Design and implement agent-based models and stochastic simulations.

## Prerequisites
- Python 3.11+ (3.12 recommended)
- `pip` and a virtual environment tool (`python -m venv` or Conda)
- Recommended: Jupyter Lab/Notebook for notebooks

## Setup
1. Create and activate a virtual environment:
   - macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     py -3 -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data & Large Files
- Datasets for Week 2 reside under `Week 2 - Networks/datasets/`.
- Large archives and generated artifacts are excluded from version control by `.gitignore`.
- If a dataset is missing, consult the corresponding `README` in the week folder or place the file under its datasets directory.

## How to Run
- Scripts (examples):
  - Run the main network visualization suite:
    ```bash
    python "Week 2 - Networks/network_visualization_suite.py"
    ```
  - Compare Facebook network with Erdős–Rényi:
    ```bash
    python "Week 2 - Networks/facebook_vs_erdos_renyi.py"
    ```
  - Watts–Strogatz visualization:
    ```bash
    python "Week 2 - Networks/watts_strogatz_visualization.py"
    ```
- Notebooks:
  - Launch Jupyter and open the relevant notebook:
    ```bash
    pip install jupyter
    jupyter lab
    ```
    Then open notebooks in `Week 1`, `Week 3`, or `Week 4`.

## Tests
- Some weeks include lightweight tests. Example:
  ```bash
  python -m pytest "Week 2 - Networks/test_virus_visualization.py"
  ```

## Git Workflow
- Keep changes focused and commit with clear messages.
- Do not commit generated data, caches, or local environment files—these are covered by `.gitignore`.
- Use branches for larger features/experiments; open pull requests for review.

## Academic Integrity
- You may experiment and extend provided code. Ensure submitted work reflects your own understanding and properly cites external sources.

## Contact
- Instructor: M. H. Lees
- Email: m.h.lees@uva.nl

## Acknowledgements
- Uses common libraries such as NetworkX, NumPy, Matplotlib, and Jupyter.
- Some datasets are derived from public sources (e.g., social network datasets).