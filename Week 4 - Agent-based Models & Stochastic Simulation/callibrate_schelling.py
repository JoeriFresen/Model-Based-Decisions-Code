"""
Schelling segregation model and a simple calibration-by-simulation workflow.

This script implements a minimal version of the Schelling model on a toroidal
grid and demonstrates how to calibrate the tolerance parameter using synthetic
data and summary statistics. The workflow is:

1) Generate synthetic data at a known "true" tolerance `p_true` by running the
   model and computing summary statistics.
2) For a grid of candidate tolerances, repeatedly simulate the model,
   compute the same summary statistics, and aggregate them (mean across runs).
3) Compare each candidate’s mean statistics to the synthetic target using a
   simple squared-error metric; select the parameter that minimizes this error.

Design decisions and conventions:
- Two groups (A and B) and empty cells; constants `EMPTY`, `GROUP_A`, `GROUP_B`.
- Moore neighborhoods with wrap-around (torus) indexing.
- Agents move if their fraction of similar neighbors is below `tolerance`.
- Movement pairs each unhappy agent with an empty cell, in random order.
- Summary statistics include overall segregation index and multiscale block
  fractions of A and B; together they form a feature vector used for matching.

This approach mirrors the method of simulated moments: choose parameters so that
model-generated summary statistics match those observed (here, a synthetic
target) as closely as possible under a chosen distance function.
"""

import numpy as np

EMPTY = 0
GROUP_A = 1
GROUP_B = 2


def initialize_grid(size=40, density=0.9, frac_group_a=0.5, rng=None):
    """
    Create a `size` x `size` grid with two groups and empty cells.

    Parameters
    - size: Grid side length (grid is `size x size`).
    - density: Fraction of cells occupied (rest are `EMPTY`).
    - frac_group_a: Fraction of occupied agents that belong to `GROUP_A`.
    - rng: Optional NumPy random generator for reproducibility.

    Returns
    - grid: 2D NumPy array of ints in {EMPTY, GROUP_A, GROUP_B}.

    Notes
    - Population composition is set by counts, then shuffled uniformly to place
      agents and empty cells randomly across the grid.
    """
    if rng is None:
        rng = np.random.default_rng()
    grid = np.zeros((size, size), dtype=int)
    num_cells = size * size
    num_agents = int(density * num_cells)
    num_a = int(frac_group_a * num_agents)
    num_b = num_agents - num_a

    # flat array with labels, then shuffle
    # We build a 1D array with the exact counts of A, B, and EMPTY,
    # then reshape into a 2D grid after shuffling to randomize placement.
    vals = np.array(
        [GROUP_A] * num_a
        + [GROUP_B] * num_b
        + [EMPTY] * (num_cells - num_agents),
        dtype=int,
    )
    rng.shuffle(vals)
    grid[:] = vals.reshape(size, size)
    return grid


def get_neighbors(grid, i, j, radius=1):
    """
    Return the Moore neighborhood values around `(i, j)` of a given radius.

    Parameters
    - grid: 2D array of agent states.
    - i, j: Focal cell coordinates.
    - radius: Number of cells to include in each direction (>= 1).

    Returns
    - neighbors: 1D NumPy array of neighbor cell values (focal excluded).

    Notes
    - Uses torus (wrap-around) indexing so the grid has no boundaries.
    - Moore neighborhood includes diagonals as well as orthogonal neighbors.
    """
    size = grid.shape[0]
    neighbors = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:
                continue
            # Wrap indices to keep neighbors within bounds (torus)
            ni = (i + di) % size
            nj = (j + dj) % size
            neighbors.append(grid[ni, nj])
    return np.array(neighbors)


def is_satisfied(grid, i, j, tolerance, radius=1):
    """
    Return whether the agent at `(i, j)` is satisfied given `tolerance`.

    An agent is satisfied if the fraction of similar (same-group) neighbors
    among occupied neighbors is at least the `tolerance` threshold.

    Parameters
    - grid: 2D array of agent states.
    - i, j: Focal cell coordinates.
    - tolerance: Required minimum fraction of same-group neighbors.
    - radius: Moore neighborhood radius used for similarity calculation.

    Returns
    - bool: True if satisfied, or if the cell is `EMPTY`, or if there are
      zero occupied neighbors (neutral case), else False.
    """
    agent = grid[i, j]
    if agent == EMPTY:
        return True  # convention: empty cells are trivially satisfied
    neighbors = get_neighbors(grid, i, j, radius)
    occupied = neighbors != EMPTY
    if occupied.sum() == 0:
        return True  # no occupied neighbors -> neutral (treated as satisfied)
    similar = neighbors == agent #True if neighbor is of same group
    frac_similar = similar[occupied].mean()
    return frac_similar >= tolerance


def segregation_index(grid, radius=1):
    """
    Compute the average fraction of similar neighbors over all occupied cells.

    Parameters
    - grid: 2D array of agent states.
    - radius: Neighborhood radius used for similarity calculation.

    Returns
    - float in [0, 1]: Higher values indicate stronger local segregation.
    """
    size = grid.shape[0]
    vals = []
    for i in range(size):
        for j in range(size):
            if grid[i, j] == EMPTY:
                continue
            neighbors = get_neighbors(grid, i, j, radius)
            occupied = neighbors != EMPTY
            if occupied.sum() == 0:
                continue
            similar = neighbors == grid[i, j]
            vals.append(similar[occupied].mean())
    if not vals:
        return 0.0
    return float(np.mean(vals))


def multi_scale_group_fractions(grid, block_sizes=(4, 8)):
    """
    Compute multiscale fractions of groups A and B over non-overlapping blocks.

    For each `b` in `block_sizes`, the grid is partitioned into `b x b` blocks
    (assuming `size % b == 0`). The fraction of A and B among occupied cells is
    computed per block and concatenated across scales to form a feature vector.

    Parameters
    - grid: 2D array of agent states.
    - block_sizes: Iterable of block sizes (integers) that divide the grid.

    Returns
    - features: 1D NumPy array with [frac_A_block1, frac_B_block1, ...] across
      all blocks and all scales.
    """
    size = grid.shape[0]
    features = []
    for b in block_sizes:
        assert size % b == 0, "Grid size must be divisible by block size"
        num_blocks = size // b
        for bi in range(num_blocks):
            for bj in range(num_blocks):
                block = grid[bi * b : (bi + 1) * b, bj * b : (bj + 1) * b]
                occupied = block != EMPTY
                if occupied.sum() == 0:
                    frac_a = 0.0
                    frac_b = 0.0
                else:
                    frac_a = (block == GROUP_A).sum() / occupied.sum()
                    frac_b = (block == GROUP_B).sum() / occupied.sum()
                features.append(frac_a)
                features.append(frac_b)
    return np.array(features, dtype=float)


def run_schelling(
    size=40,
    density=0.9,
    frac_group_a=0.5,
    tolerance=0.4,
    max_steps=100,
    radius=1,
    rng=None,
):
    """
    Run the Schelling model until equilibrium or `max_steps`.

    Parameters
    - size, density, frac_group_a: Grid setup and population composition.
    - tolerance: Satisfaction threshold (minimum fraction of same-group
      neighbors among occupied neighbors).
    - max_steps: Upper bound on number of iterations.
    - radius: Neighborhood radius used for satisfaction checks.
    - rng: Optional random generator for reproducibility.

    Returns
    - (grid, steps): Final grid state and number of steps performed.

    Algorithm
    - Identify unhappy agents and list empty cells.
    - Randomly shuffle both lists and move up to `min(len(unhappy), len(empty))`
      agents into empty cells one-to-one.
    - Stop early if no empty cells exist or if all agents are satisfied.
    """
    if rng is None:
        rng = np.random.default_rng()
    grid = initialize_grid(size, density, frac_group_a, rng=rng)
    size = grid.shape[0]

    for step in range(max_steps):
        unhappy_positions = []
        empty_positions = list(zip(*np.where(grid == EMPTY)))

        # if no empty cells we cannot move anyone
        if not empty_positions:
            return grid, step

        # Scan the grid for agents whose local similarity falls below tolerance
        for i in range(size):
            for j in range(size):
                if grid[i, j] == EMPTY:
                    continue
                if not is_satisfied(grid, i, j, tolerance, radius=radius):
                    unhappy_positions.append((i, j))

        if not unhappy_positions:
            # equilibrium reached
            return grid, step

        # randomise both unhappy agents and empty sites
        rng.shuffle(unhappy_positions)
        rng.shuffle(empty_positions)

        moves = min(len(unhappy_positions), len(empty_positions))
        # Pair the first `moves` unhappy agents with the first `moves` empties
        for k in range(moves):
            ai, aj = unhappy_positions[k]
            ei, ej = empty_positions[k]
            grid[ei, ej] = grid[ai, aj]
            grid[ai, aj] = EMPTY

    return grid, max_steps


def compute_summary_stats(grid, block_sizes=(4, 8), radius=1):
    """
    Collect summary statistics used for calibration.

    Components
    1) Overall segregation index (mean local similarity among occupied cells).
    2) Multiscale group fractions per block for A and B.

    Returns a 1D NumPy array: `[segregation_index] + multiscale_features`.
    """
    seg = segregation_index(grid, radius=radius)
    multi = multi_scale_group_fractions(grid, block_sizes=block_sizes)
    return np.concatenate(([seg], multi))


def generate_synthetic_data(
    p_true=0.4,
    size=40,
    density=0.9,
    frac_group_a=0.5,
    block_sizes=(4, 8),
    radius=1,
    max_steps=200,
    rng=None,
):
    """
    Generate one synthetic data set for a known true tolerance `p_true`.

    Returns
    - target: Summary statistics vector (1D array) used for calibration.
    """
    if rng is None:
        rng = np.random.default_rng()
    grid, steps = run_schelling(
        size=size,
        density=density,
        frac_group_a=frac_group_a,
        tolerance=p_true,
        max_steps=max_steps,
        radius=radius,
        rng=rng,
    )
    return compute_summary_stats(grid, block_sizes=block_sizes, radius=radius)


def calibrate_tolerance(
    candidate_ps,
    synthetic_target,
    num_runs_per_p=5,
    size=40,
    density=0.9,
    frac_group_a=0.5,
    block_sizes=(4, 8),
    radius=1,
    max_steps=200,
    rng=None,
):
    """
    Simple grid search for the Schelling tolerance parameter.

    For each candidate `p`:
    - run the model `num_runs_per_p` times
    - compute mean summary statistics across runs
    - compute squared error relative to `synthetic_target`

    Returns
    - results: List of dicts sorted by ascending error, each with keys
      `{"p", "error", "mean_stats"}`.

    Notes
    - This is a basic calibration approach (least-squares on summary stats).
    - For robustness, increase `num_runs_per_p` to reduce Monte Carlo noise,
      or consider different distance metrics/weights per statistic.
    """
    if rng is None:
        rng = np.random.default_rng()
    results = []
    for p in candidate_ps:
        stats_list = []
        for _ in range(num_runs_per_p):
            grid, steps = run_schelling(
                size=size,
                density=density,
                frac_group_a=frac_group_a,
                tolerance=p,
                max_steps=max_steps,
                radius=radius,
                rng=rng,
            )
            stats_list.append(
                compute_summary_stats(
                    grid,
                    block_sizes=block_sizes,
                    radius=radius,
                )
            )
        stats_arr = np.vstack(stats_list)
        mean_stats = stats_arr.mean(axis=0)
        error = np.mean((mean_stats - synthetic_target) ** 2)
        results.append(
            {
                "p": p,
                "error": float(error),
                "mean_stats": mean_stats,
            }
        )
    results.sort(key=lambda d: d["error"])
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(123)

    # Step 1: generate synthetic data with a known true tolerance
    p_true = 0.4
    # The synthetic target plays the role of "observed" data; in a real
    # application you would compute summary statistics on actual data.
    target = generate_synthetic_data(
        p_true=p_true,
        size=40,
        density=0.9,
        frac_group_a=0.5,
        block_sizes=(4, 8),
        radius=1,
        max_steps=200,
        rng=rng,
    )
    print("Synthetic target stats for true p =", p_true)
    print("Length of feature vector:", len(target))

    # Step 2: calibrate over a grid of candidate tolerances
    # We explore a coarse grid in [0.2, 0.6]; refine around the best value
    # if higher precision is needed.
    candidate_ps = np.linspace(0.2, 0.6, 9)  # 0.2, 0.25, ..., 0.6
    results = calibrate_tolerance(
        candidate_ps,
        synthetic_target=target,
        num_runs_per_p=5,
        size=40,
        density=0.9,
        frac_group_a=0.5,
        block_sizes=(4, 8),
        radius=1,
        max_steps=200,
        rng=rng,
    )
    best = results[0]
    print("\nBest calibrated p:", best["p"])
    print("Squared error:", best["error"])

    # Plot measures across p and the error curve, including target lines
    try:
        import matplotlib.pyplot as plt

        p_values = np.array([r["p"] for r in results])
        errors = np.array([r["error"] for r in results])
        mean_stats_matrix = np.vstack([r["mean_stats"] for r in results])
        target_vec = target

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        ax0, ax1 = axes

        # Left: segregation vs tolerance p (bar chart for legibility)
        seg = mean_stats_matrix[:, 0]
        bar_width0 = (p_values.max() - p_values.min()) / (len(p_values) * 1.5)
        ax0.bar(p_values, seg, width=bar_width0, color="tab:blue", alpha=0.8, edgecolor="black", label="Segregation")
        ax0.axhline(target_vec[0], color="tab:blue", linestyle="--", alpha=0.7, label="Target Segregation")
        ax0.set_xticks(p_values)
        ax0.set_xticklabels([f"{p:.2f}" for p in p_values])
        ax0.set_xlabel("Tolerance p")
        ax0.set_ylabel("Segregation index")
        ax0.set_title("Segregation vs tolerance (bar)")
        ax0.set_ylim(0, 1)
        ax0.grid(True, axis="y", alpha=0.3)
        ax0.legend(loc="best")

        # Right: error vs tolerance p (bar plot for clearer comparison)
        bar_width = (p_values.max() - p_values.min()) / (len(p_values) * 1.5)
        ax1.bar(p_values, errors, width=bar_width, color="tab:red", alpha=0.8, edgecolor="black")
        ax1.set_xticks(p_values)
        ax1.set_xticklabels([f"{p:.2f}" for p in p_values])
        ax1.set_xlabel("Tolerance p")
        ax1.set_ylabel("Squared error")
        ax1.set_title("Calibration error vs tolerance")
        ax1.set_ylim(bottom=0)
        ax1.grid(True, axis="y", alpha=0.3)

        fig.suptitle("Schelling Calibration: Measures and Error", fontsize=12)
        fig.savefig("schelling_calibration.png", dpi=150)
        plt.show()
    except Exception as e:
        print("Plotting skipped (Matplotlib not available or error occurred):", e)