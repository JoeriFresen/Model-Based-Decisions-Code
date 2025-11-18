from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np


"""
Top-level simulation parameters

Edit these defaults to change the baseline behavior of the model. Most of these
can also be overridden when constructing `SugarModel` from the server UI.
"""

# ---------------------------
# Simulation Defaults (editable)
# ---------------------------

# Initial number of agents placed on the grid at reset
N_DEFAULT = 200

# Horizontal size of the toroidal landscape (number of columns)
GRID_WIDTH_DEFAULT = 50

# Vertical size of the toroidal landscape (number of rows)
GRID_HEIGHT_DEFAULT = 50

# Standard deviation of the Gaussian hills shaping sugar capacity
# Larger values spread hills out; smaller values make them more concentrated
SIGMA_DEFAULT = 5.0

# Maximum sugar a cell can hold (upper capacity cap)
MAX_SUGAR_DEFAULT = 5.0

# Amount of sugar regrown per cell per step (bounded by capacity)
GROW_RATE_DEFAULT = 0.5

# Fraction of inherited sugar taxed away when new agents are created
TAX_RATE_DEFAULT = 0.5

# Maximum age an agent can reach before dying (in steps)
LIFESPAN_DEFAULT = 80

# Lifespan range for sampling max_age per agent (uniform integer)
LIFESPAN_MIN_DEFAULT = 60
LIFESPAN_MAX_DEFAULT = 100

# ---------------------------
# Agent Defaults
# ---------------------------

# Starting sugar for newly created agents (initial and replacements)
INITIAL_AGENT_SUGAR = 3.0

# Sugar consumed by an agent each step (metabolic upkeep cost)
AGENT_METABOLISM = 1

# Minimum vision distance an agent may have (cells scanned along axes)
VISION_MIN = 1

# Maximum vision distance an agent may have (cells scanned along axes)
VISION_MAX = 2

# ---------------------------
# Welfare Defaults
# ---------------------------

# Percentile threshold for welfare eligibility (bottom X% of alive agents)
WELFARE_PERCENT_DEFAULT = 10.0

# Fixed welfare amount added to inheritance for eligible new agents
WELFARE_AMOUNT_DEFAULT = 2.0


def gini(x):
    if len(x) == 0:
        return 0.0
    x = np.sort(np.array(x, dtype=float))
    n = len(x)
    return (2 * np.sum((np.arange(1, n + 1) * x))) / (n * np.sum(x)) - (n + 1) / n


class SugarAgent(Agent):
    def __init__(self, unique_id, model, sugar, metabolism, vision, max_age):
        super().__init__(unique_id, model)
        self.sugar = sugar
        self.metabolism = metabolism
        self.vision = vision
        self.age = 0
        self.max_age = max_age
        self.alive = True

    def step(self):
        if not self.alive:
            return

        x, y = self.pos  # current position
        # Start by assuming we stay put; only move if we see strictly more sugar
        best = (x, y)
        best_s = int(self.model.psugar[x, y])

        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 4 possible directions to move
        for dx, dy in dirs:
            for d in range(1, self.vision + 1):  # scan in all 4 directions within vision range
                nx = (x + dx * d) % self.model.width
                ny = (y + dy * d) % self.model.height
                if (
                    self.model.is_cell_free_of_agents((nx, ny))
                    and int(self.model.psugar[nx, ny]) > best_s  # found a higher-sugar patch
                ):
                    best_s = self.model.psugar[nx, ny]  # update best sugar value found
                    best = (nx, ny)  # update best position to move to

        if best == (x, y):
            # if no higher-sugar patch found, stay in current position
            print(f"Agent {self.unique_id} stays at {best}")
        else:
            print(f"Agent {self.unique_id} moves to {best}")
            #print the distance moved by agent
            d=abs(x-best[0])+abs(y-best[1])
            print(f"Agent {self.unique_id} moves {d} cells")

        self.model.grid.move_agent(self, best)  # move to the best position found
        self.sugar += float(self.model.psugar[best])  # consume all the sugar on the best patch
        self.model.psugar[best] = 0.0  # set the sugar on the best patch to 0

        # Consume fixed 1 sugar per timestep (metabolism set to 1)
        self.sugar -= 1.0
        self.age += 1

        if self.sugar <= 0 or self.age >= self.max_age:
            self.alive = False
            # Record remaining sugar for inheritance (may be 0 if died of starvation)
            self.model.death_bank.append(max(0.0, float(self.sugar)))  # store my remaining sugar for inheritance
            # Increment per-step death counter
            self.model.deaths_this_step += 1
            self.model.grid.remove_agent(self)


class SugarPatch(Agent):
    def __init__(self, pos, model, amount):
        super().__init__(f"patch-{pos[0]}-{pos[1]}", model)
        self.amount = float(amount)
        self.pos = pos


class SugarModel(Model):
    def __init__(
        self,
        N=N_DEFAULT,
        width=GRID_WIDTH_DEFAULT,
        height=GRID_HEIGHT_DEFAULT,
        sigma=SIGMA_DEFAULT,
        max_sugar=MAX_SUGAR_DEFAULT,
        grow_rate=GROW_RATE_DEFAULT,
        tax_rate=TAX_RATE_DEFAULT,
        lifespan=LIFESPAN_DEFAULT,
        lifespan_min=LIFESPAN_MIN_DEFAULT,
        lifespan_max=LIFESPAN_MAX_DEFAULT,
        welfare_percent=WELFARE_PERCENT_DEFAULT,
        welfare_amount=WELFARE_AMOUNT_DEFAULT,
        initial_agent_sugar=INITIAL_AGENT_SUGAR,
        agent_metabolism=AGENT_METABOLISM,
        vision_min=VISION_MIN,
        vision_max=VISION_MAX,
    ):
        super().__init__()
        self.N = N
        self.width = width
        self.height = height
        self.sigma = float(sigma)
        self.max_sugar = float(max_sugar)
        self.grow_rate = float(grow_rate)
        self.tax_rate = float(tax_rate)
        self.lifespan = int(lifespan)
        self.lifespan_min = int(lifespan_min)
        self.lifespan_max = int(lifespan_max)
        self.initial_agent_sugar = float(initial_agent_sugar)
        self.agent_metabolism = int(agent_metabolism)
        self.vision_min = int(vision_min)
        self.vision_max = int(vision_max)
        # Welfare params
        self.welfare_percent = float(max(0.0, min(100.0, welfare_percent)))
        self.welfare_amount = float(max(0.0, welfare_amount))
        self.death_bank = []  # store sugar amounts from dead agents for inheritance
        self.deaths_this_step = 0  # reset each step; used for death rate
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)

        self.psugar = np.zeros((width, height), dtype=int)
        self.capacity = np.zeros((width, height), dtype=int)
        self.init_landscape()
        # Create patch agents for each cell for visualization
        for x in range(width):
            for y in range(height):
                patch = SugarPatch((x, y), self, self.psugar[x, y])
                self.grid.place_agent(patch, (x, y))

        # Enforce single-agent occupancy per cell even with MultiGrid (patches coexist)
        if N > width * height:
            raise ValueError(
                "N exceeds grid capacity for one-agent-per-cell; reduce N"
            )

        for _ in range(N):
            a = SugarAgent(
                self.next_id(),
                self,
                sugar=self.initial_agent_sugar,
                metabolism=self.agent_metabolism,
                vision=self.random.randint(self.vision_min, self.vision_max),
                max_age=self.random.randint(self.lifespan_min, self.lifespan_max),
            )
            pos = (self.random.randrange(width), self.random.randrange(height))
            while not self.is_cell_free_of_agents(pos):
                pos = (self.random.randrange(width), self.random.randrange(height))
            self.grid.place_agent(a, pos)
            self.schedule.add(a)

        # Helper reporters
        def alive_sugars(m):
            return [a.sugar for a in m.schedule.agents if isinstance(a, SugarAgent) and a.alive]

        def lorenz_point(m, frac):
            xs = sorted(alive_sugars(m))
            n = len(xs)
            if n == 0:
                return 0.0
            total = sum(xs)
            if total <= 0:
                return 0.0
            k = int(frac * n)
            k = max(0, min(n, k))
            return sum(xs[:k]) / total

        def vision_count(m, v):
            return sum(1 for a in m.schedule.agents if isinstance(a, SugarAgent) and a.alive and a.vision == v)

        self.datacollector = DataCollector(
            model_reporters={
                "Gini": lambda m: gini(alive_sugars(m)),
                # Approximate Lorenz curve with deciles (cumulative share up to decile)
                "L_0": lambda m: lorenz_point(m, 0.0),
                "L_10": lambda m: lorenz_point(m, 0.1),
                "L_20": lambda m: lorenz_point(m, 0.2),
                "L_30": lambda m: lorenz_point(m, 0.3),
                "L_40": lambda m: lorenz_point(m, 0.4),
                "L_50": lambda m: lorenz_point(m, 0.5),
                "L_60": lambda m: lorenz_point(m, 0.6),
                "L_70": lambda m: lorenz_point(m, 0.7),
                "L_80": lambda m: lorenz_point(m, 0.8),
                "L_90": lambda m: lorenz_point(m, 0.9),
                "L_100": lambda m: lorenz_point(m, 1.0),
                # Vision distribution counts
                "Vision_1": lambda m: vision_count(m, 1),
                "Vision_2": lambda m: vision_count(m, 2),
                "Vision_3": lambda m: vision_count(m, 3),
                "Vision_4": lambda m: vision_count(m, 4),
                "Vision_5": lambda m: vision_count(m, 5),
                "Vision_6": lambda m: vision_count(m, 6),
                # Deaths in the last step
                "Deaths": lambda m: m.deaths_this_step,
            },
            agent_reporters={"Sugar": "sugar"},
        )

        # Collect an initial datapoint so charts have data at reset
        self.datacollector.collect(self)

    def init_landscape(self):
        # Gaussian mountains at specified relative positions
        # Top-right: 3/4 width, 1/4 height from the top -> y = int(0.25 * height)
        peak1 = (int(0.75 * self.width), int(0.25 * self.height))
        # Bottom-left: 1/4 width, 1/4 height from the bottom -> y = int(0.75 * height)
        peak2 = (int(0.25 * self.width), int(0.75 * self.height))
        two_sigma2 = 2.0 * (self.sigma ** 2)
        for x in range(self.width):
            for y in range(self.height):
                dx1 = x - peak1[0]
                dy1 = y - peak1[1]
                dx2 = x - peak2[0]
                dy2 = y - peak2[1]
                g1 = np.exp(-(dx1 * dx1 + dy1 * dy1) / two_sigma2)
                g2 = np.exp(-(dx2 * dx2 + dy2 * dy2) / two_sigma2)
                amt = self.max_sugar * (g1 + g2)
                cap = int(np.round(min(self.max_sugar, amt)))
                self.capacity[x, y] = cap
                self.psugar[x, y] = cap

    def is_cell_free_of_agents(self, pos):
        # Ignore SugarPatch occupancy; allow only one SugarAgent per cell
        for obj in self.grid.get_cell_list_contents([pos]):
            if isinstance(obj, SugarAgent):
                return False
        return True

    def growback(self):
        # Grow only in cells with initial capacity; cap by that capacity
        grown = np.minimum(self.psugar.astype(float) + self.grow_rate, self.capacity.astype(float))
        self.psugar = np.round(grown).astype(int)

    def step(self):
        # Reset per-step death counter
        self.deaths_this_step = 0
        self.schedule.step()
        self.growback()
        # Sync patch agents with psugar values for visualization
        for x in range(self.width):
            for y in range(self.height):
                # Find the patch agent at (x,y)
                for obj in self.grid.get_cell_list_contents([(x, y)]):
                    if isinstance(obj, SugarPatch):
                        obj.amount = float(self.psugar[x, y])
                        break
        # Replace dead agents with inheritance minus tax
        while self.death_bank:
            inheritance = self.death_bank.pop(0)
            new_sugar = max(0.0, inheritance * (1.0 - self.tax_rate))
            # Apply welfare if new_sugar is in the bottom X% of alive agents' wealth
            alive_wealth = [a.sugar for a in self.schedule.agents if isinstance(a, SugarAgent) and a.alive]
            qualifies = False
            if len(alive_wealth) == 0:
                qualifies = True
            else:
                threshold = float(np.percentile(np.array(alive_wealth, dtype=float), self.welfare_percent))
                qualifies = (new_sugar <= threshold)
            if qualifies and self.welfare_amount > 0.0:
                new_sugar += self.welfare_amount
            new_id = self.next_id()
            a = SugarAgent(
                new_id,
                self,
                sugar=new_sugar,
                metabolism=self.agent_metabolism,
                vision=self.random.randint(self.vision_min, self.vision_max),
                max_age=self.random.randint(self.lifespan_min, self.lifespan_max),
            )
            pos = (self.random.randrange(self.width), self.random.randrange(self.height))
            tries = 0
            while not self.is_cell_free_of_agents(pos) and tries < 1000:
                pos = (self.random.randrange(self.width), self.random.randrange(self.height))
                tries += 1
            if not self.is_cell_free_of_agents(pos):
                # If grid is saturated, skip placement
                continue
            self.grid.place_agent(a, pos)
            self.schedule.add(a)
        self.datacollector.collect(self)