import random
from typing import List, Tuple

from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector


class CultureAgent(Agent):
    def __init__(self, unique_id: int, model: Model, features: List[int], q: int):
        super().__init__(unique_id, model)
        self.features = features  # length F, traits in {0, …, q-1}
        self.q = q

    def similarity(self, other: "CultureAgent") -> float:
        matches = sum(1 for a, b in zip(self.features, other.features) if a == b)
        return matches / len(self.features)

    def step(self) -> bool:
        neighbours = self.model.grid.get_neighbors(self.pos, moore=False, include_center=False)
        if not neighbours:
            return False
        # Compute similarities and filter to active neighbors (0 < overlap < 1)
        sims = [(n, self.similarity(n)) for n in neighbours]
        active = [(n, s) for (n, s) in sims if 0.0 < s < 1.0]
        if not active:
            return False
        # Select a neighbor with probability proportional to overlap
        neighs = [n for (n, s) in active]
        weights = [s for (n, s) in active]
        neighbour = random.choices(neighs, weights=weights, k=1)[0]
        # Perform one interaction deterministically: copy one differing feature
        diffs = [i for i, (a, b) in enumerate(zip(self.features, neighbour.features)) if a != b]
        if diffs:
            i = random.choice(diffs)
            self.features[i] = neighbour.features[i]
            return True
        return False


class AxelrodModel(Model):
    def __init__(self, width: int = 20, height: int = 20, occupancy: float = 1.0, F: int = 5, q: int = 10):
        super().__init__()
        self.grid = SingleGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.width = width
        self.height = height
        self.F = F
        self.q = q
        self.running = True
        self.interactions_possible_last = 0
        self.interactions_real_last = 0

        # Determine population from occupancy fraction (one agent per cell)
        total_cells = width * height
        N = max(0, min(total_cells, int(round(occupancy * total_cells))))

        # Create agents with random feature vectors placed on unique cells
        all_positions = [(x, y) for x in range(width) for y in range(height)]
        random.shuffle(all_positions)
        for i in range(N):
            x, y = all_positions[i]
            feats = [random.randrange(q) for _ in range(F)]
            a = CultureAgent(i, self, feats, q)
            self.grid.place_agent(a, (x, y))
            self.schedule.add(a)

        self.datacollector = DataCollector(
            model_reporters={
                "NumCultures": lambda m: m.count_culture_domains(),
                "InteractionsPossible": lambda m: m.interactions_possible_last,
                "InteractionsReal": lambda m: m.interactions_real_last,
            },
            agent_reporters={}
        )

    def count_culture_domains(self) -> int:
        signatures: List[Tuple[int, ...]] = [tuple(agent.features) for agent in self.schedule.agents]
        return len(set(signatures))

    def compute_interactions_possible(self) -> int:
        # Count unique neighbor pairs where 0 < overlap < 1 (active edges)
        count = 0
        seen_pairs = set()
        for agent in self.schedule.agents:
            for nbr in self.grid.get_neighbors(agent.pos, moore=False, include_center=False):
                pair = tuple(sorted((agent.unique_id, nbr.unique_id)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                sim = agent.similarity(nbr)
                if 0.0 < sim < 1.0:
                    count += 1
        return count

    def step(self):
        # Compute interactions possible before updating
        self.interactions_possible_last = self.compute_interactions_possible()
        self.interactions_real_last = 0
        if self.interactions_possible_last == 0:
            # No more interactions possible: terminate
            self.running = False
            # Collect final state for plotting
            self.datacollector.collect(self)
            return
        # Each agent attempts exactly one interaction per timestep
        agents = list(self.schedule.agents)
        random.shuffle(agents)
        for agent in agents:
            did = agent.step()
            if did:
                self.interactions_real_last += 1
        # Collect after updates
        self.datacollector.collect(self)