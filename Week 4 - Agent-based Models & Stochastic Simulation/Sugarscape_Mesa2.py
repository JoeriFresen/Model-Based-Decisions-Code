from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np


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

        x, y = self.pos
        best = None
        best_s = -1

        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in dirs:
            for d in range(1, self.vision + 1):
                nx = (x + dx * d) % self.model.width
                ny = (y + dy * d) % self.model.height
                if (
                    self.model.grid.is_cell_empty((nx, ny))
                    and self.model.psugar[nx, ny] > best_s
                ):
                    best_s = self.model.psugar[nx, ny]
                    best = (nx, ny)

        if best is None:
            best = (x, y)
        self.model.grid.move_agent(self, best)
        self.sugar += float(self.model.psugar[best])
        self.model.psugar[best] = 0.0

        self.sugar -= self.metabolism
        self.age += 1

        if self.sugar <= 0 or self.age > self.max_age:
            self.alive = False
            self.model.grid.remove_agent(self)


class SugarModel(Model):
    def __init__(self, N=200, width=50, height=50):
        super().__init__()
        self.N = N
        self.width = width
        self.height = height
        self.grid = SingleGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)

        self.psugar = np.zeros((width, height), dtype=float)
        self.init_landscape()

        if N > width * height:
            raise ValueError(
                "N exceeds grid capacity for SingleGrid; use MultiGrid or reduce N"
            )

        for i in range(N):
            a = SugarAgent(
                i,
                self,
                sugar=5.0,
                metabolism=self.random.randint(1, 4),
                vision=self.random.randint(1, 6),
                max_age=self.random.randint(60, 100),
            )
            pos = (self.random.randrange(width), self.random.randrange(height))
            while not self.grid.is_cell_empty(pos):
                pos = (self.random.randrange(width), self.random.randrange(height))
            self.grid.place_agent(a, pos)
            self.schedule.add(a)

        self.datacollector = DataCollector(
            model_reporters={
                "Gini": lambda m: gini(
                    [a.sugar for a in m.schedule.agents if a.alive]
                )
            },
            agent_reporters={"Sugar": "sugar"},
        )

    def init_landscape(self):
        peaks = [(15, 15), (35, 35)]
        for cx, cy in peaks:
            for x in range(self.width):
                for y in range(self.height):
                    dist = abs(x - cx) + abs(y - cy)
                    height = max(0, 5 - dist)
                    self.psugar[x, y] += float(height)

    def growback(self):
        self.psugar = np.minimum(self.psugar + 1.0, 5.0)

    def step(self):
        self.schedule.step()
        self.growback()
        self.datacollector.collect(self)