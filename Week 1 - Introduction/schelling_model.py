import mesa
from mesa import Model, Agent
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider


class SchellingAgent(Agent):
    """Agent with a preference for similar neighbors."""
    def __init__(self, unique_id, model, group):
        super().__init__(unique_id, model)
        self.group = group
        self.happy = True

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        if len(neighbors) > 0:
            similar = sum(1 for n in neighbors if n.group == self.group)
            self.happy = (similar / len(neighbors)) >= self.model.similarity_threshold
        else:
            self.happy = True

        if not self.happy:
            empty = [c for c in self.model.grid.empties]
            if empty:
                self.model.grid.move_agent(self, self.random.choice(empty))


class SchellingModel(Model):
    """Schelling segregation model."""
    def __init__(self, width=20, height=20, density=0.8, similarity_threshold=0.3):
        self.num_agents = int(width * height * density)
        self.grid = SingleGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.similarity_threshold = similarity_threshold
        self.running = True

        for i in range(self.num_agents):
            group = self.random.choice(["A", "B"])
            agent = SchellingAgent(i, self, group)
            while True:
                x, y = self.random.randrange(width), self.random.randrange(height)
                if self.grid.is_cell_empty((x, y)):
                    self.grid.place_agent(agent, (x, y))
                    break
            self.schedule.add(agent)

        self.datacollector = DataCollector({
            "happy": lambda m: sum(a.happy for a in m.schedule.agents) / m.schedule.get_agent_count()
        })

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if all(a.happy for a in self.schedule.agents):
            self.running = False


def agent_portrayal(agent):
    if agent is None:
        return
    return {
        "Shape": "circle",
        "Color": "blue" if agent.group == "A" else "red",
        "Filled": "true",
        "r": 0.8,
        "Layer": 0,
    }


grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)
chart = ChartModule([{"Label": "happy", "Color": "green"}], data_collector_name="datacollector")

model_params = {
    "width": 20,
    "height": 20,
    "density": Slider(
        "Occupation Rate",
        0.8,
        0.1,
        1.0,
        0.05,
        description="Fraction of cells initially occupied"
    ),
    "similarity_threshold": Slider(
        "Similarity Threshold",
        0.3,
        0.0,
        1.0,
        0.05,
        description="Minimum similar-neighbor fraction to be happy"
    ),
}

server = ModularServer(SchellingModel, [grid, chart], "Schelling Segregation Model", model_params)
server.port = 8521

if __name__ == "__main__":
    server.launch()