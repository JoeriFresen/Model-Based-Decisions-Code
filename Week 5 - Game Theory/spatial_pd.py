from mesa import Agent, Model
from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider
import numpy as np

# 1 blue, 2 red, 3 green, 4 yellow
COLOR_MAP = {
    1: "blue", #cooperate, cooperate
    2: "red",  #defect, defect
    3: "green", #cooperate, defect
    4: "yellow" #defect, cooperate
}

class PatchAgent(Agent): # PatchAgent is an agent that lives on a grid
    def __init__(self, pos, model, cooperate):
        super().__init__(pos, model)
        # Let the Grid manage position; initialize as None until placed
        self.pos = None
        self.cooperate = cooperate #True or False, cooperate is true
        self.old_cooperate = cooperate #True means cooperated last round
        self.score = 0 # score is the payoff of the agent
        self.color_class = 1 if cooperate else 2 #initially set colours

    def interact(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, #my position
            moore=True, #look at 8 neighboring cells
            include_center=False #don't include myself
        )
        total_cooperators = sum([n.cooperate for n in neighbors]) #count how many cooperators

        #calculate payoff - if i cooperate I get the payoff of the number of cooperators
        #if i defect I get the payoff of the number of cooperators times the defection award, 
        # otherwise 0
        if self.cooperate:
            self.score = total_cooperators #if i cooperate I get the payoff of the number of cooperators
        else:
            self.score = self.model.defection_award * total_cooperators #if i defect I get the payoff of the number of cooperators times the defection award

    def select_strategy(self): #select strategy based on payoff of neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False
        )
        # Break ties randomly to avoid directional bias that can cause banding
        if not neighbors:
            return
        max_score = max(a.score for a in neighbors) #find the maximum score of the neighbors
        top_neighbors = [a for a in neighbors if a.score == max_score] #find all neighbors with the maximum score
        best_neighbor = self.random.choice(top_neighbors) #break ties randomly, pick one of the top neighbours at random
        new_strategy = best_neighbor.cooperate #adopt the strategy of the best neighbor

        self.old_cooperate = self.cooperate #update old cooperate to current cooperate
        self.cooperate = new_strategy #update current cooperate to new strategy

        # establish-color logic
        if self.old_cooperate and self.cooperate: # cooperate twice
            self.color_class = 1
        elif not self.old_cooperate and not self.cooperate: #  defect twice
            self.color_class = 2
        elif self.old_cooperate and not self.cooperate: # now defected, cooperated before
            self.color_class = 3
        else: # now cooperated, defected before
            self.color_class = 4

    def step(self): #every step executed interact
        self.interact()

    def advance(self):#advance is called at the end of each timestep, after all agents are stepped
        self.select_strategy()


class SpatialPD(Model):
    def __init__(self, N=50, initial_cooperation=50, defection_award=2):
        self.height = N
        self.width = N
        self.schedule = SimultaneousActivation(self) #every agent is activated at the same time
        self.grid = SingleGrid(self.width, self.height, torus=True) #SingleGrid is a grid where each cell can contain at most one agent
        self.defection_award = defection_award #the award for defecting (multiply by number cooperators)

        for x in range(self.width):
            for y in range(self.height):
                coop = (self.random.random() < initial_cooperation / 100)
                agent = PatchAgent((x, y), self, coop)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)

        self.datacollector = DataCollector({ #used to collect data at the end of each timestep
            "CooperatorFraction": lambda m: self._cooperator_fraction(m),
            "(C,C)": lambda m: self._count_color(m, 1),
            "(D,D)": lambda m: self._count_color(m, 2),
            "(C,D)": lambda m: self._count_color(m, 3),
            "(D,C)": lambda m: self._count_color(m, 4)
        })

    @staticmethod
    def _cooperator_fraction(model):
        agents = model.schedule.agents
        return sum(a.cooperate for a in agents) / len(agents)

    @staticmethod
    def _count_color(model, col):
        return sum(1 for a in model.schedule.agents if a.color_class == col)

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


def agent_portrayal(agent):
    portrayal = {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Color": COLOR_MAP[agent.color_class],
        "Layer": 0
    }
    return portrayal

class ModelInfo(TextElement):
    """Renders explanatory text and the payoff matrix on the UI."""
    def render(self, model):
        coop_frac = SpatialPD._cooperator_fraction(model)
        da = model.defection_award
        html = f"""
        <h3>Spatial Prisoner's Dilemma</h3>
        <p>
          Agents occupy an <strong>N×N</strong> toroidal grid. Each agent is either
          <strong>Cooperate (C)</strong> or <strong>Defect (D)</strong>. Each tick:
        </p>
        <ol>
          <li>Agents interact with all 8 Moore neighbors and accumulate payoffs.</li>
          <li>Each agent adopts the strategy of the neighbor with the highest payoff
              (ties are broken randomly).</li>
        </ol>
        <p><strong>Colors:</strong> Blue=C→C, Red=D→D, Green=C→D, Yellow=D→C.</p>
        <p><strong>Current:</strong> Defection Award = {da:.2f}; Cooperator Fraction = {coop_frac:.2f}</p>
        <h4>Per-neighbor Payoff Matrix</h4>
        <table border="1" cellpadding="6" cellspacing="0">
          <tr><th>You</th><th>Neighbor</th><th>Your payoff increment</th></tr>
          <tr><td>C</td><td>C</td><td>R = 1</td></tr>
          <tr><td>C</td><td>D</td><td>S = 0</td></tr>
          <tr><td>D</td><td>C</td><td>T = defection_award = {da:.2f}</td></tr>
          <tr><td>D</td><td>D</td><td>P = 0</td></tr>
        </table>
        <p>Total payoff = sum over all neighbors. Parameters: R=1, S=0, P=0, T=defection_award.</p>
        """
        return html

def run_server(): #runs the model in a local server, visualised in the browser
    N = 200 #number of agents in the grid

    grid_viz = CanvasGrid(agent_portrayal, N, N, 800, 800)

    chart = ChartModule([
        {"Label": "(C,C)", "Color": "blue"},
        {"Label": "(D,D)", "Color": "red"},
        {"Label": "(C,D)", "Color": "green"},
        {"Label": "(D,C)", "Color": "yellow"},
    ], data_collector_name="datacollector")

    model_params = {
        "N": N,
        "initial_cooperation": Slider(
            "Initial Cooperation (%)",
            50,
            0,
            100,
            1,
            description="Percentage of cooperators at initialization"
        ),
        "defection_award": Slider(
            "Defection Award",
            1.3,
            1.0,
            2.0,
            0.05,
            description="Reward multiplier for defectors"
        )
    }

    info = ModelInfo()
    server = ModularServer(
        SpatialPD,
        [info, grid_viz, chart],
        "Model-Based Decisions Spatial PD (Adapted from NetLogo)",
        model_params
    )
    server.port = 8521
    server.launch()


if __name__ == "__main__":
    run_server()