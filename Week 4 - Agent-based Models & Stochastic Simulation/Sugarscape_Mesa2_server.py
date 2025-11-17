from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from Sugarscape_Mesa2 import SugarModel


def agent_portrayal(agent):
    if not getattr(agent, "alive", True):
        return {"Shape": "circle", "r": 0.5, "Color": "gray", "Filled": "true"}
    s = agent.sugar
    if s < 5:
        color = "red"
    elif s < 15:
        color = "orange"
    else:
        color = "green"
    return {"Shape": "circle", "r": 0.6, "Color": color, "Filled": "true"}


def make_server():
    grid_vis = CanvasGrid(agent_portrayal, 50, 50, 600, 600)
    chart = ChartModule([{"Label": "Gini", "Color": "Black"}], data_collector_name="datacollector")

    model_params = {
        "N": UserSettableParameter("slider", "N (agents)", 200, 10, 250, 10),
        "width": UserSettableParameter("slider", "Grid width", 50, 10, 80, 5),
        "height": UserSettableParameter("slider", "Grid height", 50, 10, 80, 5),
    }

    server = ModularServer(
        SugarModel,
        [grid_vis, chart],
        "Sugarscape (Basic)",
        model_params,
    )
    server.port = 8521
    return server


if __name__ == "__main__":
    make_server().launch()