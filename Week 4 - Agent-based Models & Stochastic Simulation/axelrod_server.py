import colorsys
from typing import Dict

from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
try:
    # Newer Mesa exposes Slider; older exposes UserSettableParameter
    from mesa.visualization.UserParam import Slider
except Exception:
    from mesa.visualization.UserParam import UserSettableParameter as Slider

try:
    from .axelrod_model import AxelrodModel, CultureAgent  # when part of a package
except Exception:
    from axelrod_model import AxelrodModel, CultureAgent  # fallback for direct script execution


def _color_from_agent(agent: CultureAgent) -> str:
    # Encode the culture vector as a base-q integer to produce a stable hue.
    code = 0
    for v in agent.features:
        code = code * agent.q + v
    bins = 64  # number of distinct hue bins
    hue = (code % bins) / bins
    sat = 0.95
    val = 0.95
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def agent_portrayal(agent: CultureAgent) -> Dict:
    color = _color_from_agent(agent)
    return {
        "Shape": "rect",
        "w": 1,
        "h": 1,
        "Filled": "true",
        "Layer": 0,
        "Color": color,
    }


def make_server():
    grid = CanvasGrid(agent_portrayal, 20, 20, 600, 600)
    chart = ChartModule(
        [
            {"Label": "NumCultures", "Color": "Black"},
            {"Label": "InteractionsPossible", "Color": "Red"},
            {"Label": "InteractionsReal", "Color": "Green"},
        ],
        data_collector_name="datacollector",
    )

    model_params = {
        "width": Slider("Grid width", 20, 5, 50, 1),
        "height": Slider("Grid height", 20, 5, 50, 1),
        "occupancy": Slider("Occupancy fraction", 1.0, 0.05, 1.0, 0.05),
        "F": Slider("Features F", 5, 1, 10, 1),
        "q": Slider("Traits per feature q", 10, 1, 30, 1),
    }

    server = ModularServer(AxelrodModel, [grid, chart], "Axelrod Cultural Dissemination", model_params)
    server.port = 8800
    return server


if __name__ == "__main__":
    make_server().launch()