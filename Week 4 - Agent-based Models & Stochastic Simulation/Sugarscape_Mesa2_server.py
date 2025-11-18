from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, TextElement
try:
    from mesa.visualization.UserParam import Slider
except Exception:
    # Older Mesa versions expose UserSettableParameter
    from mesa.visualization.UserParam import UserSettableParameter as Slider

from Sugarscape_Mesa2 import (
    SugarModel,
    SugarPatch,
    SugarAgent,
    N_DEFAULT,
    GRID_WIDTH_DEFAULT,
    GRID_HEIGHT_DEFAULT,
    SIGMA_DEFAULT,
    GROW_RATE_DEFAULT,
    TAX_RATE_DEFAULT,
    LIFESPAN_MIN_DEFAULT,
    LIFESPAN_MAX_DEFAULT,
    WELFARE_PERCENT_DEFAULT,
    WELFARE_AMOUNT_DEFAULT,
)
import numpy as np


def agent_portrayal(agent):
    # Draw sugar patches as background heatmap
    if isinstance(agent, SugarPatch):
        max_sugar = getattr(agent.model, "max_sugar", 5.0)
        norm = 0.0
        if max_sugar > 0:
            norm = max(0.0, min(1.0, agent.amount / max_sugar))
        shade = int(255 * norm)
        # Yellowish gradient: high sugar -> bright yellow
        color = f"#{shade:02x}{shade:02x}00"
        return {"Shape": "rect", "w": 1, "h": 1, "Color": color, "Filled": "true", "Layer": 0}

    # Draw agents on top
    if not getattr(agent, "alive", True):
        return {"Shape": "circle", "r": 0.5, "Color": "gray", "Filled": "true", "Layer": 1}
    s = getattr(agent, "sugar", 0)
    if s < 5:
        color = "red"
    elif s < 15:
        color = "orange"
    else:
        color = "green"
    return {"Shape": "circle", "r": 0.6, "Color": color, "Filled": "true", "Layer": 1}


def make_server():
    # Use model default grid size so canvas matches initial model dimensions
    grid_vis = CanvasGrid(agent_portrayal, GRID_WIDTH_DEFAULT, GRID_HEIGHT_DEFAULT, 600, 600)
    
    class LorenzCurve(TextElement):
        def __init__(self, width=300, height=300):
            self.width = width
            self.height = height
            self.label = "Lorenz Curve"
            self.js_code = ""  # not using a custom JS; we render HTML directly

        def render(self, model):
            sugars = sorted([a.sugar for a in model.schedule.agents if isinstance(a, SugarAgent) and a.alive])
            n = len(sugars)
            total = float(sum(sugars)) if n > 0 else 0.0
            if n == 0 or total <= 0:
                return f"<div style='font-family: sans-serif; padding: 6px;'>Lorenz Curve: no data</div>"
            cum = np.cumsum(sugars) / total
            pop = np.arange(1, n + 1) / n
            # prepend origin
            cum = np.concatenate(([0.0], cum))
            pop = np.concatenate(([0.0], pop))
            # Build SVG path; coordinate system is top-left origin
            W, H = self.width, self.height
            points = [f"{int(W * x)},{int(H * (1 - y))}" for x, y in zip(pop, cum)]
            path_d = "M " + " L ".join(points)
            # equality line from bottom-left to top-right
            eq_line = f"M 0,{H} L {W},0"
            svg = f"""
            <div style='font-family: sans-serif;'>
              <div style='margin-bottom:4px;'>{self.label}</div>
              <svg width='{W}' height='{H}' viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' style='border:1px solid #ddd;background:#ffffff;'>
                <path d='{eq_line}' stroke='#999' stroke-width='1' fill='none' />
                <path d='{path_d}' stroke='#000' stroke-width='2' fill='none' />
                <!-- shade area below curve -->
                <polygon points='0,{H} {" ".join(points)} {W},0 {W},{H}' fill='rgba(100,149,237,0.25)' />
                <!-- shade area above curve -->
                <polygon points='0,{H} 0,0 {W},0 {" ".join(points[::-1])}' fill='rgba(200,200,200,0.35)' />
                <!-- axis labels -->
                <text x='{int(W/2)}' y='{H-6}' font-size='12' text-anchor='middle' fill='#333'>Cumulative population share</text>
                <text x='12' y='{int(H/2)}' font-size='12' text-anchor='middle' fill='#333' transform='rotate(-90, 12, {int(H/2)})'>Cumulative sugar share</text>
              </svg>
            </div>
            """
            return svg

    lorenz_element = LorenzCurve(width=360, height=360)

    class GiniSparkline(TextElement):
        def __init__(self, width=360, height=140, max_points=200):
            self.width = width
            self.height = height
            self.max_points = max_points
            self.history = []
            self.label = "Gini (sparkline)"

        def render(self, model):
            sugars = [a.sugar for a in model.schedule.agents if isinstance(a, SugarAgent) and a.alive]
            if len(sugars) == 0:
                g = 0.0
            else:
                x = np.sort(np.array(sugars, dtype=float))
                n = len(x)
                g = (2 * np.sum((np.arange(1, n + 1) * x))) / (n * np.sum(x)) - (n + 1) / n if np.sum(x) > 0 else 0.0
            self.history.append(max(0.0, min(1.0, float(g))))
            if len(self.history) > self.max_points:
                self.history = self.history[-self.max_points:]
            W, H = self.width, self.height
            m = len(self.history)
            if m <= 1:
                return f"<div style='font-family:sans-serif;padding:6px;'>Gini (sparkline): no data</div>"
            xs = np.linspace(0, W, m)
            pts = [f"{int(xs[i])},{int(H * (1 - self.history[i]))}" for i in range(m)]
            path_d = "M " + " L ".join(pts)
            svg = f"""
            <div style='font-family:sans-serif;'>
              <div style='margin-bottom:4px;'>{self.label}</div>
              <svg width='{W}' height='{H}' viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' style='border:1px solid #ddd;background:#fff;'>
                <rect x='0' y='0' width='{W}' height='{H}' fill='#ffffff' />
                <path d='{path_d}' stroke='#1f77b4' stroke-width='2' fill='none' />
                <!-- axis labels -->
                <text x='{int(W/2)}' y='{H-6}' font-size='12' text-anchor='middle' fill='#333'>Time (steps)</text>
                <text x='12' y='{int(H/2)}' font-size='12' text-anchor='middle' fill='#333' transform='rotate(-90, 12, {int(H/2)})'>Gini</text>
              </svg>
            </div>
            """
            return svg

    class DeathRateSparkline(TextElement):
        def __init__(self, width=360, height=140, max_points=200):
            self.width = width
            self.height = height
            self.max_points = max_points
            self.history = []
            self.label = "Death rate (per step)"

        def render(self, model):
            # Rate is deaths in this step divided by initial N
            denom = getattr(model, "N", None)
            if denom is None or denom <= 0:
                denom = max(1, len([a for a in model.schedule.agents if isinstance(a, SugarAgent)]))
            rate = float(model.deaths_this_step) / float(denom)
            rate = max(0.0, min(1.0, rate))
            self.history.append(rate)
            if len(self.history) > self.max_points:
                self.history = self.history[-self.max_points:]
            W, H = self.width, self.height
            m = len(self.history)
            if m <= 1:
                return f"<div style='font-family:sans-serif;padding:6px;'>Death rate: no data</div>"
            xs = np.linspace(0, W, m)
            pts = [f"{int(xs[i])},{int(H * (1 - self.history[i]))}" for i in range(m)]
            path_d = "M " + " L ".join(pts)
            svg = f"""
            <div style='font-family:sans-serif;'>
              <div style='margin-bottom:4px;'>{self.label}</div>
              <svg width='{W}' height='{H}' viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' style='border:1px solid #ddd;background:#fff;'>
                <rect x='0' y='0' width='{W}' height='{H}' fill='#ffffff' />
                <path d='{path_d}' stroke='#d62728' stroke-width='2' fill='none' />
                <!-- axis labels -->
                <text x='{int(W/2)}' y='{H-6}' font-size='12' text-anchor='middle' fill='#333'>Time (steps)</text>
                <text x='12' y='{int(H/2)}' font-size='12' text-anchor='middle' fill='#333' transform='rotate(-90, 12, {int(H/2)})'>Death rate</text>
              </svg>
            </div>
            """
            return svg

    class VisionBars(TextElement):
        def __init__(self, width=360, height=160):
            self.width = width
            self.height = height
            self.label = "Vision distribution"

        def render(self, model):
            counts = [sum(1 for a in model.schedule.agents if isinstance(a, SugarAgent) and a.alive and a.vision == v) for v in range(1, 7)]
            W, H = self.width, self.height
            max_c = max(counts) if counts else 1
            bar_w = int(W / 6) - 8
            bars = []
            for i, c in enumerate(counts):
                h = int((c / max_c) * (H - 30))
                x = 4 + i * (bar_w + 8)
                y = H - h - 20
                color = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"][i]
                bars.append(f"<rect x='{x}' y='{y}' width='{bar_w}' height='{h}' fill='{color}' />")
                bars.append(f"<text x='{x + bar_w/2}' y='{H - 6}' font-size='12' text-anchor='middle' fill='#333'>V{i+1}</text>")
            svg = f"""
            <div style='font-family:sans-serif;'>
              <div style='margin-bottom:4px;'>{self.label}</div>
              <svg width='{W}' height='{H}' viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' style='border:1px solid #ddd;background:#fff;'>
                {''.join(bars)}
                <!-- axis labels -->
                <text x='{int(W/2)}' y='{H-4}' font-size='12' text-anchor='middle' fill='#333'>Vision</text>
                <text x='12' y='{int(H/2)}' font-size='12' text-anchor='middle' fill='#333' transform='rotate(-90, 12, {int(H/2)})'>Agents</text>
              </svg>
            </div>
            """
            return svg

    gini_element = GiniSparkline(width=360, height=140)
    death_element = DeathRateSparkline(width=360, height=140)
    vision_element = VisionBars(width=360, height=160)

    # Initialize sliders from model defaults so UI reflects code-level defaults
    model_params = {
        "N": Slider("N (agents)", N_DEFAULT, 10, 250, 10),
        "width": Slider("Grid width", GRID_WIDTH_DEFAULT, 10, 80, 5),
        "height": Slider("Grid height", GRID_HEIGHT_DEFAULT, 10, 80, 5),
        "sigma": Slider("Sigma (std dev)", SIGMA_DEFAULT, 1.0, 20.0, 0.5),
        "grow_rate": Slider("Sugar grow rate", GROW_RATE_DEFAULT, 0.0, 5.0, 0.1),
        "tax_rate": Slider("Inheritance tax rate", TAX_RATE_DEFAULT, 0.0, 1.0, 0.05),
        "lifespan_min": Slider("Lifespan min", LIFESPAN_MIN_DEFAULT, 10, 200, 5),
        "lifespan_max": Slider("Lifespan max", LIFESPAN_MAX_DEFAULT, 10, 300, 5),
        "welfare_percent": Slider("Welfare threshold (% bottom)", WELFARE_PERCENT_DEFAULT, 0.0, 50.0, 1.0),
        "welfare_amount": Slider("Welfare amount (sugar)", WELFARE_AMOUNT_DEFAULT, 0.0, 10.0, 0.5),
    }

    server = ModularServer(
        SugarModel,
        [grid_vis, lorenz_element, gini_element, death_element, vision_element],
        "Sugarscape (Basic)",
        model_params,
    )
    server.port = 8521
    return server


if __name__ == "__main__":
    make_server().launch()