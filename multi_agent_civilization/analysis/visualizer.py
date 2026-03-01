from __future__ import annotations
import os
from typing import List


class SimulationVisualizer:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def plot_resource_distribution(self, turn_history: List[dict], filename: str = "resources.png") -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        turns = [t["turn"] for t in turn_history]
        # Use gini as proxy for resource distribution
        gini_vals = [t.get("gini", 0) for t in turn_history]
        fig, ax = plt.subplots()
        ax.plot(turns, gini_vals, label="Gini (food inequality)")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Gini Coefficient")
        ax.set_title("Resource Distribution Over Time")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close(fig)

    def plot_agent_health(self, turn_history: List[dict], filename: str = "health.png") -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        turns = [t["turn"] for t in turn_history]
        health_vals = [t.get("avg_health", 1.0) for t in turn_history]
        fig, ax = plt.subplots()
        ax.plot(turns, health_vals, color="green", label="Avg Health")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Average Health")
        ax.set_title("Agent Health Over Time")
        ax.set_ylim(0, 1.1)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close(fig)

    def plot_faction_evolution(self, turn_history: List[dict], filename: str = "factions.png") -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        turns = [t["turn"] for t in turn_history]
        faction_counts = [t.get("faction_count", 0) for t in turn_history]
        fig, ax = plt.subplots()
        ax.step(turns, faction_counts, color="purple", label="Faction Count")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Number of Factions")
        ax.set_title("Faction Evolution Over Time")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close(fig)

    def plot_all(self, turn_history: List[dict]) -> None:
        try:
            self.plot_resource_distribution(turn_history)
            self.plot_agent_health(turn_history)
            self.plot_faction_evolution(turn_history)
        except ImportError:
            pass
