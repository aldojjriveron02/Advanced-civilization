#!/usr/bin/env python3
"""
Multi-Agent Civilization Simulation
Entry point for running the simulation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'multi_agent_civilization'))

import json
import argparse
from datetime import datetime


def run_simulation(num_turns=100, model="gpt-4o", output_dir=None, seed=None):
    from models.world_state import WorldState
    from systems.world_init import create_default_world
    from systems.turn_engine import TurnEngine
    from systems.llm_interface import LLMInterface, LLMConfig
    from analysis.visualizer import SimulationVisualizer
    from analysis.metrics import compute_metrics

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("simulation_output", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    world = create_default_world(seed=seed)
    config = LLMConfig(model=model)
    llm = LLMInterface(config=config)
    engine = TurnEngine(world, llm, seed=seed)

    turn_history = []
    deaths = []

    print(f"Starting simulation: {num_turns} turns, model={model}, seed={seed}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    for turn in range(num_turns):
        snapshot = engine.execute_turn()
        turn_history.append({
            "turn": snapshot.turn,
            "season": snapshot.season,
            "weather": snapshot.weather,
            "alive_count": snapshot.alive_count,
            "gini": snapshot.gini,
            "alliance_density": snapshot.alliance_density,
            "avg_health": snapshot.avg_health,
            "avg_hunger": snapshot.avg_hunger,
            "faction_count": snapshot.faction_count,
            "structure_count": snapshot.structure_count,
            "events": snapshot.events,
        })
        deaths.extend(snapshot.deaths)

        print(f"Turn {snapshot.turn:3d} | Season: {snapshot.season:6s} | Weather: {snapshot.weather:8s} | "
              f"Alive: {snapshot.alive_count:2d} | Gini: {snapshot.gini:.3f} | "
              f"Alliance: {snapshot.alliance_density:.3f} | Factions: {snapshot.faction_count}")
        if snapshot.events:
            print(f"         Events: {', '.join(snapshot.events)}")

        # Save per-turn JSON
        turn_file = os.path.join(output_dir, f"turn_{snapshot.turn:04d}.json")
        with open(turn_file, 'w') as f:
            json.dump(turn_history[-1], f, indent=2)

        # Check termination
        if snapshot.alive_count == 0:
            print("\nAll agents have died. Simulation ended early.")
            break

    # Final report
    metrics = compute_metrics(world, turn_history)
    survivors = [a for a in world.agents.values() if a.is_alive]
    report = {
        "total_turns": len(turn_history),
        "survivors": [{"id": a.id, "name": a.name, "archetype": a.archetype.value, "health": a.health} for a in survivors],
        "deaths": deaths,
        "factions": [{"id": f.id, "name": f.name, "members": f.members, "size": f.size} for f in world.factions.values()],
        "structures": sum(len(r.structures) for r in world.regions.values()),
        "metrics": metrics,
        "llm_calls": llm.call_count,
        "total_tokens": llm.total_tokens,
    }
    report_file = os.path.join(output_dir, "final_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"Simulation complete!")
    print(f"Survivors: {len(survivors)}/{10}")
    print(f"Factions: {len(world.factions)}")
    print(f"LLM calls: {llm.call_count}, tokens: {llm.total_tokens}")
    print(f"Report saved to: {report_file}")

    # Visualize
    viz = SimulationVisualizer(output_dir)
    viz.plot_all(turn_history)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Civilization Simulation")
    parser.add_argument("--turns", type=int, default=100, help="Number of turns")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()
    run_simulation(num_turns=args.turns, model=args.model, output_dir=args.output, seed=args.seed)
