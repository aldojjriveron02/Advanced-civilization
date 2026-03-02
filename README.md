# Multi-Agent Civilization Simulation

A production-ready multi-agent simulation powered by LLM-driven autonomous agents. Ten agents with distinct archetypes and personalities survive in a dynamic environment through emergent behavior.

## Overview

Agents must gather resources, form alliances, build structures, trade, fight, and develop culture — all driven by LLM reasoning rather than scripted outcomes. The simulation features:

- **10 unique agent archetypes** with Big Five personality traits
- **5 regions** with distinct resources and terrain
- **16 action types** with full resolution logic
- **14-phase turn engine** with simultaneous action declaration
- **Emergent faction formation** through relationship clustering
- **3-tier knowledge/innovation system**
- **10 random world events**
- **Markov chain weather system** with seasonal modifiers

## Installation

Requirements: Python 3.10+

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_key_here
```

## Quick Start

```bash
python main.py --turns 100 --model gpt-4o --seed 42
```

## Agent Archetypes

| Archetype | Focus | Key Traits |
|-----------|-------|------------|
| Builder | Construction | High conscientiousness, low aggression |
| Diplomat | Alliances | High extraversion, agreeableness |
| Opportunist | Personal gain | High greed, low honesty |
| Healer | Medicine | High agreeableness, low greed |
| Explorer | Discovery | High openness, curiosity |
| Hoarder | Resource accumulation | High greed, low agreeableness |
| Inventor | Innovation | High openness, curiosity |
| Protector | Defense | High loyalty, aggression |
| Altruist | Sharing | Highest agreeableness, lowest greed |
| Strategist | Planning | High conscientiousness, cognitive bandwidth |

## Architecture

- `models/` — Core data structures (WorldState, Agent, Actions, Factions)
- `systems/` — Simulation logic (TurnEngine, ActionResolver, FactionManager, etc.)
- `analysis/` — Metrics and visualization
- `tests/` — Unit tests

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--turns` | 100 | Number of simulation turns |
| `--model` | gpt-4o | OpenAI model to use |
| `--output` | auto | Output directory |
| `--seed` | None | Random seed for reproducibility |

## Output Format

Each run creates a timestamped directory under `simulation_output/` containing:
- `turn_XXXX.json` — Per-turn metrics snapshot
- `final_report.json` — Complete simulation summary
- `resources.png`, `health.png`, `factions.png` — Visualizations (if matplotlib available)

## Running Tests

```bash
pytest multi_agent_civilization/tests/ -v
```
