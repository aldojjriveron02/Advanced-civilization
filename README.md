# Advanced Civilization — Emotion Simulation

A multi-agent simulation where agents think, feel, and react to each other and the
world using a deep 9-emotion system with contagion, breaking points, complex
emotions, memory, and mood.

## Quick Start

```bash
# Install dependencies
pip install pygame matplotlib

# Text-only mode (prints turn summaries to the terminal)
python simulation.py

# 2D visual mode — opens a pygame window showing the live world
python simulation.py --visual

# Options
python simulation.py --visual --turns 100 --delay 0.5 --verbose
```

| Flag | Default | Description |
|------|---------|-------------|
| `--visual` | off | Open the 2D pygame window |
| `--turns N` | 200 | Number of turns to simulate |
| `--delay S` | 0.3 | Seconds between turns |
| `--seed N` | 42 | Random seed for reproducibility |
| `--verbose` | off | Show all agents each turn (not just active ones) |

Press **Escape** or close the window to stop.

## 2D Visualizer

The pygame window shows:

- **World map** — 7 named regions arranged on a grid
- **Agent circles** — color represents the dominant emotion; brightness reflects intensity; a pulsing ring grows with emotion strength
- **Emotion colors** — legend in the bottom-right corner
- **Right sidebar** — per-agent cards with name, region, dominant emotion bar, and mood bar
- **Bottom log** — live world events and breaking point alerts

## Emotion Colors

| Emotion | Color |
|---------|-------|
| Fear | Purple |
| Anger | Red |
| Hope | Green |
| Grief | Dark Blue |
| Gratitude | Gold |
| Pride | Orange |
| Envy | Dark Green |
| Loneliness | Steel Blue |
| Disgust | Olive |

## Modules

| File | Purpose |
|------|---------|
| `agent_state.py` | Agent, EmotionType, Mood, Memory, Personality, Needs |
| `emotion_system.py` | Contagion, Breaking Points, Complex Emotion Detector |
| `memory_manager.py` | Memory retrieval, vivid memory filtering, pruning |
| `communication_parser.py` | Speech modifier, masking detection, message pipeline |
| `turn_engine.py` | 18-phase turn orchestrator |
| `simulation.py` | Autonomous runner — world events, agent messages, turn loop |
| `visualizer.py` | 2D pygame visualizer |

## Running Tests

```bash
pip install pytest
pytest tests/test_emotion_system.py -v
```
