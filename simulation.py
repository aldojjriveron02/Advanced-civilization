"""
simulation.py — Autonomous simulation runner for the Advanced Civilization emotion system.

Run from the repo root:
    python simulation.py              # Text-only mode
    python simulation.py --visual     # Opens a 2D pygame window
    python simulation.py --turns 50   # Run for 50 turns then exit
    python simulation.py --visual --turns 100

Press Ctrl+C to stop at any time.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import List, Optional

from agent_state import Agent, EmotionType, Personality
from communication_parser import CommunicationPipeline, Message
from turn_engine import TurnEngine


# ---------------------------------------------------------------------------
# World layout — agents distributed across named regions on a 2-D grid
# ---------------------------------------------------------------------------

# Each region has a display-coordinate (col, row) used by the visualizer.
REGIONS: dict[str, tuple[float, float]] = {
    "Northkeep":  (1, 0),
    "Eastholm":   (3, 0),
    "Thornwood":  (0, 2),
    "Cinderfall": (2, 2),
    "Greymarsh":  (4, 2),
    "Sunvale":    (1, 4),
    "Ashridge":   (3, 4),
}

# Pixel jitter so agents in the same region don't overlap.  Injected by the
# visualizer, ignored here.


def _make_agent(
    agent_id: str,
    name: str,
    region: str,
    *,
    extraversion: float,
    neuroticism: float,
    conscientiousness: float,
    honesty: float,
    agreeableness: float,
) -> Agent:
    agent = Agent(
        agent_id=agent_id,
        name=name,
        region=region,
        personality=Personality(
            extraversion=extraversion,
            neuroticism=neuroticism,
            conscientiousness=conscientiousness,
            honesty=honesty,
            agreeableness=agreeableness,
        ),
    )
    return agent


def build_world(seed: int = 0) -> List[Agent]:
    """Create the starting population."""
    rng = random.Random(seed)

    def r(lo: float = 0.2, hi: float = 0.9) -> float:
        return round(rng.uniform(lo, hi), 2)

    def make(agent_id: str, name: str, region: str) -> Agent:
        agent = _make_agent(
            agent_id, name, region,
            extraversion=r(), neuroticism=r(), conscientiousness=r(),
            honesty=r(), agreeableness=r(),
        )
        # Set needs using the seeded rng for full reproducibility
        agent.needs.safety     = round(rng.uniform(0.4, 0.8), 2)
        agent.needs.belonging  = round(rng.uniform(0.3, 0.7), 2)
        agent.needs.sustenance = round(rng.uniform(0.5, 0.9), 2)
        agent.needs.health     = round(rng.uniform(0.7, 1.0), 2)
        return agent

    agents = [
        make("a01", "Aldric",   "Northkeep"),
        make("a02", "Seren",    "Northkeep"),
        make("a03", "Draven",   "Eastholm"),
        make("a04", "Lyra",     "Eastholm"),
        make("a05", "Cormac",   "Thornwood"),
        make("a06", "Isla",     "Thornwood"),
        make("a07", "Vorin",    "Cinderfall"),
        make("a08", "Nessa",    "Cinderfall"),
        make("a09", "Roan",     "Greymarsh"),
        make("a10", "Faelith",  "Greymarsh"),
        make("a11", "Oryn",     "Sunvale"),
        make("a12", "Mira",     "Sunvale"),
        make("a13", "Calder",   "Ashridge"),
        make("a14", "Thandrel", "Ashridge"),
    ]
    return agents


# ---------------------------------------------------------------------------
# Random world events injected each turn
# ---------------------------------------------------------------------------

_EVENT_TEMPLATES = [
    # (description, emotion_type, intensity_range, target="region"|"all"|"random")
    ("A wildfire threatens {region}!", EmotionType.FEAR, (0.4, 0.75), "region"),
    ("A bountiful harvest lifts spirits in {region}.", EmotionType.HOPE, (0.3, 0.6), "region"),
    ("Raiders attack {region}, sparking outrage.", EmotionType.ANGER, (0.5, 0.8), "region"),
    ("A plague quietly spreads through {region}.", EmotionType.GRIEF, (0.35, 0.65), "region"),
    ("Trade routes reopen, gratitude spreads.", EmotionType.GRATITUDE, (0.3, 0.55), "all"),
    ("A champion emerges, filling hearts with pride.", EmotionType.PRIDE, (0.3, 0.5), "random"),
    ("Rations are short; isolation sets in.", EmotionType.LONELINESS, (0.3, 0.6), "region"),
    ("A rival faction flaunts their wealth.", EmotionType.ENVY, (0.3, 0.55), "region"),
    ("Contaminated water disgusts the population.", EmotionType.DISGUST, (0.4, 0.65), "region"),
]

def _inject_random_events(
    agents: List[Agent],
    rng: random.Random,
    turn: int,
) -> list[str]:
    """Apply 0–2 random world events this turn.  Returns event descriptions."""
    num_events = rng.choices([0, 1, 2], weights=[3, 4, 2])[0]
    messages: list[str] = []
    for _ in range(num_events):
        template, emotion_type, (lo, hi), scope = rng.choice(_EVENT_TEMPLATES)
        intensity = round(rng.uniform(lo, hi), 2)
        region = rng.choice(list(REGIONS.keys()))
        desc = template.format(region=region)
        messages.append(f"[Turn {turn}] EVENT: {desc}")

        if scope == "region":
            targets = [a for a in agents if a.region == region]
        elif scope == "all":
            targets = agents
        else:  # "random" — single random agent
            targets = [rng.choice(agents)]

        for agent in targets:
            agent.add_emotion(emotion_type, intensity)
            # Needs tick from events
            if emotion_type in (EmotionType.GRIEF, EmotionType.FEAR):
                agent.needs.safety = max(0.0, agent.needs.safety - 0.05)
            elif emotion_type == EmotionType.HOPE:
                agent.needs.safety = min(1.0, agent.needs.safety + 0.03)
    return messages


def _inject_random_messages(
    agents: List[Agent],
    rng: random.Random,
    turn: int,
) -> List[Message]:
    """Generate 0–3 inter-agent messages this turn."""
    num_msgs = rng.choices([0, 1, 2, 3], weights=[2, 4, 3, 1])[0]
    messages: List[Message] = []
    topics = [
        "We should stand together against the coming threat.",
        "I cannot forgive what happened. Things must change.",
        "Better days are ahead — I believe in us.",
        "I have lost too much. I need support.",
        "Thank you for what you did for our region.",
        "Our achievements speak for themselves.",
        "Why do they thrive while we suffer?",
        "The silence here is unbearable.",
        "What they did is revolting. I won't forget it.",
    ]
    for _ in range(num_msgs):
        sender = rng.choice(agents)
        # Broadcast 30% of the time, otherwise pick 1–3 recipients
        if rng.random() < 0.3:
            recipients: list[str] = []   # broadcast
        else:
            pool = [a.agent_id for a in agents if a.agent_id != sender.agent_id]
            recipients = rng.sample(pool, min(rng.randint(1, 3), len(pool)))
        content = rng.choice(topics)
        messages.append(
            Message(sender_id=sender.agent_id, content=content, recipients=recipients, turn=turn)
        )
    return messages


# ---------------------------------------------------------------------------
# Text-mode display helpers
# ---------------------------------------------------------------------------

_EMOTION_ABBREV = {
    EmotionType.FEAR: "FEAR",
    EmotionType.ANGER: "ANGR",
    EmotionType.HOPE: "HOPE",
    EmotionType.GRIEF: "GRIF",
    EmotionType.GRATITUDE: "GRAT",
    EmotionType.PRIDE: "PRID",
    EmotionType.ENVY: "ENVY",
    EmotionType.LONELINESS: "LONE",
    EmotionType.DISGUST: "DISG",
}


def _bar(value: float, width: int = 10) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_turn_summary(
    turn: int,
    agents: List[Agent],
    events: list[str],
    metrics: dict,
    breaking_events: list,
    verbose: bool = False,
) -> None:
    print(f"\n{'═' * 68}")
    print(f"  TURN {turn:>3}   |  avg mood optimism: {metrics.get('avg_mood_optimism', 0):+.3f}"
          f"  |  contagion: {metrics.get('contagion_event_count', 0)}"
          f"  |  breaking pts: {metrics.get('breaking_point_count', 0)}")
    print(f"{'═' * 68}")

    for msg in events:
        print(f"  {msg}")

    if breaking_events:
        for evt in breaking_events:
            print(f"  ⚡ BREAKING POINT — {evt.breaking_point_type} ({evt.agent_id})")

    if verbose:
        for agent in agents:
            dom = agent.dominant_emotion()
            dom_str = f"{_EMOTION_ABBREV[dom.emotion_type]}={dom.intensity:.2f}" if dom else "neutral"
            mood_opt = agent.mood.optimism
            bar = _bar(max(0.0, (mood_opt + 1.0) / 2.0))
            print(f"  {agent.name:<10} [{agent.region:<12}] {dom_str:<14} mood {bar}")
    else:
        # Compact view — only show agents with notable emotion activity
        notable = [a for a in agents if a.dominant_emotion_intensity() > 0.3]
        for agent in notable:
            dom = agent.dominant_emotion()
            dom_str = f"{_EMOTION_ABBREV[dom.emotion_type]}={dom.intensity:.2f}" if dom else "neutral"
            print(f"  {agent.name:<10} [{agent.region:<12}] {dom_str}")


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run(
    *,
    turns: int = 200,
    seed: int = 42,
    turn_delay: float = 0.3,
    verbose: bool = False,
    visual: bool = False,
) -> None:
    """Run the autonomous simulation."""
    rng = random.Random(seed)
    agents = build_world(seed=seed)
    engine = TurnEngine(agents=agents, seed=seed)
    comm_pipeline = CommunicationPipeline()

    visualizer = None
    if visual:
        from visualizer import Visualizer  # type: ignore[import]
        visualizer = Visualizer(agents=agents, regions=REGIONS)
        visualizer.start()

    print("=" * 68)
    print("  ADVANCED CIVILIZATION — Emotion Simulation")
    print(f"  {len(agents)} agents  |  {len(REGIONS)} regions  |  {turns} turns")
    print("=" * 68)

    for turn_num in range(1, turns + 1):
        # Check if visualizer window was closed
        if visualizer and not visualizer.is_running():
            print("\n[Visualizer closed — exiting]")
            break

        # --- World events ---
        events = _inject_random_events(agents, rng, turn_num)

        # --- Agent messages ---
        messages = _inject_random_messages(agents, rng, turn_num)
        # Assign turn number (engine also does this, but set here for the pipeline)
        for msg in messages:
            msg.turn = turn_num

        # --- Run the turn engine ---
        snapshot = engine.run_turn(messages=messages)

        # --- Print text summary ---
        print_turn_summary(
            turn_num,
            agents,
            events,
            snapshot.metrics,
            snapshot.breaking_point_events,
            verbose=verbose,
        )

        # --- Update visualizer ---
        if visualizer:
            visualizer.update(turn=turn_num, snapshot=snapshot, events=events)

        time.sleep(turn_delay)

    if visualizer:
        visualizer.wait()
    print("\n[Simulation complete]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced Civilization — emotion simulation")
    p.add_argument("--turns",   type=int,   default=200, help="Number of turns to simulate (default: 200)")
    p.add_argument("--seed",    type=int,   default=42,  help="Random seed (default: 42)")
    p.add_argument("--delay",   type=float, default=0.3, help="Seconds between turns (default: 0.3)")
    p.add_argument("--verbose", action="store_true",     help="Show all agents every turn")
    p.add_argument("--visual",  action="store_true",     help="Open 2D pygame window")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        turns=args.turns,
        seed=args.seed,
        turn_delay=args.delay,
        verbose=args.verbose,
        visual=args.visual,
    )
