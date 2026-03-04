"""
visualizer.py — 2D pygame visualizer for the Advanced Civilization simulation.

Renders a top-down world map showing:
  • Regions as labeled tiles
  • Agents as colored circles (color = dominant emotion)
  • Emotion intensity encoded in circle brightness / pulse ring
  • A HUD sidebar with per-agent detail cards
  • A live event log at the bottom
  • An emotion color legend

Usage (via simulation.py):
    python simulation.py --visual
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import pygame

from agent_state import Agent, EmotionType
from turn_engine import TurnSnapshot


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

BLACK   = (10,  10,  10)
WHITE   = (240, 240, 240)
GREY_D  = (40,  40,  40)
GREY_M  = (80,  80,  80)
GREY_L  = (140, 140, 140)
PANEL   = (22,  22,  35)
TILE_BG = (30,  42,  55)
TILE_BD = (60,  80,  100)

# Dominant-emotion → RGB color
EMOTION_COLOR: Dict[EmotionType, Tuple[int, int, int]] = {
    EmotionType.FEAR:      (130, 60,  180),   # purple
    EmotionType.ANGER:     (220, 50,  50),    # red
    EmotionType.HOPE:      (80,  200, 120),   # green
    EmotionType.GRIEF:     (60,  80,  160),   # dark blue
    EmotionType.GRATITUDE: (240, 190, 60),    # gold
    EmotionType.PRIDE:     (240, 130, 30),    # orange
    EmotionType.ENVY:      (50,  170, 80),    # darker green
    EmotionType.LONELINESS:(100, 130, 180),   # steel blue
    EmotionType.DISGUST:   (110, 150, 50),    # olive
}
NEUTRAL_COLOR = (120, 120, 120)

# Emotion abbreviation labels
EMOTION_ABBREV: Dict[EmotionType, str] = {
    EmotionType.FEAR:      "FEAR",
    EmotionType.ANGER:     "ANGR",
    EmotionType.HOPE:      "HOPE",
    EmotionType.GRIEF:     "GRIF",
    EmotionType.GRATITUDE: "GRAT",
    EmotionType.PRIDE:     "PRID",
    EmotionType.ENVY:      "ENVY",
    EmotionType.LONELINESS:"LONE",
    EmotionType.DISGUST:   "DISG",
}


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

SCREEN_W = 1280
SCREEN_H = 780

MAP_LEFT   = 0
MAP_TOP    = 0
MAP_RIGHT  = 780
MAP_BOTTOM = 680

SIDEBAR_LEFT  = MAP_RIGHT + 4
SIDEBAR_W     = SCREEN_W - SIDEBAR_LEFT

LOG_TOP    = MAP_BOTTOM + 4
LOG_H      = SCREEN_H - LOG_TOP

GRID_COLS  = 5    # region grid columns
GRID_ROWS  = 5    # region grid rows

AGENT_R    = 14   # base circle radius
PULSE_MAX  = 8    # extra radius at peak pulse


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _lerp_color(
    c1: Tuple[int,int,int],
    c2: Tuple[int,int,int],
    t: float,
) -> Tuple[int,int,int]:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))  # type: ignore[return-value]


def _dim(color: Tuple[int,int,int], factor: float) -> Tuple[int,int,int]:
    return tuple(max(0, int(c * factor)) for c in color)  # type: ignore[return-value]


def _brightness(color: Tuple[int,int,int], intensity: float) -> Tuple[int,int,int]:
    """Brighten the color proportionally to intensity (0–1)."""
    factor = 0.3 + 0.7 * intensity
    return _dim(color, factor)


def _bar_surface(
    value: float,
    width: int,
    height: int,
    fg: Tuple[int,int,int],
    bg: Tuple[int,int,int] = GREY_D,
) -> pygame.Surface:
    surf = pygame.Surface((width, height))
    surf.fill(bg)
    filled = int(round(value * width))
    if filled > 0:
        pygame.draw.rect(surf, fg, (0, 0, filled, height))
    return surf


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class Visualizer:
    """Runs the pygame window in a background thread."""

    def __init__(
        self,
        agents: List[Agent],
        regions: Dict[str, Tuple[float, float]],
        target_fps: int = 30,
    ) -> None:
        self.agents = agents
        self.regions = regions
        self.target_fps = target_fps

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Shared state written by simulation, read by renderer
        self._turn: int = 0
        self._snapshot: Optional[TurnSnapshot] = None
        self._event_log: deque[str] = deque(maxlen=8)

        # Per-agent jitter so agents in the same region don't stack
        self._jitter: Dict[str, Tuple[int, int]] = {}
        self._assign_jitter()

        # Per-agent pulse animation phase (0–2π)
        self._pulse: Dict[str, float] = {a.agent_id: 0.0 for a in agents}

        # Pre-compute region tile pixel positions
        self._region_pixel: Dict[str, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the pygame window in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Give pygame a moment to initialise
        time.sleep(0.15)

    def is_running(self) -> bool:
        return self._running

    def update(
        self,
        turn: int,
        snapshot: TurnSnapshot,
        events: List[str],
    ) -> None:
        """Called by the simulation loop after each turn."""
        with self._lock:
            self._turn = turn
            self._snapshot = snapshot
            for evt in events:
                self._event_log.append(evt)

    def wait(self) -> None:
        """Block until the pygame window is closed."""
        if self._thread:
            self._thread.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_jitter(self) -> None:
        """Distribute agents within a region so their circles don't overlap."""
        region_counts: Dict[str, int] = {}
        for agent in self.agents:
            idx = region_counts.get(agent.region, 0)
            region_counts[agent.region] = idx + 1
            angle = idx * (2 * math.pi / 4)   # up to 4 slots per region
            radius = 28 if idx > 0 else 0
            jx = int(radius * math.cos(angle))
            jy = int(radius * math.sin(angle))
            self._jitter[agent.agent_id] = (jx, jy)

    def _region_to_pixel(self, region: str) -> Tuple[int, int]:
        """Convert (col, row) grid coords to pixel center on the map."""
        if region in self._region_pixel:
            return self._region_pixel[region]

        col, row = self.regions.get(region, (2, 2))
        cell_w = (MAP_RIGHT - MAP_LEFT) / GRID_COLS
        cell_h = (MAP_BOTTOM - MAP_TOP) / GRID_ROWS
        px = int(MAP_LEFT + (col + 0.5) * cell_w)
        py = int(MAP_TOP  + (row + 0.5) * cell_h)
        self._region_pixel[region] = (px, py)
        return (px, py)

    def _agent_pixel(self, agent: Agent) -> Tuple[int, int]:
        rx, ry = self._region_to_pixel(agent.region)
        jx, jy = self._jitter.get(agent.agent_id, (0, 0))
        return (rx + jx, ry + jy)

    # ------------------------------------------------------------------
    # Pygame render loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Advanced Civilization — Emotion Simulation")
        clock = pygame.time.Clock()

        font_sm  = pygame.font.SysFont("monospace", 11)
        font_med = pygame.font.SysFont("monospace", 13, bold=True)
        font_lg  = pygame.font.SysFont("monospace", 16, bold=True)
        font_hd  = pygame.font.SysFont("monospace", 18, bold=True)

        tick = 0
        while self._running:
            # --- Events ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False
                    return

            # Advance pulse animation
            tick += 1
            for aid in self._pulse:
                self._pulse[aid] = (self._pulse[aid] + 0.12) % (2 * math.pi)

            # Read shared state
            with self._lock:
                turn = self._turn
                snapshot = self._snapshot
                event_lines = list(self._event_log)

            screen.fill(BLACK)

            self._draw_map(screen, font_sm, font_med)
            self._draw_agents(screen, font_sm, font_med, tick)
            self._draw_sidebar(screen, font_sm, font_med, font_lg, font_hd, turn)
            self._draw_event_log(screen, font_sm, event_lines, snapshot)
            self._draw_legend(screen, font_sm)
            self._draw_hud_bar(screen, font_hd, turn)

            pygame.display.flip()
            clock.tick(self.target_fps)

        pygame.quit()

    # ------------------------------------------------------------------
    # Draw sections
    # ------------------------------------------------------------------

    def _draw_hud_bar(self, screen: pygame.Surface, font: pygame.font.Font, turn: int) -> None:
        """Top title / turn counter bar."""
        bar_rect = pygame.Rect(0, 0, MAP_RIGHT, 30)
        pygame.draw.rect(screen, PANEL, bar_rect)
        txt = font.render(f"  ADVANCED CIVILIZATION   Turn: {turn}", True, WHITE)
        screen.blit(txt, (8, 6))

    def _draw_map(
        self,
        screen: pygame.Surface,
        font_sm: pygame.font.Font,
        font_med: pygame.font.Font,
    ) -> None:
        """Draw the region grid tiles."""
        cell_w = (MAP_RIGHT - MAP_LEFT) / GRID_COLS
        cell_h = (MAP_BOTTOM - MAP_TOP - 30) / GRID_ROWS

        # Draw grid background
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                rx = int(MAP_LEFT + col * cell_w)
                ry = int(MAP_TOP + 30 + row * cell_h)
                rw = int(cell_w) - 2
                rh = int(cell_h) - 2
                pygame.draw.rect(screen, TILE_BG, (rx, ry, rw, rh), border_radius=6)
                pygame.draw.rect(screen, TILE_BD, (rx, ry, rw, rh), width=1, border_radius=6)

        # Draw region labels
        for region, (col, row) in self.regions.items():
            rx, ry = self._region_to_pixel(region)
            ry += 15  # shift label below center
            lbl = font_sm.render(region, True, GREY_L)
            screen.blit(lbl, (rx - lbl.get_width() // 2, ry - 8))

    def _draw_agents(
        self,
        screen: pygame.Surface,
        font_sm: pygame.font.Font,
        font_med: pygame.font.Font,
        tick: int,
    ) -> None:
        """Draw each agent as a pulsing colored circle."""
        for agent in self.agents:
            px, py = self._agent_pixel(agent)
            dom = agent.dominant_emotion()
            intensity = dom.intensity if dom else 0.0
            base_color = EMOTION_COLOR.get(dom.emotion_type, NEUTRAL_COLOR) if dom else NEUTRAL_COLOR
            color = _brightness(base_color, intensity)

            # Pulse ring — grows with emotion intensity
            phase = self._pulse.get(agent.agent_id, 0.0)
            pulse_r = int(AGENT_R + PULSE_MAX * intensity * abs(math.sin(phase)))
            ring_alpha = int(160 * intensity * abs(math.sin(phase)))
            if ring_alpha > 10:
                ring_surf = pygame.Surface((pulse_r * 2 + 4, pulse_r * 2 + 4), pygame.SRCALPHA)
                ring_color = (*base_color, ring_alpha)
                pygame.draw.circle(ring_surf, ring_color, (pulse_r + 2, pulse_r + 2), pulse_r + 2, width=2)
                screen.blit(ring_surf, (px - pulse_r - 2, py - pulse_r - 2))

            # Main circle
            pygame.draw.circle(screen, color, (px, py), AGENT_R)
            pygame.draw.circle(screen, WHITE, (px, py), AGENT_R, width=1)

            # Name label
            name_surf = font_sm.render(agent.name, True, WHITE)
            screen.blit(name_surf, (px - name_surf.get_width() // 2, py + AGENT_R + 2))

            # Emotion label below name
            if dom and intensity > 0.1:
                emo_lbl = f"{EMOTION_ABBREV[dom.emotion_type]} {intensity:.2f}"
                emo_surf = font_sm.render(emo_lbl, True, base_color)
                screen.blit(emo_surf, (px - emo_surf.get_width() // 2, py + AGENT_R + 14))

    def _draw_sidebar(
        self,
        screen: pygame.Surface,
        font_sm: pygame.font.Font,
        font_med: pygame.font.Font,
        font_lg: pygame.font.Font,
        font_hd: pygame.font.Font,
        turn: int,
    ) -> None:
        """Right sidebar: per-agent detail cards."""
        pygame.draw.rect(screen, PANEL, (SIDEBAR_LEFT, 0, SIDEBAR_W, SCREEN_H))
        pygame.draw.line(screen, GREY_M, (SIDEBAR_LEFT, 0), (SIDEBAR_LEFT, SCREEN_H), 2)

        x0 = SIDEBAR_LEFT + 8
        y = 8
        header = font_hd.render("AGENTS", True, WHITE)
        screen.blit(header, (x0, y))
        y += 26

        card_h = (SCREEN_H - y - 4) // max(1, len(self.agents))
        card_h = min(card_h, 50)

        for agent in self.agents:
            dom = agent.dominant_emotion()
            intensity = dom.intensity if dom else 0.0
            base_color = EMOTION_COLOR.get(dom.emotion_type, NEUTRAL_COLOR) if dom else NEUTRAL_COLOR

            # Card background
            card_rect = pygame.Rect(x0 - 4, y, SIDEBAR_W - 8, card_h - 2)
            pygame.draw.rect(screen, GREY_D, card_rect, border_radius=4)
            # Colored left strip
            strip_rect = pygame.Rect(x0 - 4, y, 4, card_h - 2)
            pygame.draw.rect(screen, base_color, strip_rect, border_radius=2)

            # Agent name
            name_s = font_med.render(agent.name, True, WHITE)
            screen.blit(name_s, (x0 + 4, y + 2))

            # Region
            reg_s = font_sm.render(agent.region, True, GREY_L)
            screen.blit(reg_s, (x0 + 4, y + 16))

            # Dominant emotion bar
            if dom and intensity > 0.01:
                emo_label = f"{EMOTION_ABBREV[dom.emotion_type]}"
                emo_s = font_sm.render(emo_label, True, base_color)
                screen.blit(emo_s, (x0 + 4, y + 28))
                bar_w = SIDEBAR_W - 60
                bar_surf = _bar_surface(intensity, bar_w, 7, base_color, GREY_D)
                screen.blit(bar_surf, (x0 + 34, y + 30))

            # Mood line
            opt = agent.mood.optimism
            opt_norm = (opt + 1.0) / 2.0   # -1…1 → 0…1
            opt_color = (80, 200, 80) if opt >= 0 else (200, 80, 80)
            mood_bar = _bar_surface(opt_norm, SIDEBAR_W - 60, 5, opt_color)
            screen.blit(mood_bar, (x0 + 34, y + card_h - 9))
            m_s = font_sm.render("mood", True, GREY_L)
            screen.blit(m_s, (x0 + 4, y + card_h - 10))

            y += card_h

    def _draw_event_log(
        self,
        screen: pygame.Surface,
        font_sm: pygame.font.Font,
        event_lines: List[str],
        snapshot: Optional[TurnSnapshot],
    ) -> None:
        """Bottom strip: live event log."""
        log_rect = pygame.Rect(0, LOG_TOP, MAP_RIGHT, LOG_H)
        pygame.draw.rect(screen, PANEL, log_rect)
        pygame.draw.line(screen, GREY_M, (0, LOG_TOP), (MAP_RIGHT, LOG_TOP), 1)

        y = LOG_TOP + 4
        # Breaking point events stand out
        if snapshot and snapshot.breaking_point_events:
            for bpe in snapshot.breaking_point_events[-3:]:
                txt = f"⚡ {bpe.breaking_point_type} — {bpe.agent_id}"
                s = font_sm.render(txt, True, (240, 120, 30))
                screen.blit(s, (6, y))
                y += 13

        # World events
        for line in reversed(event_lines):
            if y > LOG_TOP + LOG_H - 14:
                break
            s = font_sm.render(line[:90], True, GREY_L)
            screen.blit(s, (6, y))
            y += 13

    def _draw_legend(self, screen: pygame.Surface, font_sm: pygame.font.Font) -> None:
        """Small emotion-color legend in bottom-right of the map area."""
        lx = MAP_RIGHT - 130
        ly = MAP_BOTTOM - len(EMOTION_COLOR) * 14 - 4
        for i, (et, color) in enumerate(EMOTION_COLOR.items()):
            pygame.draw.circle(screen, color, (lx + 6, ly + i * 14 + 6), 5)
            label = font_sm.render(EMOTION_ABBREV[et], True, GREY_L)
            screen.blit(label, (lx + 16, ly + i * 14))
