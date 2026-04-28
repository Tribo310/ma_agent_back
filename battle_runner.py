"""
Run a battle_v4 episode and stream JSON frames over a WebSocket.

Design notes
------------
* Uses magent2's PettingZoo *parallel* API so every step is a full tick
  (all living agents act simultaneously).  The AEC agent_iter() equivalent
  would process agents one-by-one; the parallel API is semantically
  identical but much more efficient for batch torch inference.
* Torch inference is dispatched to a thread-pool executor so the event
  loop (and the WebSocket keepalive) never blocks.
* Grid positions are extracted from the underlying GridWorld object that
  sits two layers beneath the parallel wrapper:
      parallel_env.env  -> magent_aec_env
      magent_aec_env.env -> GridWorld
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import WebSocket

MAP_SIZE = 45
MAX_CYCLES = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_infer(model: nn.Module, obs_list: list[np.ndarray]) -> list[int]:
    """Run a single forward pass for a batch of observations."""
    batch = torch.tensor(np.stack(obs_list), dtype=torch.float32)
    with torch.no_grad():
        logits = model(batch)
    return torch.argmax(logits, dim=-1).tolist()


def _get_positions(env: Any) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Extract (x, y) positions for both teams from the internal GridWorld.

    Tries env.env.env (parallel -> aec -> GridWorld) then env.env as fallback.
    Returns empty lists silently if internal API is unavailable.
    """
    try:
        inner = getattr(env, "env", None)
        if inner is None:
            return [], []
        gridworld = getattr(inner, "env", inner)
        handles = gridworld.get_handles()
        def _to_pairs(arr: np.ndarray) -> list[tuple[int, int]]:
            return [(int(p[0]), int(p[1])) for p in arr]
        return _to_pairs(gridworld.get_pos(handles[0])), _to_pairs(gridworld.get_pos(handles[1]))
    except Exception:
        return [], []


def _build_grid(red_pos: list[tuple[int, int]], blue_pos: list[tuple[int, int]]) -> tuple[list[list[int]], list[list[int]]]:
    grid: list[list[int]] = [[0] * MAP_SIZE for _ in range(MAP_SIZE)]
    team: list[list[int]] = [[0] * MAP_SIZE for _ in range(MAP_SIZE)]
    for x, y in red_pos:
        if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
            grid[y][x] = 1
            team[y][x] = 1
    for x, y in blue_pos:
        if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
            grid[y][x] = 1
            team[y][x] = 2
    return grid, team


# ---------------------------------------------------------------------------
# Main coroutine
# ---------------------------------------------------------------------------

async def run_battle(websocket: WebSocket, red_model: nn.Module, blue_model: nn.Module) -> None:
    from magent2.environments import battle_v4

    loop = asyncio.get_event_loop()

    env = battle_v4.parallel_env(map_size=MAP_SIZE, max_cycles=MAX_CYCLES)

    # PettingZoo <1.22 returns just observations; >=1.22 returns (obs, infos)
    reset_result = env.reset()
    observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    tick = 0

    while env.agents:
        # ---- collect living agents by team --------------------------------
        red_agents = [a for a in env.agents if a.startswith("red")]
        blue_agents = [a for a in env.agents if a.startswith("blue")]

        # ---- batch inference in executor so event loop stays free ---------
        actions: dict[str, int] = {}

        if red_agents:
            red_obs = [observations[a] for a in red_agents]
            red_actions: list[int] = await loop.run_in_executor(
                None, _batch_infer, red_model, red_obs
            )
            for agent, action in zip(red_agents, red_actions):
                actions[agent] = action

        if blue_agents:
            blue_obs = [observations[a] for a in blue_agents]
            blue_actions: list[int] = await loop.run_in_executor(
                None, _batch_infer, blue_model, blue_obs
            )
            for agent, action in zip(blue_agents, blue_actions):
                actions[agent] = action

        # ---- step ----------------------------------------------------------
        # PettingZoo <1.22: (obs, rewards, dones, infos)  — 4 values
        # PettingZoo >=1.22: (obs, rewards, terms, truncs, infos) — 5 values
        step_result = env.step(actions)
        observations = step_result[0]
        tick += 1

        red_alive = sum(1 for a in env.agents if a.startswith("red"))
        blue_alive = sum(1 for a in env.agents if a.startswith("blue"))

        # ---- build grid frame ----------------------------------------------
        red_pos, blue_pos = _get_positions(env)
        grid, team_grid = _build_grid(red_pos, blue_pos)

        frame: dict[str, Any] = {
            "tick": tick,
            "red_alive": red_alive,
            "blue_alive": blue_alive,
            "grid": grid,
            "team": team_grid,
        }
        await websocket.send_text(json.dumps(frame))

        # Yield control so WS heartbeats / other coroutines can run
        await asyncio.sleep(0)

        if tick >= MAX_CYCLES:
            break

    # ---- determine winner --------------------------------------------------
    red_alive = sum(1 for a in env.agents if a.startswith("red"))
    blue_alive = sum(1 for a in env.agents if a.startswith("blue"))

    if red_alive > blue_alive:
        winner = "red"
    elif blue_alive > red_alive:
        winner = "blue"
    else:
        winner = "draw"

    await websocket.send_text(json.dumps({"done": True, "winner": winner}))
    env.close()
