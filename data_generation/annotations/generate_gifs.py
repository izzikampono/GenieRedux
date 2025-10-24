import argparse
import multiprocessing
import os
import sys
from pathlib import Path

multiprocessing.set_start_method("fork")

import logging

import imageio
import numpy as np

import retro

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.generator.connector_retro_act import make_retro

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(processName)s: %(message)s")

CONTROL_ACTIONS = [
    "RIGHT",
    "LEFT",
    "UP",
    "DOWN",
    "ACTION_PRIMARY",
    "ACTION_SECONDARY",
]


def run_game(
    selected_game,
    n_frames,
    save_path,
    args,
    *,
    action_name=None,
    valid_action_combos=None,
):
    print("Running game", selected_game)
    try:
        env = make_retro(
            game=selected_game,
            state=args.state,
            scenario=args.scenario,
            skip_frames=args.skip_frames,
            render_mode="rgb_array",
            valid_action_combos=valid_action_combos,
        )
    except FileNotFoundError as e:
        print(e)
        exit(0)
    env.reset()

    step = 0
    frames = []
    if action_name is not None:
        for idx, combo in enumerate(env.combos):
            if len(combo) == 1 and combo[0].upper() == action_name.upper():
                action_vector = np.zeros(len(env.combos), dtype=np.int64)
                action_vector[idx] = 1
                break
        else:
            raise ValueError(
                f"Action '{action_name}' not available for game '{selected_game}'."
            )
    while step < n_frames:
        if action_name is None:
            sample = env.action_space.sample()
        else:
            sample = action_vector
        ob, totrew, terminated, truncated, info = env.step(sample)
        frames.append(ob)
        if terminated or truncated:
            env.reset()
        step += 1

    # Save frames as a GIF
    suffix = f"_{action_name.lower()}" if action_name is not None else ""
    save_path = os.path.join(save_path, f"{selected_game}{suffix}.gif")
    print(save_path)
    imageio.mimsave(save_path, frames, duration=1 / 30)

    env.close()


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--game", default="Airstriker-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--skip_frames", default=1, type=int)
    parser.add_argument("--scenario", default=None)
    parser.add_argument(
        "--output",
        "-o",
        dest="output_fpath",
        default="annotations/previews",
        help="Directory to save output GIFs",
    )
    parser.add_argument(
        "--mode",
        choices=["random", "control"],
        default="control",
        help="Generation mode. 'random' samples random actions. 'control' generates one rollout per action.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel worker processes.",
    )
    args = parser.parse_args()

    games_list = retro.data.list_games(inttype=retro.data.Integrations.ALL)
    selected_games = games_list
    n_frames = args.max_steps if args.max_steps is not None else 120
    output_fpath = args.output_fpath
    os.makedirs(output_fpath, exist_ok=True)

    pool = multiprocessing.Pool(args.workers)

    valid_action_combos = CONTROL_ACTIONS if args.mode == "control" else None

    for game in selected_games:
        if args.mode == "control":
            for action in CONTROL_ACTIONS:
                pool.apply_async(
                    run_game,
                    args=(game, n_frames, output_fpath, args),
                    kwds={
                        "action_name": action,
                        "valid_action_combos": valid_action_combos,
                    },
                )
        else:
            pool.apply_async(
                run_game,
                args=(game, n_frames, output_fpath, args),
                kwds={"valid_action_combos": valid_action_combos},
            )
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
