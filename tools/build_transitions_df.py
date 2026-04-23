#!/usr/bin/env python
"""
Build a tidy transitions DataFrame from a sampled retro_act dataset.

Each row is one (src_frame -[action]-> tgt_frame) transition — the atomic unit
needed for action discovery. Supports both output modes:

  frame mode: individual JPEGs per step
    → columns: src_frame_path, tgt_frame_path  (absolute paths to .jpg)
    → video_path is None

  video mode: one MP4 per session, no per-frame JPEGs
    → columns: video_path (absolute path to frames.mp4)
    → src/tgt frame IDs index into the video; use tgt_frame_id to seek
    → src_frame_path / tgt_frame_path are None

Usage:
    python tools/build_transitions_df.py \
        --dataset_dpath /path/to/datasets/retro_act_v0.0.0 \
        [--output_fpath /path/to/transitions.parquet] \
        [--format parquet|csv]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def _parse_action(action_vec: list, action_captions: list) -> tuple:
    try:
        idx = action_vec.index(1)
        name = "_".join(action_captions[idx])
    except (ValueError, IndexError):
        idx = -1
        name = "NOOP"
    return idx, name


def build_transitions(dataset_dpath: Path) -> pd.DataFrame:
    """Walk dataset_dpath and return one DataFrame row per transition."""
    info_fpaths = sorted(dataset_dpath.rglob("info.json"))
    if not info_fpaths:
        raise FileNotFoundError(f"No info.json files found under {dataset_dpath}")

    rows = []

    for info_fpath in tqdm(info_fpaths, desc="Games"):
        game_dpath = info_fpath.parent

        with open(info_fpath) as f:
            info = json.load(f)

        game = info["info"]["game"]
        platform = game.rsplit("-", 1)[-1]
        action_captions = info["info"]["action_captions"]
        n_skip_frames = info["info"]["config"].get("n_skip_frames", 1)
        output_mode = info.get("generator_config", {}).get("output_mode", "frame")

        for actions_fpath in sorted(game_dpath.rglob("actions.json")):
            session_dpath = actions_fpath.parent
            rel_parts = session_dpath.relative_to(game_dpath).parts
            instance_id = int(rel_parts[0])
            session_id = int(rel_parts[1])

            # Resolve media — both may coexist when output_mode=both
            video_fpath = session_dpath / "frames.mp4"
            video_path = str(video_fpath) if video_fpath.exists() else None
            has_frames = (session_dpath / "frames").is_dir()
            is_video_mode = video_path is not None and not has_frames

            with open(actions_fpath) as f:
                transitions = json.load(f)["actions"]

            n = len(transitions)
            for i, entry in enumerate(transitions):
                src_id = entry["src_id"]
                tgt_id = entry["tgt_id"]
                action_vec = entry["action"]
                extras = entry.get("extras", {})

                action_idx, action_name = _parse_action(action_vec, action_captions)

                if has_frames:
                    tgt_frame_path = str(session_dpath / "frames" / f"{tgt_id:06d}.jpg")
                    src_frame_path = (
                        str(session_dpath / "frames" / f"{src_id:06d}.jpg")
                        if src_id >= 0
                        else None
                    )
                else:
                    src_frame_path = None
                    tgt_frame_path = None

                row = {
                    "game": game,
                    "platform": platform,
                    "instance_id": instance_id,
                    "session_id": session_id,
                    "src_frame_id": src_id,
                    "tgt_frame_id": tgt_id,
                    # frame mode: absolute paths to JPEGs; None in video mode
                    "src_frame_path": src_frame_path,
                    "tgt_frame_path": tgt_frame_path,
                    # video mode: path to MP4, seek with tgt_frame_id; None in frame mode
                    "video_path": video_path,
                    "action_idx": action_idx,
                    "action_name": action_name,
                    "n_skip_frames": n_skip_frames,
                    "is_episode_start": src_id < 0,
                    "is_episode_end": i == n - 1,
                }

                for k, v in extras.items():
                    row[f"extra_{k}"] = v

                rows.append(row)

    df = pd.DataFrame(rows)

    for col in ("instance_id", "session_id", "src_frame_id", "tgt_frame_id",
                "action_idx", "n_skip_frames"):
        df[col] = df[col].astype("int32")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build tidy transitions DataFrame from retro_act dataset"
    )
    parser.add_argument(
        "--dataset_dpath", required=True, type=Path,
        help="Root of the retro_act dataset (e.g. datasets/retro_act_v0.0.0)",
    )
    parser.add_argument(
        "--output_fpath", type=Path, default=None,
        help="Output path. Defaults to <dataset_dpath>/transitions.<format>",
    )
    parser.add_argument(
        "--format", choices=["parquet", "csv"], default="parquet",
        help="Output format (default: parquet)",
    )
    args = parser.parse_args()

    dataset_dpath = args.dataset_dpath.resolve()
    if not dataset_dpath.exists():
        raise FileNotFoundError(f"Not found: {dataset_dpath}")

    output_fpath = args.output_fpath or (dataset_dpath / f"transitions.{args.format}")

    print(f"Scanning: {dataset_dpath}")
    df = build_transitions(dataset_dpath)

    n_video = df["video_path"].notna().sum()
    n_frame = df["tgt_frame_path"].notna().sum()
    print(f"\n{len(df):,} transitions | {df['game'].nunique()} games | "
          f"{df.groupby(['game','instance_id','session_id']).ngroups} sessions")
    print(f"Storage mode: {n_video:,} video-mode rows | {n_frame:,} frame-mode rows")

    print("\nAction distribution:")
    print(df["action_name"].value_counts().to_string())

    print("\nColumns:")
    print(df.dtypes.to_string())

    if args.format == "parquet":
        df.to_parquet(output_fpath, index=False)
    else:
        df.to_csv(output_fpath, index=False)

    print(f"\nSaved → {output_fpath}")


if __name__ == "__main__":
    main()
