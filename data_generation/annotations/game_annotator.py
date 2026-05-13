import argparse
import csv
import os
from pathlib import Path
from tkinter import BooleanVar, Button, Checkbutton, Entry, Frame, Label, Tk

from PIL import Image, ImageSequence, ImageTk

CONTROL_ACTIONS = [
    "RIGHT",
    "LEFT",
    "UP",
    "DOWN",
    "ACTION_PRIMARY",
    "ACTION_SECONDARY",
]

FLAG_FIELDS = ["blinking", "delayed"]
TAG_FIELDS = ["genre", "motion", "view", "camera"]
PREVIEW_ONLY_ACTIONS = ["IDLE"]
ALL_ACTIONS = CONTROL_ACTIONS + PREVIEW_ONLY_ACTIONS
TAG_DEFAULT_VALUES = {"camera": "none"}

DEFAULT_ACTION_VALUES = {
    "RIGHT": "right",
    "LEFT": "left",
    "UP": "none",
    "DOWN": "crouch",
    "ACTION_PRIMARY": "none",
    "ACTION_SECONDARY": "none",
}


def normalize_game_name(name: str) -> str:
    return name.strip().lower()


class AnnotationTool:
    def __init__(
        self,
        preview_dpath,
        annotation_fpath,
        edit_only=False,
        new_only: bool = False,
        only_game: str | None = None,
    ):
        self.preview_dpath = Path(preview_dpath)
        self.annotation_fpath = Path(annotation_fpath)
        self.edit_only = edit_only
        self.new_only = new_only
        self.only_game = normalize_game_name(only_game) if only_game else None

        self.game_previews: dict[str, dict[str, Path]] = {}
        self.game_fallbacks: dict[str, dict[str, tuple[str, Path]]] = {}
        self.games: list[str] = []
        self.current_index = 0

        self.action_entries: dict[str, Entry] = {}
        self.tag_entries: dict[str, Entry] = {}
        self.animation_frames: dict[str, list[ImageTk.PhotoImage]] = {}
        self.preview_labels: dict[str, Label] = {}
        self.animation_jobs: dict[str, str] = {}
        self.preview_frame_cache: dict[Path, list[ImageTk.PhotoImage]] = {}

        self.annotation_rows: list[dict[str, str]] = []
        self.annotations: dict[str, dict[str, str]] = {}
        self.fieldnames: list[str] = []

        self.root = Tk()
        self.root.title("Control Annotation Tool")

        self.game_label = Label(self.root, text="", font=("Arial", 18, "bold"))
        self.game_label.pack(pady=(10, 0))

        self.preview_container = Frame(self.root)
        self.preview_container.pack(padx=10, pady=10)

        self.flag_frame = Frame(self.root)
        self.flag_frame.pack(pady=(0, 10))

        self.blink_var = BooleanVar(value=False)
        self.delayed_var = BooleanVar(value=False)

        Checkbutton(
            self.flag_frame, text="Blinking", variable=self.blink_var
        ).pack(side="left", padx=10)
        Checkbutton(
            self.flag_frame, text="Delayed actions", variable=self.delayed_var
        ).pack(side="left", padx=10)

        for tag in TAG_FIELDS:
            tag_frame = Frame(self.flag_frame)
            tag_frame.pack(side="left", padx=10)
            Label(tag_frame, text=tag.capitalize()).pack(side="top")
            entry = Entry(tag_frame, width=15)
            entry.pack(side="top")
            self.tag_entries[tag] = entry

        self.next_button = Button(self.root, text="Save & Next", command=self.next_game)
        self.next_button.pack(pady=(0, 10))

        self.status_label = Label(self.root, text="", font=("Arial", 10), fg="blue")
        self.status_label.pack(pady=(0, 10))

        self.root.bind("<Return>", lambda event: self.next_game())

        self.load_annotations()

    def load_annotations(self):
        if self.annotation_fpath.exists():
            with self.annotation_fpath.open() as file:
                reader = csv.DictReader(file)
                self.fieldnames = reader.fieldnames[:] if reader.fieldnames else []
                for row in reader:
                    game = row.get("game")
                    if not game:
                        continue
                    self.annotation_rows.append(row)
                    self.annotations[normalize_game_name(game)] = row
        else:
            self.fieldnames = []

        if "game" not in self.fieldnames:
            self.fieldnames.insert(0, "game")
        for action in CONTROL_ACTIONS:
            if action not in self.fieldnames:
                self.fieldnames.append(action)
        for flag in FLAG_FIELDS:
            if flag not in self.fieldnames:
                self.fieldnames.append(flag)
        for tag in TAG_FIELDS:
            if tag not in self.fieldnames:
                self.fieldnames.append(tag)

    def write_annotations(self):
        self.annotation_fpath.parent.mkdir(parents=True, exist_ok=True)
        with self.annotation_fpath.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in self.annotation_rows:
                writer.writerow(
                    {field: row.get(field, "") or "" for field in self.fieldnames}
                )

    def run(self):
        self.load_preview_files()
        if self.only_game is not None:
            filtered = [
                name
                for name in self.games
                if normalize_game_name(name) == self.only_game
            ]
            self.game_previews = {name: self.game_previews[name] for name in filtered}
            self.games = filtered
        if not self.games:
            self.status_label.config(
                text="No action previews found in the selected directory."
            )
        else:
            self.display_current_game()
        self.root.mainloop()

    def load_preview_files(self):
        if not self.preview_dpath.exists():
            self.status_label.config(
                text=f"Preview directory '{self.preview_dpath}' not found."
            )
            return

        self.game_previews = {}
        self.game_fallbacks = {}

        for filename in os.listdir(self.preview_dpath):
            if not filename.endswith(".gif"):
                continue
            lowered = filename.lower()
            matched_action = None
            for action in ALL_ACTIONS:
                suffix = f"_{action.lower()}.gif"
                if lowered.endswith(suffix):
                    matched_action = action
                    base_name = filename[: -len(suffix)]
                    game_entry = self.game_previews.setdefault(base_name, {})
                    game_entry[matched_action] = self.preview_dpath / filename
                    break
            if matched_action is None:
                fallback_suffixes = {
                    "ACTION_PRIMARY": "_action_jump.gif",
                }
                for target_action, suffix in fallback_suffixes.items():
                    if lowered.endswith(suffix):
                        base_name = filename[: -len(suffix)]
                        self.game_previews.setdefault(base_name, {})
                        fallback_entry = self.game_fallbacks.setdefault(base_name, {})
                        fallback_entry[target_action] = ("ACTION_PRIMARY", self.preview_dpath / filename)
                        matched_action = target_action
                        break
                if matched_action is None:
                    continue

        all_games = set(self.game_previews.keys()) | set(self.game_fallbacks.keys())
        for game_name in all_games:
            entry = self.game_previews.setdefault(game_name, {})
            fallbacks = self.game_fallbacks.get(game_name, {})
            for target_action, (source_action, path) in (fallbacks or {}).items():
                if target_action not in entry:
                    entry[target_action] = path
                    warning = (
                        f"[warning] Using {source_action} preview in place of "
                        f"{target_action} for '{game_name}'."
                    )
                    print(warning)
                    self.status_label.config(text=warning)

        self.games = sorted(self.game_previews.keys())
        if self.edit_only:
            filtered_games = [
                game_name
                for game_name in self.games
                if normalize_game_name(game_name) in self.annotations
            ]
            self.game_previews = {name: self.game_previews[name] for name in filtered_games}
            self.games = filtered_games
        elif self.new_only:
            filtered_games = [
                game_name
                for game_name in self.games
                if normalize_game_name(game_name) not in self.annotations
            ]
            self.game_previews = {name: self.game_previews[name] for name in filtered_games}
            self.games = filtered_games

    def stop_animations(self):
        for job in self.animation_jobs.values():
            try:
                self.root.after_cancel(job)
            except Exception:
                pass
        self.animation_jobs.clear()
        self.animation_frames.clear()
        self.preview_labels.clear()

    def display_current_game(self):
        self.stop_animations()
        for child in self.preview_container.winfo_children():
            child.destroy()

        if not self.games or self.current_index >= len(self.games):
            self.root.destroy()
            return

        game_name = self.games[self.current_index]
        game_key = normalize_game_name(game_name)
        self.game_label.config(text=game_name)

        existing_row = self.annotations.get(game_key)
        blink_value = (
            existing_row.get("blinking", "").strip().upper() == "YES"
            if existing_row
            else False
        )
        delayed_value = (
            existing_row.get("delayed", "").strip().upper() == "YES"
            if existing_row
            else False
        )
        self.blink_var.set(blink_value)
        self.delayed_var.set(delayed_value)

        for tag, entry in self.tag_entries.items():
            entry.delete(0, "end")
            tag_value = existing_row.get(tag, "") if existing_row else ""
            default_value = TAG_DEFAULT_VALUES.get(tag, "")
            entry.insert(0, (tag_value or default_value))

        self.action_entries.clear()

        combined_actions = CONTROL_ACTIONS + PREVIEW_ONLY_ACTIONS
        for idx, action in enumerate(combined_actions):
            action_frame = Frame(
                self.preview_container, borderwidth=1, relief="solid", padx=8, pady=8
            )
            action_frame.grid(row=0, column=idx, padx=10, pady=5)

            Label(action_frame, text=action, font=("Arial", 12, "bold")).pack()

            Button(
                action_frame,
                text="Restart Preview",
                command=lambda act=action: self.reset_animation(act),
            ).pack(pady=(5, 2))

            preview_label = Label(action_frame)
            preview_label.pack(pady=(5, 5))
            self.preview_labels[action] = preview_label

            preview_path = self.game_previews[game_name].get(action)
            frames: list[ImageTk.PhotoImage] = []
            if preview_path and preview_path.exists():
                frames = self.preview_frame_cache.get(preview_path)
                if frames is None:
                    preview_image = Image.open(preview_path)
                    frames = [
                        ImageTk.PhotoImage(frame.copy())
                        for frame in ImageSequence.Iterator(preview_image)
                    ]
                    preview_image.close()
                    self.preview_frame_cache[preview_path] = frames
                self.animation_frames[action] = frames
                self.start_animation(action, preview_label, frames)
            else:
                preview_label.config(text="No preview", font=("Arial", 10, "italic"))

            if action in CONTROL_ACTIONS:
                entry = Entry(action_frame, width=18)
                entry.pack()
                default_text = ""
                if existing_row:
                    default_text = existing_row.get(action, "")
                if not default_text:
                    default_text = DEFAULT_ACTION_VALUES.get(action, "")
                entry.insert(0, default_text or "")
                self.action_entries[action] = entry

    def start_animation(
        self,
        action: str,
        preview_label: Label,
        frames: list[ImageTk.PhotoImage],
        start_index: int = 0,
    ):
        if not frames:
            return

        if action in self.animation_jobs:
            try:
                self.root.after_cancel(self.animation_jobs[action])
            except Exception:
                pass

        def _update(frame_index: int = 0):
            frame = frames[frame_index]
            preview_label.config(image=frame)
            preview_label.image = frame
            next_index = (frame_index + 1) % len(frames)
            job_id = self.root.after(100, _update, next_index)
            self.animation_jobs[action] = job_id

        _update(start_index % len(frames))

    def reset_animation(self, action: str):
        frames = self.animation_frames.get(action)
        preview_label = self.preview_labels.get(action)
        if not frames or not preview_label:
            return
        if action in self.animation_jobs:
            try:
                self.root.after_cancel(self.animation_jobs[action])
            except Exception:
                pass
            self.animation_jobs.pop(action, None)
        self.start_animation(action, preview_label, frames, start_index=0)

    def save_current_annotations(self):
        if not self.games or self.current_index >= len(self.games):
            return

        game_name = self.games[self.current_index]
        game_key = normalize_game_name(game_name)

        row = self.annotations.get(game_key)
        if row is None:
            row = {field: "" for field in self.fieldnames}
            row["game"] = game_key
            self.annotation_rows.append(row)
            self.annotations[game_key] = row
            row_was_new = True
        else:
            row_was_new = False
            for field in self.fieldnames:
                row.setdefault(field, "")

        changed = row_was_new

        def update_field(field_name: str, value: str):
            nonlocal changed
            current_value = row.get(field_name, "")
            if current_value != value:
                row[field_name] = value
                changed = True

        for action in CONTROL_ACTIONS:
            entry = self.action_entries.get(action)
            value = entry.get().strip() if entry else ""
            update_field(action, value)

        update_field("blinking", "YES" if self.blink_var.get() else "NO")
        update_field("delayed", "YES" if self.delayed_var.get() else "NO")

        for tag, entry in self.tag_entries.items():
            value = entry.get().strip() if entry else ""
            if not value:
                value = TAG_DEFAULT_VALUES.get(tag, "")
            update_field(tag, value)

        if changed:
            self.write_annotations()

    def next_game(self):
        if not self.games:
            self.root.destroy()
            return

        self.save_current_annotations()
        self.current_index += 1
        if self.current_index >= len(self.games):
            self.status_label.config(text="All games annotated. Exiting...")
            self.root.after(1500, self.root.destroy)
        else:
            self.status_label.config(
                text=f"Saved annotations for {self.games[self.current_index - 1]}."
            )
            self.display_current_game()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate control actions for RetroAct GIFs."
    )
    parser.add_argument(
        "--preview-dpath",
        default="data_generation/annotations/previews",
        help="Directory containing per-action GIF previews (default: %(default)s)",
    )
    parser.add_argument(
        "--annotation-fpath",
        default="data_generation/annotations/RetroAct_v0.1_control.csv",
        help="CSV file to read/write control annotations (default: %(default)s)",
    )
    parser.add_argument(
        "--edit",
        action="store_true",
        help="Only load games already present in the control annotation file.",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Only load games that are missing from the control annotation file.",
    )
    parser.add_argument(
        "--game",
        default=None,
        help="Annotate only the specified game (case-insensitive).",
    )
    args = parser.parse_args()

    if args.edit and args.new:
        parser.error("Specify only one of '--edit' or '--new'.")

    tool = AnnotationTool(
        args.preview_dpath,
        args.annotation_fpath,
        edit_only=args.edit,
        new_only=args.new,
        only_game=args.game,
    )
    tool.run()
