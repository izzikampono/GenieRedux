import pandas as pd


class GameData:
    def __init__(
        self,
        annotation_fpath: str = "annotations/RetroAct_v1.4.csv",
        exclude_blinking: bool = False,
        exclude_delayed: bool = False,
        enable_sort=False,
    ):
        self.annotation_fpath = annotation_fpath
        self.df = pd.read_csv(annotation_fpath, delimiter=",")

        if enable_sort:
            self.df = self.df.sort_values(by="game")

        required_cols = {"view", "motion", "genre"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Annotation file '{annotation_fpath}' missing required columns: {sorted(missing)}"
            )

        mask_new = (
            self.df[["view", "motion", "genre"]]
            .astype(str)
            .apply(lambda s: s.str.contains("new", case=False, na=False))
            .any(axis=1)
        )
        self.df = self.df[~mask_new]
        self.df[["platform"]] = self.df["game"].str.split("-", expand=True)[[1]]

        control_columns = [
            "RIGHT",
            "LEFT",
            "UP",
            "DOWN",
            "ACTION_PRIMARY",
            "ACTION_SECONDARY",
            "blinking",
            "delayed",
        ]

        if exclude_blinking:
            self.df = self.df[~self.df["blinking"].fillna("").str.upper().eq("YES")]
        if exclude_delayed:
            self.df = self.df[~self.df["delayed"].fillna("").str.upper().eq("YES")]

    def filter(self, conditions):
        if not conditions:
            return

        action_columns = [
            col
            for col in [
                "RIGHT",
                "LEFT",
                "UP",
                "DOWN",
                "ACTION_PRIMARY",
                "ACTION_SECONDARY",
            ]
            if col in self.df.columns
        ]

        if not action_columns:
            return

        actions = self.df[action_columns].fillna("").astype(str)
        actions = actions.applymap(lambda s: s.strip().lower())

        combined_mask = None
        for pattern in conditions:
            if not pattern:
                continue

            clause = pattern.strip().lower()
            if not clause:
                continue

            if "|" in clause:
                options = {opt.strip() for opt in clause.split("|") if opt.strip()}

                def row_match(row):
                    for value in row:
                        val = value.strip()
                        if val in options:
                            return True
                    return False

                mask = actions.apply(row_match, axis=1)
            else:
                tokens = [tok for tok in clause.split() if tok]
                if not tokens:
                    continue
                token_count = len(tokens)
                token_set = set(tokens)

                def row_match(row):
                    for value in row:
                        val_tokens = [tok for tok in value.strip().split() if tok]
                        if len(val_tokens) != token_count:
                            continue
                        if set(val_tokens) == token_set:
                            return True
                    return False

                mask = actions.apply(row_match, axis=1)

            combined_mask = mask if combined_mask is None else (combined_mask & mask)

        if combined_mask is not None:
            self.df = self.df[combined_mask]

    def query(self, view=None, motion=None, genre=None, game=None, platform=None):
        df = self.df
        if view is not None:
            df = df[df["view"].isin(view)]
        if motion is not None:
            df = df[df["motion"].isin(motion)]
        if genre is not None:
            df = df[df["genre"].isin(genre)]
        if game is not None:
            df = df[df["game"].isin(game)]
        if platform is not None:
            df = df[df["platform"].isin(platform)]
        return df["game"].tolist()
