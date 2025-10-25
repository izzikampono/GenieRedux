import pandas as pd


class GameData:
    def __init__(
        self,
        annotation_fpath: str = "annotations/RetroAct_v0.1.csv",
        control_annotation_fpath: str = "annotations/RetroAct_v0.1_control_GenieRedux-G-50_sublist.csv",
        exclude_blinking: bool = False,
        exclude_delayed: bool = False,
        enable_sort=False,
    ):
        self.annotation_fpath = annotation_fpath
        # read the csv into a pandas dataframe
        self.df = pd.read_csv(annotation_fpath, delimiter=",")
        # separate df["tags"] using splitting by space into three columns: view, motion and genre
        # if "new" is in tags remove from the data frame
        # sort by game
        if enable_sort:
            self.df = self.df.sort_values(by="game")
        self.df = self.df[~self.df["tags"].str.contains("new")]
        self.df[["view", "motion", "genre"]] = self.df["tags"].str.split(
            " ", expand=True
        )[[0, 1, 2]]
        self.df[["platform"]] = self.df["game"].str.split("-", expand=True)[[1]]

        if control_annotation_fpath is not None:
            self.enable_controls = True
            self.df["game_upper"] = self.df["game"]
            self.df["game"] = self.df["game"].str.lower()
            self.df_controls = pd.read_csv(control_annotation_fpath, delimiter=",")
            if enable_sort:
                self.df_controls = self.df_controls.sort_values(by="game")

            # join the two dataframes on the game column but the game column in df is with some capitilized letters while in df_controls it is all lower case. After joining keep the capitalization
            # copy the game column in df to a new one called game_upper

            self.df = self.df.merge(
                self.df_controls,
                on="game",
                how="left",
                suffixes=("", "_controls"),
            )
            self.df["game"] = self.df["game_upper"]
            self.df = self.df.drop(columns=["game_upper"])

            if exclude_blinking and "blinking" in self.df.columns:
                self.df = self.df[
                    ~self.df["blinking"].fillna("").str.upper().eq("YES")
                ]
            if exclude_delayed and "delayed" in self.df.columns:
                self.df = self.df[
                    ~self.df["delayed"].fillna("").str.upper().eq("YES")
                ]
        else:
            self.enable_controls = False

    def filter(self, action_map):
        # clean actions
        if self.enable_controls:
            # remove all entries with ACTION_PRIMARY different than jump
            self.df = self.df[self.df["ACTION_PRIMARY"] == "jump"]
            # remove all entries with DOWN different than down crouch, climb down or  climb contained in the entry
            self.df = self.df[self.df["DOWN"].str.contains("none|crouch|climb")]
            # remove all entries with UP different than climb or none
            self.df = self.df[self.df["UP"].str.contains("climb|none")]
            # remove all entries with LEFT different than left
            self.df = self.df[self.df["LEFT"] == "left"]
            # remove all entries with RIGHT different than right
            self.df = self.df[self.df["RIGHT"] == "right"]
            # remove all entries with transition equal to 1
            # self.df = self.df[self.df["transition"] == 0]

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
