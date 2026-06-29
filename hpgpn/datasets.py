"""A PyTorch dataset containing all passes."""
import itertools
import math
import os
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import torch
from rich.progress import track
from torch.utils.data import Dataset

from hpgpn import features, labels
from hpgpn.config import logger as log


class PassesDataset(Dataset):
    """A dataset containing passes.

    Parameters
    ----------
    xfns : dict(str or callable -> list(str))
        The feature generators and columns to use.
    yfns : list(str or callable)
        The label generators.
    transform : Callable, optional
        A function/transform that takes a sample and returns a transformed
        version of it.
    path : Path
        The path to the directory where pre-computed features are stored. By
        default all features and labels are computed on the fly, but
        pre-computing them will speed up training significantly.
    load_cached : bool, default: True
        Whether to attempt to load the dataset from disk.
    """

    def __init__(
        self,
        xfns: Union[List, Dict[Union[str, Callable], Optional[List]]],
        yfns: List[Union[str, Callable]],
        transform: Optional[Callable] = None,
        path: Optional[os.PathLike[str]] = None,
        load_cached: bool = True,
    ):
        # Check requested features and labels
        self.xfns = self._parse_xfns(xfns)
        self.yfns = self._parse_yfns(yfns)
        self.transform = transform

        # Try to load the dataset
        self.store = Path(path) if path is not None else None
        self._features = None
        self._labels = None
        if load_cached:
            if self.store is None:
                raise ValueError("No path to cached dataset provided.")
            try:
                log.info("Loading dataset from %s", self.store)
                if len(self.xfns):
                    self._features = pd.concat(
                        [
                            pd.read_parquet(self.store / f"x_{xfn.__name__}.parquet")[cols]
                            for xfn, cols in self.xfns.items()
                        ],
                        axis=1,
                    )
                if len(self.yfns):
                    self._labels = pd.concat(
                        [
                            pd.read_parquet(self.store / f"y_{yfn.__name__}.parquet")
                            for yfn in self.yfns
                        ],
                        axis=1,
                    )
            except FileNotFoundError:
                log.error(
                    "No complete dataset found at %s. Run 'create' to create it.", self.store
                )

    @staticmethod
    def _parse_xfns(
        xfns: Union[List, Dict[Union[str, Callable], Optional[List]]]
    ) -> Dict[Callable, Optional[List]]:
        parsed_xfns = {}
        if isinstance(xfns, list):
            xfns = {xfn: None for xfn in xfns}
        for xfn, cols in xfns.items():
            parsed_xfn = xfn
            parsed_cols = cols
            if isinstance(parsed_xfn, str):
                try:
                    parsed_xfn = getattr(features, parsed_xfn)
                except AttributeError:
                    raise ValueError(f"No feature function found that matches '{parsed_xfn}'.")
            if parsed_cols is None:
                parsed_cols = features.feature_column_names([parsed_xfn])
            parsed_xfns[parsed_xfn] = parsed_cols
        return parsed_xfns

    @staticmethod
    def _parse_yfns(yfns: List[Union[str, Callable]]) -> List[Callable]:
        parsed_yfns = []
        for yfn in yfns:
            if isinstance(yfn, str):
                try:
                    parsed_yfns.append(getattr(labels, yfn))
                except AttributeError:
                    raise ValueError(f"No labeling function found that matches '{yfn}'.")
            else:
                parsed_yfns.append(yfn)
        return parsed_yfns

    @staticmethod
    def actionfilter(actions: pd.DataFrame) -> pd.Series:
        is_pass = actions.type_name.isin(["pass", "cross"])
        by_foot = actions.bodypart_name.str.contains("foot")
        in_freeze_frame = actions.in_visible_area_360
        return is_pass & by_foot & in_freeze_frame

    def create(self, db, filters: Optional[List[Dict]] = None) -> None:
        """Create the dataset.

        Parameters
        ----------
        db : Database
            The database with raw data.
        filters : list(dict), optional
            A list of filters used to select a sample of the dataset. Each filter
            is a dictionary with (a subset of) the following keys:
                - "competition_id": int
                - "season_id": int
                - "game_id": int
            For example, [{"competition_id": 55, "season_id": 43}] will select
            all passes from EURO2020. If no filters are provided, all available
            passes are returned.
        """
        # Create directory
        if self.store is not None:
            self.store.mkdir(parents=True, exist_ok=True)

        # Select games to include
        self.games_idx = list(
            itertools.chain.from_iterable(
                [x["game_id"]] if "game_id" in x else db.games(**x).index.tolist() for x in filters
            )
            if filters is not None
            else db.games().index
        )

        # Compute features for each pass
        if len(self.xfns):
            df_features = []
            for xfn, _ in self.xfns.items():
                df_features_xfn = []
                for game_id in track(
                    self.games_idx, description=f"Computing {xfn.__name__} feature"
                ):
                    df_features_xfn.append(
                        features.get_features(
                            db,
                            game_id=game_id,
                            xfns=[xfn],
                            nb_prev_actions=3,
                            actionfilter=PassesDataset.actionfilter,
                        )
                    )
                df_features_xfn = pd.concat(df_features_xfn)
                if self.store is not None:
                    assert self.store is not None
                    df_features_xfn.to_parquet(self.store / f"x_{xfn.__name__}.parquet")
                df_features.append(df_features_xfn)
            self._features = pd.concat(df_features, axis=1)

        # Compute labels for each pass
        if len(self.yfns):
            df_labels = []
            for yfn in self.yfns:
                df_labels_yfn = []
                for game_id in track(
                    self.games_idx, description=f"Computing {yfn.__name__} label"
                ):
                    df_labels_yfn.append(
                        labels.get_labels(
                            db,
                            game_id=game_id,
                            yfns=[yfn],
                            actionfilter=PassesDataset.actionfilter,
                        )
                    )
                df_labels_yfn = pd.concat(df_labels_yfn)
                if self.store is not None:
                    df_labels_yfn.to_parquet(self.store / f"y_{yfn.__name__}.parquet")
                df_labels.append(df_labels_yfn)
            self._labels = pd.concat(df_labels, axis=1)

    @property
    def features(self):
        if self._features is None:
            assert self._labels is not None, "First, create the dataset."
            return pd.DataFrame(index=self._labels.index)
        return self._features

    @property
    def labels(self):
        if self._labels is None:
            assert self._features is not None, "First, create the dataset."
            return pd.DataFrame(index=self._features.index)
        return self._labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.features is not None:
            return len(self.features)
        if self.labels is not None:
            return len(self.labels)
        return 0

    def __getitem__(self, idx: int) -> Dict:
        """Return a sample from the dataset at the given index.

        Parameters
        ----------
        idx : int
            The index of the sample to return.

        Returns
        -------
        sample : (dict, dict)
            A dictionary containing the sample and target.
        """
        game_id, action_id = None, None
        sample_features = {}
        if self.features is not None:
            sample_features = self.features.iloc[idx].to_dict()
            game_id, action_id = self.features.iloc[idx].name
        sample_target = {}
        if self.labels is not None:
            sample_target = self.labels.iloc[idx].to_dict()
            game_id, action_id = self.labels.iloc[idx].name
        # freezedf = self.db.freeze_frame(game_id=sample_idx[0], action_id=sample_idx[1], ltr=True)
        sample = {
            "game_id": game_id,
            "action_id": action_id,
            **sample_features,
            **sample_target,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def apply_overrides(
        self,
        db,
        overrides: pd.DataFrame,
    ):
        """Override a set of action attributes before applying the feature generators.

        Parameters
        ----------
        db : Database
            The database with raw data.
        overrides : pd.DataFrame
            A dataframe with action attributes that override the values in the
            database. The dataframe should be indexed by game_id and action_id.

        Returns
        -------
        pd.DataFrame
            A dataframe with the features computed from the modified actions.
        """
        # select features that are not affected by overrides
        xfns_fixed = [
            fn
            for fn in self.xfns
            if not any(field in fn.required_fields for field in overrides.columns)
        ]
        xfns_fixed_cols = list(itertools.chain(*[self.xfns[fn] for fn in xfns_fixed]))
        df_fixed_features = self.features.loc[:, xfns_fixed_cols]

        games_idx = self.features.index.get_level_values("game_id").unique()
        df_simulated_features = []
        for xfn, cols in self.xfns.items():
            # skip fixed features
            if xfn in xfns_fixed:
                continue

            # simulate other features
            df_features_xfn = []
            for game_id in track(games_idx, description=f"Computing {xfn.__name__} feature"):
                df_features_xfn.append(
                    features.get_features(
                        db,
                        game_id=game_id,
                        xfns=[xfn],
                        nb_prev_actions=3,
                        actionfilter=PassesDataset.actionfilter,
                        overrides=overrides,
                    )[cols]
                )
            df_simulated_features.append(pd.concat(df_features_xfn))

        modified_dataset = copy(self)
        modified_dataset._features = pd.concat(
            [df_fixed_features] + df_simulated_features, axis=1
        )[self.features.columns]
        return modified_dataset


class CompletedPassesDataset(PassesDataset):
    """A dataset containing only completed passes."""

    def __init__(
        self,
        path: os.PathLike[str],
        xfns: Union[List, Dict[Union[str, Callable], Optional[List]]],
        yfns: List[Union[str, Callable]],
        transform: Optional[Callable] = None,
    ):
        super().__init__(path, xfns, yfns + ["success"], transform)

    @property
    def features(self):
        if self._features is None:
            return pd.DataFrame(index=self.labels.index)
        return self._features.loc[self.labels.index]

    @property
    def labels(self):
        assert self._labels is not None
        df_labels = self._labels[self._labels.success]
        df_labels.drop("success", axis=1, inplace=True)
        return df_labels


class FailedPassesDataset(PassesDataset):
    """A dataset containing only failed passes."""

    def __init__(
        self,
        path: os.PathLike[str],
        xfns: Union[List, Dict[Union[str, Callable], Optional[List]]],
        yfns: List[Union[str, Callable]],
        transform: Optional[Callable] = None,
    ):
        super().__init__(path, xfns, yfns + ["success"], transform)

    @property
    def features(self):
        if self._features is None:
            return pd.DataFrame(index=self.labels.index)
        return self._features.loc[self.labels.index]

    @property
    def labels(self):
        assert self._labels is not None
        df_labels = self._labels[~self._labels.success]
        df_labels.drop("success", axis=1, inplace=True)
        return df_labels


@dataclass
class GraphSample:
    """A graph sample for pass receiver ranking."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    candidate_mask: torch.Tensor
    target_index: torch.Tensor
    game_id: int
    action_id: int
    pass_option_ids: List[int]


def _node_feat(x: float, y: float, node_type: str) -> List[float]:
    dx_goal, dy_goal = 105.0 - x, 34.0 - y
    return [
        x / 105.0,
        y / 68.0,
        (dx_goal**2 + dy_goal**2) ** 0.5 / 105.0,
        math.atan2(abs(dy_goal), abs(dx_goal)) / math.pi,
        float(node_type == "passer"),
        float(node_type == "candidate"),
        float(node_type == "defender"),
    ]


def _candidate_extra_feat(candidate_row: pd.Series) -> List[float]:
    return [
        float(candidate_row["distance"]) / 105.0,
        float(candidate_row["angle"]) / math.pi,
        float(candidate_row["origin_angle_to_goal"]) / math.pi,
        float(candidate_row["destination_angle_to_goal"]) / math.pi,
        float(candidate_row["destination_distance_defender"]) / 20.0,
        float(candidate_row["pass_distance_defender"]) / 20.0,
    ]


def _pass_edge_feat(candidate_row: pd.Series) -> List[float]:
    return _candidate_extra_feat(candidate_row)


def _geom_edge_feat(src_x: float, src_y: float, dst_x: float, dst_y: float) -> List[float]:
    dx = (dst_x - src_x) / 105.0
    dy = (dst_y - src_y) / 68.0
    dist = math.sqrt(dx**2 + dy**2)
    return [dx, dy, dist, 0.0, 0.0, 0.0]


def build_pass_graph(
    game_id: int,
    action_id: int,
    action_row: pd.Series,
    candidate_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> GraphSample:
    """Build a relational graph for a single pass action."""
    freeze_frame = action_row.get("freeze_frame_360_a0", action_row.get("freeze_frame_360"))
    if freeze_frame is None:
        freeze_frame = []

    start_x = float(action_row.get("start_x_a0", action_row.get("start_x")))
    start_y = float(action_row.get("start_y_a0", action_row.get("start_y")))

    defenders = [p for p in freeze_frame if not p["teammate"]]

    if isinstance(candidate_df, pd.Series):
        candidate_df = candidate_df.to_frame().T
    if isinstance(label_df, pd.Series):
        label_df = label_df.to_frame().T

    candidate_df = candidate_df.sort_index()
    label_df = label_df.sort_index()

    if len(candidate_df) == 0:
        raise ValueError("No candidate receivers found for graph construction.")

    node_features = []

    passer_feat = _node_feat(start_x, start_y, "passer") + [0.0] * 6
    node_features.append(passer_feat)

    for _, cand_row in candidate_df.iterrows():
        cx = float(cand_row["destination_x"])
        cy = float(cand_row["destination_y"])
        base_feat = _node_feat(cx, cy, "candidate")
        extra_feat = _candidate_extra_feat(cand_row)
        node_features.append(base_feat + extra_feat)

    for defender in defenders:
        def_feat = _node_feat(float(defender["x"]), float(defender["y"]), "defender")
        node_features.append(def_feat + [0.0] * 6)

    n_cand = len(candidate_df)
    n_def = len(defenders)
    n_nodes = 1 + n_cand + n_def

    candidate_mask = torch.zeros(n_nodes, dtype=torch.bool)
    candidate_mask[1 : 1 + n_cand] = True

    ordered_labels = label_df["receiver"].tolist()
    target_index = torch.tensor(ordered_labels.index(True), dtype=torch.long)

    edges: List[List[int]] = []
    edge_features: List[List[float]] = []
    cand_nodes = list(range(1, 1 + n_cand))
    def_nodes = list(range(1 + n_cand, n_nodes))

    for local_idx, (_, cand_row) in enumerate(candidate_df.iterrows()):
        c = cand_nodes[local_idx]
        edges.append([0, c])
        edge_features.append(_pass_edge_feat(cand_row))

    for local_idx, (_, cand_row) in enumerate(candidate_df.iterrows()):
        c = cand_nodes[local_idx]
        cx = float(cand_row["destination_x"])
        cy = float(cand_row["destination_y"])
        for def_offset, defender in enumerate(defenders):
            d = def_nodes[def_offset]
            dx = float(defender["x"])
            dy = float(defender["y"])
            edges.append([c, d])
            edge_features.append(_geom_edge_feat(cx, cy, dx, dy))
            edges.append([d, c])
            edge_features.append(_geom_edge_feat(dx, dy, cx, cy))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_tensor = torch.tensor(edge_features, dtype=torch.float32)

    return GraphSample(
        node_features=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_features=edge_tensor,
        candidate_mask=candidate_mask,
        target_index=target_index,
        game_id=game_id,
        action_id=action_id,
        pass_option_ids=candidate_df.index.tolist(),
    )

def build_pass_graph_with_cand_cand(
    game_id: int,
    action_id: int,
    action_row: pd.Series,
    candidate_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> GraphSample:
    freeze_frame = action_row.get("freeze_frame_360_a0", action_row.get("freeze_frame_360"))
    if freeze_frame is None:
        freeze_frame = []

    start_x = float(action_row.get("start_x_a0", action_row.get("start_x")))
    start_y = float(action_row.get("start_y_a0", action_row.get("start_y")))

    defenders = [p for p in freeze_frame if not p["teammate"]]

    if isinstance(candidate_df, pd.Series):
        candidate_df = candidate_df.to_frame().T
    if isinstance(label_df, pd.Series):
        label_df = label_df.to_frame().T

    candidate_df = candidate_df.sort_index()
    label_df = label_df.sort_index()

    if len(candidate_df) == 0:
        raise ValueError("No candidate receivers found for graph construction.")

    node_features = []

    # passer
    passer_feat = _node_feat(start_x, start_y, "passer") + [0.0] * 6
    node_features.append(passer_feat)

    # candidates
    cand_xy = []
    for _, cand_row in candidate_df.iterrows():
        cx = float(cand_row["destination_x"])
        cy = float(cand_row["destination_y"])
        cand_xy.append((cx, cy))
        node_features.append(_node_feat(cx, cy, "candidate") + _candidate_extra_feat(cand_row))

    # defenders
    for defender in defenders:
        dx = float(defender["x"])
        dy = float(defender["y"])
        def_feat = _node_feat(dx, dy, "defender") + [0.0] * 6
        node_features.append(def_feat)

    n_cand = len(candidate_df)
    n_def = len(defenders)
    n_nodes = 1 + n_cand + n_def

    candidate_mask = torch.zeros(n_nodes, dtype=torch.bool)
    candidate_mask[1:1 + n_cand] = True

    ordered_labels = label_df["receiver"].tolist()
    target_index = torch.tensor(ordered_labels.index(True), dtype=torch.long)

    cand_nodes = list(range(1, 1 + n_cand))
    def_nodes = list(range(1 + n_cand, n_nodes))

    edges = []
    edge_features = []

    # passer -> candidate
    for local_idx, (_, cand_row) in enumerate(candidate_df.iterrows()):
        c = cand_nodes[local_idx]
        edges.append([0, c])
        edge_features.append(_pass_edge_feat(cand_row))

    # candidate <-> defender
    for local_idx, (_, cand_row) in enumerate(candidate_df.iterrows()):
        c = cand_nodes[local_idx]
        cx = float(cand_row["destination_x"])
        cy = float(cand_row["destination_y"])

        for def_offset, defender in enumerate(defenders):
            d = def_nodes[def_offset]
            dx = float(defender["x"])
            dy = float(defender["y"])

            edges.append([c, d])
            edge_features.append(_geom_edge_feat(cx, cy, dx, dy))

            edges.append([d, c])
            edge_features.append(_geom_edge_feat(dx, dy, cx, cy))

    # NEW: candidate <-> candidate
    for i in range(n_cand):
        for j in range(n_cand):
            if i == j:
                continue

            src = cand_nodes[i]
            dst = cand_nodes[j]
            src_x, src_y = cand_xy[i]
            dst_x, dst_y = cand_xy[j]

            edges.append([src, dst])
            edge_features.append(_geom_edge_feat(src_x, src_y, dst_x, dst_y))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_tensor = torch.tensor(edge_features, dtype=torch.float32)

    return GraphSample(
        node_features=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_features=edge_tensor,
        candidate_mask=candidate_mask,
        target_index=target_index,
        game_id=game_id,
        action_id=action_id,
        pass_option_ids=candidate_df.index.tolist(),
    )


class GraphPassSelectionDataset(Dataset):
    """Graph dataset for pass-selection receiver ranking."""

    required_files = (
        "x_pass_options.parquet",
        "x_freeze_frame_360.parquet",
        "x_startlocation.parquet",
        "y_receiver.parquet",
    )

    def __init__(self, path: os.PathLike[str]):
        self.store = Path(path)
        missing = [name for name in self.required_files if not (self.store / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing graph dataset files in {self.store}: {', '.join(missing)}"
            )
        """
        self.df_candidates = pd.read_parquet(self.store / "x_pass_options.parquet")
        self.df_actions = pd.concat(
            [
                pd.read_parquet(self.store / "x_freeze_frame_360.parquet")[
                    ["freeze_frame_360_a0"]
                ],
                pd.read_parquet(self.store / "x_startlocation.parquet")[
                    ["start_x_a0", "start_y_a0"]
                ],
            ],
            axis=1,
        )
        self.df_labels = pd.read_parquet(self.store / "y_receiver.parquet")
        self.keys = list(self.df_candidates.index.droplevel("pass_option_id").unique())"""
        self.df_candidates = (
            pd.read_parquet(self.store / "x_pass_options.parquet")
            .sort_index()
        )

        self.df_actions = pd.concat(
            [
                pd.read_parquet(self.store / "x_freeze_frame_360.parquet")[
                    ["freeze_frame_360_a0"]
                ],
                pd.read_parquet(self.store / "x_startlocation.parquet")[
                    ["start_x_a0", "start_y_a0"]
                ],
            ],
            axis=1,
        ).sort_index()

        self.df_labels = (
            pd.read_parquet(self.store / "y_receiver.parquet")
            .sort_index()
        )

        self.keys = list(
            self.df_candidates.index.droplevel("pass_option_id").unique()
        )

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> GraphSample:
        game_id, action_id = self.keys[idx]
        action_row = self.df_actions.loc[(game_id, action_id)]
        candidate_df = self.df_candidates.loc[(game_id, action_id)]
        label_df = self.df_labels.loc[(game_id, action_id)]
        return build_pass_graph(game_id, action_id, action_row, candidate_df, label_df)
    
class GraphPassSelectionDatasetCandCand(Dataset):
    required_files = (
        "x_pass_options.parquet",
        "x_freeze_frame_360.parquet",
        "x_startlocation.parquet",
        "y_receiver.parquet",
    )

    def __init__(self, path):
        self.store = Path(path)

        missing = [name for name in self.required_files if not (self.store / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing graph dataset files in {self.store}: {', '.join(missing)}"
            )

        self.df_candidates = pd.read_parquet(self.store / "x_pass_options.parquet")
        self.df_actions = pd.concat(
            [
                pd.read_parquet(self.store / "x_freeze_frame_360.parquet")[["freeze_frame_360_a0"]],
                pd.read_parquet(self.store / "x_startlocation.parquet")[["start_x_a0", "start_y_a0"]],
            ],
            axis=1,
        )
        self.df_labels = pd.read_parquet(self.store / "y_receiver.parquet")
        self.keys = list(self.df_candidates.index.droplevel("pass_option_id").unique())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        game_id, action_id = self.keys[idx]
        action_row = self.df_actions.loc[(game_id, action_id)]
        candidate_df = self.df_candidates.loc[(game_id, action_id)]
        label_df = self.df_labels.loc[(game_id, action_id)]

        return build_pass_graph_with_cand_cand(
            game_id=game_id,
            action_id=action_id,
            action_row=action_row,
            candidate_df=candidate_df,
            label_df=label_df,
        )



# Multigraph GNN22,23,24,25
@dataclass
class SimpleGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    candidate_mask: torch.Tensor


@dataclass
class MultiGraphSample:
    pass_graph: SimpleGraph
    pressure_graph: SimpleGraph
    target_index: torch.Tensor
    game_id: int
    action_id: int

# 基础 node 特征
def _base_feat(x, y, node_type):
    dx_goal, dy_goal = 105.0 - x, 34.0 - y
    return [
        x / 105.0,
        y / 68.0,
        (dx_goal ** 2 + dy_goal ** 2) ** 0.5 / 105.0,
        math.atan2(abs(dy_goal), abs(dx_goal)) / math.pi,
        float(node_type == "passer"),
        float(node_type == "candidate"),
        float(node_type == "defender"),
    ]

# candidate pass 特征
def _candidate_pass_feat(cand_row):
    return [
        float(cand_row["distance"]) / 105.0,
        float(cand_row["angle"]) / math.pi,
        float(cand_row["origin_angle_to_goal"]) / math.pi,
        float(cand_row["destination_angle_to_goal"]) / math.pi,
        float(cand_row["destination_distance_defender"]) / 20.0,
        float(cand_row["pass_distance_defender"]) / 20.0,
    ]

class TwoGraphPassSelectionDataset(Dataset):
    def __init__(self, path):
        self.store = Path(path)

        self.df_candidates = pd.read_parquet(self.store / "x_pass_options.parquet")
        self.df_freeze = pd.read_parquet(self.store / "x_freeze_frame_360.parquet")
        self.df_start = pd.read_parquet(self.store / "x_startlocation.parquet")
        self.df_labels = pd.read_parquet(self.store / "y_receiver.parquet")

        self.df_actions = self.df_freeze.join(self.df_start, how="inner")
        self.keys = list(
            self.df_candidates.index.droplevel("pass_option_id").unique()
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        game_id, action_id = self.keys[idx]

        action_row = self.df_actions.loc[(game_id, action_id)]
        candidate_df = self.df_candidates.loc[(game_id, action_id)]
        label_df = self.df_labels.loc[(game_id, action_id)]

        if isinstance(candidate_df, pd.Series):
            candidate_df = candidate_df.to_frame().T
        if isinstance(label_df, pd.Series):
            label_df = label_df.to_frame().T

        candidate_df = candidate_df.sort_index()
        label_df = label_df.sort_index()

        freeze_frame = action_row["freeze_frame_360_a0"]
        if freeze_frame is None:
            freeze_frame = []

        defenders = [p for p in freeze_frame if not p["teammate"]]

        start_x = float(action_row["start_x_a0"])
        start_y = float(action_row["start_y_a0"])

        # --------------------------
        # Pass Graph
        # nodes = [passer] + [candidates]
        # --------------------------
        pass_node_features = []

        passer_feat = _base_feat(start_x, start_y, "passer") + [0.0] * 6
        pass_node_features.append(passer_feat)

        for _, row in candidate_df.iterrows():
            cx = float(row["destination_x"])
            cy = float(row["destination_y"])
            cand_feat = _base_feat(cx, cy, "candidate") + _candidate_pass_feat(row)
            pass_node_features.append(cand_feat)

        n_cand = len(candidate_df)
        pass_edges = [[0, c] for c in range(1, 1 + n_cand)]

        pass_candidate_mask = torch.zeros(1 + n_cand, dtype=torch.bool)
        pass_candidate_mask[1:] = True

        pass_graph = SimpleGraph(
            node_features=torch.tensor(pass_node_features, dtype=torch.float32),
            edge_index=torch.tensor(pass_edges, dtype=torch.long).t().contiguous(),
            candidate_mask=pass_candidate_mask,
        )

        # --------------------------
        # Pressure Graph
        # nodes = [candidates] + [defenders]
        # --------------------------
        pressure_node_features = []

        for _, row in candidate_df.iterrows():
            cx = float(row["destination_x"])
            cy = float(row["destination_y"])
            cand_feat = _base_feat(cx, cy, "candidate")
            pressure_node_features.append(cand_feat)

        for d in defenders:
            dx = float(d["x"])
            dy = float(d["y"])
            def_feat = _base_feat(dx, dy, "defender")
            pressure_node_features.append(def_feat)

        pressure_edges = []
        def_start = n_cand

        for c in range(n_cand):
            for d_idx in range(len(defenders)):
                d = def_start + d_idx
                pressure_edges.append([c, d])
                pressure_edges.append([d, c])

        if len(pressure_edges) == 0:
            # avoid empty edge_index
            pressure_edges = [[0, 0]]

        pressure_candidate_mask = torch.zeros(n_cand + len(defenders), dtype=torch.bool)
        pressure_candidate_mask[:n_cand] = True

        pressure_graph = SimpleGraph(
            node_features=torch.tensor(pressure_node_features, dtype=torch.float32),
            edge_index=torch.tensor(pressure_edges, dtype=torch.long).t().contiguous(),
            candidate_mask=pressure_candidate_mask,
        )

        target_index = torch.tensor(
            label_df["receiver"].tolist().index(True),
            dtype=torch.long,
        )

        return MultiGraphSample(
            pass_graph=pass_graph,
            pressure_graph=pressure_graph,
            target_index=target_index,
            game_id=int(game_id),
            action_id=int(action_id),
        )

# heterognn
@dataclass
class HeteroGraphSample:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    node_types: torch.Tensor        # [N], 0=passer, 1=candidate, 2=defender
    edge_types: torch.Tensor        # [E], 0=pc, 1=cd, 2=dc
    candidate_mask: torch.Tensor
    target_index: torch.Tensor
    game_id: int
    action_id: int

def _node_feat(x: float, y: float, node_type: str):
    dx_goal, dy_goal = 105.0 - x, 34.0 - y
    return [
        x / 105.0,
        y / 68.0,
        (dx_goal ** 2 + dy_goal ** 2) ** 0.5 / 105.0,
        math.atan2(abs(dy_goal), abs(dx_goal)) / math.pi,
        float(node_type == "passer"),
        float(node_type == "candidate"),
        float(node_type == "defender"),
    ]


def _candidate_extra_feat(candidate_row: pd.Series):
    return [
        float(candidate_row["distance"]) / 105.0,
        float(candidate_row["angle"]) / math.pi,
        float(candidate_row["origin_angle_to_goal"]) / math.pi,
        float(candidate_row["destination_angle_to_goal"]) / math.pi,
        float(candidate_row["destination_distance_defender"]) / 20.0,
        float(candidate_row["pass_distance_defender"]) / 20.0,
    ]


def _pass_edge_feat(candidate_row: pd.Series):
    return _candidate_extra_feat(candidate_row)


def _geom_edge_feat(src_x: float, src_y: float, dst_x: float, dst_y: float):
    dx = (dst_x - src_x) / 105.0
    dy = (dst_y - src_y) / 68.0
    dist = math.sqrt(dx ** 2 + dy ** 2)
    return [dx, dy, dist, 0.0, 0.0, 0.0]

def build_pass_graph_hetero(
    game_id: int,
    action_id: int,
    action_row: pd.Series,
    candidate_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> HeteroGraphSample:
    freeze_frame = action_row["freeze_frame_360_a0"]
    if freeze_frame is None:
        freeze_frame = []

    start_x = float(action_row["start_x_a0"])
    start_y = float(action_row["start_y_a0"])

    defenders = [p for p in freeze_frame if not p["teammate"]]

    if isinstance(candidate_df, pd.Series):
        candidate_df = candidate_df.to_frame().T
    if isinstance(label_df, pd.Series):
        label_df = label_df.to_frame().T

    candidate_df = candidate_df.sort_index()
    label_df = label_df.sort_index()

    if len(candidate_df) == 0:
        raise ValueError("No candidate receivers found.")

    node_features = []
    node_types = []

    # node type ids
    PASSER = 0
    CANDIDATE = 1
    DEFENDER = 2

    # edge type ids
    PC = 0   # passer -> candidate
    CD = 1   # candidate -> defender
    DC = 2   # defender -> candidate

    # passer node
    passer_feat = _node_feat(start_x, start_y, "passer") + [0.0] * 6
    node_features.append(passer_feat)
    node_types.append(PASSER)

    # candidate nodes
    for _, cand_row in candidate_df.iterrows():
        cx = float(cand_row["destination_x"])
        cy = float(cand_row["destination_y"])
        node_features.append(_node_feat(cx, cy, "candidate") + _candidate_extra_feat(cand_row))
        node_types.append(CANDIDATE)

    # defender nodes
    for defender in defenders:
        dx = float(defender["x"])
        dy = float(defender["y"])
        node_features.append(_node_feat(dx, dy, "defender") + [0.0] * 6)
        node_types.append(DEFENDER)

    n_cand = len(candidate_df)
    n_def = len(defenders)
    n_nodes = 1 + n_cand + n_def

    candidate_mask = torch.zeros(n_nodes, dtype=torch.bool)
    candidate_mask[1:1 + n_cand] = True

    ordered_labels = label_df["receiver"].tolist()
    target_index = torch.tensor(ordered_labels.index(True), dtype=torch.long)

    edges = []
    edge_features = []
    edge_types = []

    cand_nodes = list(range(1, 1 + n_cand))
    def_nodes = list(range(1 + n_cand, n_nodes))

    # passer -> candidate
    for local_idx, (_, cand_row) in enumerate(candidate_df.iterrows()):
        c = cand_nodes[local_idx]
        edges.append([0, c])
        edge_features.append(_pass_edge_feat(cand_row))
        edge_types.append(PC)

    # candidate <-> defender
    for local_idx, (_, cand_row) in enumerate(candidate_df.iterrows()):
        c = cand_nodes[local_idx]
        cx = float(cand_row["destination_x"])
        cy = float(cand_row["destination_y"])

        for def_offset, defender in enumerate(defenders):
            d = def_nodes[def_offset]
            dx = float(defender["x"])
            dy = float(defender["y"])

            edges.append([c, d])
            edge_features.append(_geom_edge_feat(cx, cy, dx, dy))
            edge_types.append(CD)

            edges.append([d, c])
            edge_features.append(_geom_edge_feat(dx, dy, cx, cy))
            edge_types.append(DC)

    return HeteroGraphSample(
        node_features=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_features=torch.tensor(edge_features, dtype=torch.float32),
        node_types=torch.tensor(node_types, dtype=torch.long),
        edge_types=torch.tensor(edge_types, dtype=torch.long),
        candidate_mask=candidate_mask,
        target_index=target_index,
        game_id=int(game_id),
        action_id=int(action_id),
    )

class HeteroGraphPassSelectionDataset(Dataset):
    def __init__(self, path):
        self.store = Path(path)

        self.df_candidates = pd.read_parquet(self.store / "x_pass_options.parquet")
        self.df_freeze = pd.read_parquet(self.store / "x_freeze_frame_360.parquet")
        self.df_start = pd.read_parquet(self.store / "x_startlocation.parquet")
        self.df_labels = pd.read_parquet(self.store / "y_receiver.parquet")

        self.df_actions = self.df_freeze.join(self.df_start, how="inner")
        self.keys = list(self.df_candidates.index.droplevel("pass_option_id").unique())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        game_id, action_id = self.keys[idx]

        action_row = self.df_actions.loc[(game_id, action_id)]
        candidate_df = self.df_candidates.loc[(game_id, action_id)]
        label_df = self.df_labels.loc[(game_id, action_id)]

        return build_pass_graph_hetero(
            game_id=game_id,
            action_id=action_id,
            action_row=action_row,
            candidate_df=candidate_df,
            label_df=label_df,
        )


@dataclass
class EdgeSimpleGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    candidate_mask: torch.Tensor


@dataclass
class EdgeMultiGraphSample:
    pass_graph: EdgeSimpleGraph
    pressure_graph: EdgeSimpleGraph
    target_index: torch.Tensor
    game_id: int
    action_id: int

class TwoGraphEdgePassSelectionDataset(Dataset):
    def __init__(self, path):
        self.store = Path(path)

        self.df_candidates = pd.read_parquet(self.store / "x_pass_options.parquet")
        self.df_freeze = pd.read_parquet(self.store / "x_freeze_frame_360.parquet")
        self.df_start = pd.read_parquet(self.store / "x_startlocation.parquet")
        self.df_labels = pd.read_parquet(self.store / "y_receiver.parquet")

        self.df_actions = self.df_freeze.join(self.df_start, how="inner")
        self.keys = list(
            self.df_candidates.index.droplevel("pass_option_id").unique()
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        game_id, action_id = self.keys[idx]

        action_row = self.df_actions.loc[(game_id, action_id)]
        candidate_df = self.df_candidates.loc[(game_id, action_id)]
        label_df = self.df_labels.loc[(game_id, action_id)]

        if isinstance(candidate_df, pd.Series):
            candidate_df = candidate_df.to_frame().T
        if isinstance(label_df, pd.Series):
            label_df = label_df.to_frame().T

        candidate_df = candidate_df.sort_index()
        label_df = label_df.sort_index()

        freeze_frame = action_row["freeze_frame_360_a0"]
        if freeze_frame is None:
            freeze_frame = []

        defenders = [p for p in freeze_frame if not p["teammate"]]

        start_x = float(action_row["start_x_a0"])
        start_y = float(action_row["start_y_a0"])

        n_cand = len(candidate_df)

        # --------------------------
        # Pass Graph: [passer] + [candidates]
        # --------------------------
        pass_node_features = []
        pass_edges = []
        pass_edge_features = []

        passer_feat = _base_feat(start_x, start_y, "passer") + [0.0] * 6
        pass_node_features.append(passer_feat)

        candidate_xy = []

        for local_idx, (_, row) in enumerate(candidate_df.iterrows()):
            cx = float(row["destination_x"])
            cy = float(row["destination_y"])
            candidate_xy.append((cx, cy))

            cand_feat = _base_feat(cx, cy, "candidate") + _candidate_pass_feat(row)
            pass_node_features.append(cand_feat)

            # node 0 is passer, candidates start at 1
            pass_edges.append([0, local_idx + 1])
            pass_edge_features.append(_candidate_pass_feat(row))

        if len(pass_edges) == 0:
            pass_edges = [[0, 0]]
            pass_edge_features = [[0.0] * 6]

        pass_candidate_mask = torch.zeros(1 + n_cand, dtype=torch.bool)
        pass_candidate_mask[1:] = True

        pass_graph = EdgeSimpleGraph(
            node_features=torch.tensor(pass_node_features, dtype=torch.float32),
            edge_index=torch.tensor(pass_edges, dtype=torch.long).t().contiguous(),
            edge_features=torch.tensor(pass_edge_features, dtype=torch.float32),
            candidate_mask=pass_candidate_mask,
        )

        # --------------------------
        # Pressure Graph: [candidates] + [defenders]
        # --------------------------
        pressure_node_features = []
        pressure_edges = []
        pressure_edge_features = []

        for cx, cy in candidate_xy:
            pressure_node_features.append(_base_feat(cx, cy, "candidate"))

        for defender in defenders:
            dx = float(defender["x"])
            dy = float(defender["y"])
            pressure_node_features.append(_base_feat(dx, dy, "defender"))

        def_start = n_cand

        for c_idx, (cx, cy) in enumerate(candidate_xy):
            for d_idx, defender in enumerate(defenders):
                d = def_start + d_idx
                dx = float(defender["x"])
                dy = float(defender["y"])

                pressure_edges.append([c_idx, d])
                pressure_edge_features.append(_geom_edge_feat(cx, cy, dx, dy))

                pressure_edges.append([d, c_idx])
                pressure_edge_features.append(_geom_edge_feat(dx, dy, cx, cy))

        if len(pressure_edges) == 0:
            pressure_edges = [[0, 0]]
            pressure_edge_features = [[0.0] * 6]

        pressure_candidate_mask = torch.zeros(n_cand + len(defenders), dtype=torch.bool)
        pressure_candidate_mask[:n_cand] = True

        pressure_graph = EdgeSimpleGraph(
            node_features=torch.tensor(pressure_node_features, dtype=torch.float32),
            edge_index=torch.tensor(pressure_edges, dtype=torch.long).t().contiguous(),
            edge_features=torch.tensor(pressure_edge_features, dtype=torch.float32),
            candidate_mask=pressure_candidate_mask,
        )

        target_index = torch.tensor(
            label_df["receiver"].tolist().index(True),
            dtype=torch.long,
        )

        return EdgeMultiGraphSample(
            pass_graph=pass_graph,
            pressure_graph=pressure_graph,
            target_index=target_index,
            game_id=int(game_id),
            action_id=int(action_id),
        )

class GraphPassSelectionDatasetWithEventContext(Dataset):
    def __init__(self, path, context_mean=None, context_std=None):
        self.base = GraphPassSelectionDataset(path=path)

        context_path = Path(path) / "x_pass_event_context.parquet"
        self.context_df = pd.read_parquet(context_path)
        self.context_df = self.context_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Same event context is repeated for each pass_option_id.
        self.context_by_action = (
            self.context_df
            .groupby(level=["game_id", "action_id"])
            .first()
            .astype("float32")
        )

        if context_mean is None or context_std is None:
            self.context_mean = self.context_by_action.mean()
            self.context_std = self.context_by_action.std().replace(0, 1.0)
        else:
            self.context_mean = context_mean
            self.context_std = context_std

    def __len__(self):
        return len(self.base)

    @property
    def keys(self):
        return self.base.keys

    def __getitem__(self, idx):
        sample = self.base[idx]
        key = (sample.game_id, sample.action_id)

        context = self.context_by_action.loc[key]
        context = (context - self.context_mean) / self.context_std

        sample.event_context = torch.tensor(
            context.values,
            dtype=torch.float32,
        )
        return sample

# gnn60
class GraphPassSelectionDatasetWithEventContextAndHistory(Dataset):
    def __init__(
        self,
        path,
        history_len=30,
        context_mean=None,
        context_std=None,
        history_cols=None,
        history_mean=None,
        history_std=None,
    ):
        self.path = Path(path)
        self.history_len = history_len

        self.base = GraphPassSelectionDatasetWithEventContext(
            path=self.path,
            context_mean=context_mean,
            context_std=context_std,
        )

        self.context_mean = self.base.context_mean
        self.context_std = self.base.context_std

        history_path = self.path / "x_pass_possession_history.parquet"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history file: {history_path}")

        history_df = (
            pd.read_parquet(history_path)
            .sort_index()
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        if history_cols is None:
            history_cols = [
                c for c in history_df.columns
                if c != "hist_action_id"
            ]

        self.history_cols = list(history_cols)
        self.history_df = (
            history_df
            .reindex(columns=self.history_cols, fill_value=0.0)
            .sort_index()
        )

        if history_mean is None:
            history_mean = self.history_df.mean(axis=0)
        if history_std is None:
            history_std = self.history_df.std(axis=0).replace(0, 1.0)

        self.history_mean = pd.Series(history_mean, index=self.history_cols).fillna(0.0)
        self.history_std = (
            pd.Series(history_std, index=self.history_cols)
            .replace(0, 1.0)
            .fillna(1.0)
        )

    def __len__(self):
        return len(self.base)

    @property
    def keys(self):
        return self.base.keys

    @property
    def history_dim(self):
        return len(self.history_cols)

    def _get_history(self, game_id, action_id):
        key = (game_id, action_id)

        try:
            hist = self.history_df.loc[key]
        except KeyError:
            hist = pd.DataFrame(columns=self.history_cols)

        if isinstance(hist, pd.Series):
            hist = hist.to_frame().T

        stored_len = len(hist)
        has_history = float(stored_len > 0)
        stored_history_len_norm = min(stored_len, 50) / 50.0

        if self.history_len == 0 or stored_len == 0:
            history_tensor = torch.zeros(
                0,
                self.history_dim,
                dtype=torch.float32,
            )
        else:
            hist = hist.sort_index().tail(self.history_len)
            hist = hist.reindex(columns=self.history_cols, fill_value=0.0).fillna(0.0)
            hist = (hist - self.history_mean) / self.history_std

            history_tensor = torch.tensor(
                hist.to_numpy(dtype="float32"),
                dtype=torch.float32,
            )

        return history_tensor, has_history, stored_history_len_norm, stored_len

    def __getitem__(self, idx):
        sample = self.base[idx]

        history, has_history, stored_history_len_norm, stored_len = self._get_history(
            sample.game_id,
            sample.action_id,
        )

        sample.possession_history = history
        sample.has_history = torch.tensor([has_history], dtype=torch.float32)
        sample.stored_history_len_norm = torch.tensor(
            [stored_history_len_norm],
            dtype=torch.float32,
        )
        sample.stored_history_len = stored_len

        return sample
    
# gnn66
class GraphPassSelectionDatasetWithEventContextAndHistoryNoFixedHistory(Dataset):
    def __init__(
        self,
        path,
        history_len=30,
        context_mean=None,
        context_std=None,
        context_cols=None,
        history_cols=None,
        history_mean=None,
        history_std=None,
    ):
        self.path = Path(path)
        self.history_len = history_len

        self.base = GraphPassSelectionDatasetWithEventContext(path=self.path)

        context_by_action = self.base.context_by_action.copy()

        if context_cols is None:
            context_cols = [
                c for c in context_by_action.columns
                if not (
                    "_a1" in c
                    or "_a2" in c
                )
            ]

        self.context_cols = list(context_cols)
        self.base.context_by_action = context_by_action.reindex(
            columns=self.context_cols,
            fill_value=0.0,
        )

        if context_mean is None:
            self.context_mean = self.base.context_by_action.mean(axis=0)
        else:
            self.context_mean = context_mean

        if context_std is None:
            self.context_std = self.base.context_by_action.std(axis=0).replace(0, 1.0)
        else:
            self.context_std = context_std

        self.base.context_mean = pd.Series(
            self.context_mean,
            index=self.context_cols,
        ).fillna(0.0)

        self.base.context_std = (
            pd.Series(
                self.context_std,
                index=self.context_cols,
            )
            .replace(0, 1.0)
            .fillna(1.0)
        )

        history_path = self.path / "x_pass_possession_history.parquet"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history file: {history_path}")

        history_df = (
            pd.read_parquet(history_path)
            .sort_index()
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        if history_cols is None:
            history_cols = [
                c for c in history_df.columns
                if c != "hist_action_id"
            ]

        self.history_cols = list(history_cols)
        self.history_df = (
            history_df
            .reindex(columns=self.history_cols, fill_value=0.0)
            .sort_index()
        )

        if history_mean is None:
            history_mean = self.history_df.mean(axis=0)
        if history_std is None:
            history_std = self.history_df.std(axis=0).replace(0, 1.0)

        self.history_mean = pd.Series(history_mean, index=self.history_cols).fillna(0.0)
        self.history_std = (
            pd.Series(history_std, index=self.history_cols)
            .replace(0, 1.0)
            .fillna(1.0)
        )

    def __len__(self):
        return len(self.base)

    @property
    def keys(self):
        return self.base.keys

    @property
    def history_dim(self):
        return len(self.history_cols)

    @property
    def context_dim(self):
        return len(self.context_cols)

    def _get_history(self, game_id, action_id):
        key = (game_id, action_id)

        try:
            hist = self.history_df.loc[key]
        except KeyError:
            hist = pd.DataFrame(columns=self.history_cols)

        if isinstance(hist, pd.Series):
            hist = hist.to_frame().T

        stored_len = len(hist)
        has_history = float(stored_len > 0)
        stored_history_len_norm = min(stored_len, 50) / 50.0

        if self.history_len == 0 or stored_len == 0:
            history_tensor = torch.zeros(
                0,
                self.history_dim,
                dtype=torch.float32,
            )
        else:
            hist = hist.sort_index().tail(self.history_len)
            hist = hist.reindex(columns=self.history_cols, fill_value=0.0).fillna(0.0)
            hist = (hist - self.history_mean) / self.history_std

            history_tensor = torch.tensor(
                hist.to_numpy(dtype="float32"),
                dtype=torch.float32,
            )

        return history_tensor, has_history, stored_history_len_norm, stored_len

    def __getitem__(self, idx):
        sample = self.base[idx]

        history, has_history, stored_history_len_norm, stored_len = self._get_history(
            sample.game_id,
            sample.action_id,
        )

        sample.possession_history = history
        sample.has_history = torch.tensor([has_history], dtype=torch.float32)
        sample.stored_history_len_norm = torch.tensor(
            [stored_history_len_norm],
            dtype=torch.float32,
        )
        sample.stored_history_len = stored_len

        return sample

# gnn68
class GraphPassSelectionDatasetWithEventContextAndEarlierHistory(Dataset):
    def __init__(
        self,
        path,
        history_len=30,
        exclude_recent_steps=2,
        context_mean=None,
        context_std=None,
        history_cols=None,
        history_mean=None,
        history_std=None,
    ):
        self.path = Path(path)
        self.history_len = history_len
        self.exclude_recent_steps = exclude_recent_steps

        self.base = GraphPassSelectionDatasetWithEventContext(
            path=self.path,
            context_mean=context_mean,
            context_std=context_std,
        )

        self.context_mean = self.base.context_mean
        self.context_std = self.base.context_std

        history_path = self.path / "x_pass_possession_history.parquet"
        if not history_path.exists():
            raise FileNotFoundError(f"Missing history file: {history_path}")

        history_df = (
            pd.read_parquet(history_path)
            .sort_index()
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        if history_cols is None:
            history_cols = [
                c for c in history_df.columns
                if c != "hist_action_id"
            ]

        self.history_cols = list(history_cols)
        self.history_df = (
            history_df
            .reindex(columns=self.history_cols, fill_value=0.0)
            .sort_index()
        )

        if history_mean is None:
            history_mean = self.history_df.mean(axis=0)
        if history_std is None:
            history_std = self.history_df.std(axis=0).replace(0, 1.0)

        self.history_mean = pd.Series(
            history_mean,
            index=self.history_cols,
        ).fillna(0.0)

        self.history_std = (
            pd.Series(history_std, index=self.history_cols)
            .replace(0, 1.0)
            .fillna(1.0)
        )

    def __len__(self):
        return len(self.base)

    @property
    def keys(self):
        return self.base.keys

    @property
    def history_dim(self):
        return len(self.history_cols)

    def _get_history(self, game_id, action_id):
        key = (game_id, action_id)

        try:
            hist = self.history_df.loc[key]
        except KeyError:
            hist = pd.DataFrame(columns=self.history_cols)

        if isinstance(hist, pd.Series):
            hist = hist.to_frame().T

        hist = hist.sort_index()
        stored_len = len(hist)

        earlier_available_len = max(
            stored_len - self.exclude_recent_steps,
            0,
        )

        if earlier_available_len > 0:
            earlier_hist = hist.iloc[:-self.exclude_recent_steps]
        else:
            earlier_hist = hist.iloc[0:0]

        has_history = float(earlier_available_len > 0)
        stored_history_len_norm = min(earlier_available_len, 50) / 50.0

        if self.history_len == 0 or earlier_available_len == 0:
            history_tensor = torch.zeros(
                0,
                self.history_dim,
                dtype=torch.float32,
            )
        else:
            earlier_hist = earlier_hist.tail(self.history_len)
            earlier_hist = (
                earlier_hist
                .reindex(columns=self.history_cols, fill_value=0.0)
                .fillna(0.0)
            )

            earlier_hist = (
                earlier_hist - self.history_mean
            ) / self.history_std

            history_tensor = torch.tensor(
                earlier_hist.to_numpy(dtype="float32"),
                dtype=torch.float32,
            )

        return (
            history_tensor,
            has_history,
            stored_history_len_norm,
            stored_len,
            earlier_available_len,
        )

    def __getitem__(self, idx):
        sample = self.base[idx]

        (
            history,
            has_history,
            stored_history_len_norm,
            stored_len,
            earlier_available_len,
        ) = self._get_history(
            sample.game_id,
            sample.action_id,
        )

        sample.possession_history = history
        sample.has_history = torch.tensor(
            [has_history],
            dtype=torch.float32,
        )
        sample.stored_history_len_norm = torch.tensor(
            [stored_history_len_norm],
            dtype=torch.float32,
        )

        sample.stored_history_len = stored_len
        sample.earlier_available_len = earlier_available_len

        return sample
    
# gnn71
class GraphPassSelectionDatasetWithEventContextAndHistoryRelation(
    GraphPassSelectionDatasetWithEventContextAndHistory
):
    """Event-context and dynamic-history dataset with raw spatial history geometry."""

    geometry_cols = [
        "hist_start_x",
        "hist_start_y",
        "hist_end_x",
        "hist_end_y",
    ]

    def _get_history_and_geometry(self, game_id, action_id):
        key = (game_id, action_id)

        try:
            hist = self.history_df.loc[key]
        except KeyError:
            hist = pd.DataFrame(columns=self.history_cols)

        if isinstance(hist, pd.Series):
            hist = hist.to_frame().T

        stored_len = len(hist)
        has_history = float(stored_len > 0)
        stored_history_len_norm = min(stored_len, 50) / 50.0

        if self.history_len == 0 or stored_len == 0:
            history_tensor = torch.zeros(
                0,
                self.history_dim,
                dtype=torch.float32,
            )
            history_geometry = torch.zeros(
                0,
                4,
                dtype=torch.float32,
            )
        else:
            selected_hist = hist.sort_index().tail(self.history_len)
            selected_hist = selected_hist.reindex(
                columns=self.history_cols,
                fill_value=0.0,
            ).fillna(0.0)

            geometry = selected_hist[self.geometry_cols].copy()
            geometry["hist_start_x"] = geometry["hist_start_x"] / 105.0
            geometry["hist_start_y"] = geometry["hist_start_y"] / 68.0
            geometry["hist_end_x"] = geometry["hist_end_x"] / 105.0
            geometry["hist_end_y"] = geometry["hist_end_y"] / 68.0

            history_geometry = torch.tensor(
                geometry.to_numpy(dtype="float32"),
                dtype=torch.float32,
            )

            normalized_hist = (
                selected_hist - self.history_mean
            ) / self.history_std

            history_tensor = torch.tensor(
                normalized_hist.to_numpy(dtype="float32"),
                dtype=torch.float32,
            )

        return (
            history_tensor,
            history_geometry,
            has_history,
            stored_history_len_norm,
            stored_len,
        )

    def __getitem__(self, idx):
        sample = self.base[idx]

        (
            history,
            history_geometry,
            has_history,
            stored_history_len_norm,
            stored_len,
        ) = self._get_history_and_geometry(
            sample.game_id,
            sample.action_id,
        )

        sample.possession_history = history
        sample.possession_history_geometry = history_geometry
        sample.has_history = torch.tensor(
            [has_history],
            dtype=torch.float32,
        )
        sample.stored_history_len_norm = torch.tensor(
            [stored_history_len_norm],
            dtype=torch.float32,
        )
        sample.stored_history_len = stored_len

        return sample
    
# gnn72
class GraphPassSelectionDatasetWithEventContextAndPassOnlyHistory(
    GraphPassSelectionDatasetWithEventContextAndHistory
):
    """Keep pass/cross actions only within the latest history_len event window."""

    pass_type_cols = (
        "hist_type_name_pass",
        "hist_type_name_cross",
    )

    def _get_history(self, game_id, action_id):
        key = (game_id, action_id)

        try:
            hist = self.history_df.loc[key]
        except KeyError:
            hist = pd.DataFrame(columns=self.history_cols)

        if isinstance(hist, pd.Series):
            hist = hist.to_frame().T

        hist = hist.sort_index()

        # Keep metadata definition identical to gnn65:
        # it describes whether the original dynamic event history exists.
        stored_len = len(hist)
        has_history = float(stored_len > 0)
        stored_history_len_norm = min(stored_len, 50) / 50.0

        if self.history_len == 0 or stored_len == 0:
            history_tensor = torch.zeros(
                0,
                self.history_dim,
                dtype=torch.float32,
            )
        else:
            # Step 1: use the same recent-event observation window as gnn65.
            event_window = hist.tail(self.history_len)

            # Step 2: retain pass-like actions only inside this window.
            pass_mask = (
                event_window["hist_type_name_pass"]
                + event_window["hist_type_name_cross"]
            ) > 0.5

            pass_history = event_window.loc[pass_mask]

            if len(pass_history) == 0:
                history_tensor = torch.zeros(
                    0,
                    self.history_dim,
                    dtype=torch.float32,
                )
            else:
                pass_history = (
                    pass_history
                    .reindex(columns=self.history_cols, fill_value=0.0)
                    .fillna(0.0)
                )

                # Keep the same train-derived normalization interface as gnn65.
                pass_history = (
                    pass_history - self.history_mean
                ) / self.history_std

                history_tensor = torch.tensor(
                    pass_history.to_numpy(dtype="float32"),
                    dtype=torch.float32,
                )

        return (
            history_tensor,
            has_history,
            stored_history_len_norm,
            stored_len,
        )

# gnn73
class GraphPassSelectionDatasetWithEventContextAndDualHistory(
    GraphPassSelectionDatasetWithEventContextAndHistory
):
    """Return all-event history and pass-only history from the same event window."""

    pass_type_cols = (
        "hist_type_name_pass",
        "hist_type_name_cross",
    )

    def __init__(
        self,
        path,
        history_len=10,
        context_mean=None,
        context_std=None,
        history_cols=None,
        history_mean=None,
        history_std=None,
        pass_history_mean=None,
        pass_history_std=None,
    ):
        super().__init__(
            path=path,
            history_len=history_len,
            context_mean=context_mean,
            context_std=context_std,
            history_cols=history_cols,
            history_mean=history_mean,
            history_std=history_std,
        )

        missing_cols = [
            col for col in self.pass_type_cols
            if col not in self.history_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing pass-history type columns: {missing_cols}"
            )

        pass_mask = (
            self.history_df[list(self.pass_type_cols)].sum(axis=1) > 0.5
        )

        pass_rows = self.history_df.loc[pass_mask]

        if len(pass_rows) == 0:
            raise ValueError("No historical pass/cross actions were found.")

        if pass_history_mean is None:
            pass_history_mean = pass_rows.mean(axis=0)

        if pass_history_std is None:
            pass_history_std = (
                pass_rows.std(axis=0)
                .replace(0, 1.0)
            )

        self.pass_history_mean = (
            pd.Series(pass_history_mean, index=self.history_cols)
            .fillna(0.0)
        )
        self.pass_history_std = (
            pd.Series(pass_history_std, index=self.history_cols)
            .replace(0, 1.0)
            .fillna(1.0)
        )

    def _get_dual_history(self, game_id, action_id):
        key = (game_id, action_id)

        try:
            hist = self.history_df.loc[key]
        except KeyError:
            hist = pd.DataFrame(columns=self.history_cols)

        if isinstance(hist, pd.Series):
            hist = hist.to_frame().T

        hist = hist.sort_index()

        stored_len = len(hist)
        has_history = float(stored_len > 0)
        stored_history_len_norm = min(stored_len, 50) / 50.0

        if self.history_len == 0 or stored_len == 0:
            all_history_tensor = torch.zeros(
                0,
                self.history_dim,
                dtype=torch.float32,
            )
            pass_history_tensor = torch.zeros(
                0,
                self.history_dim,
                dtype=torch.float32,
            )
            pass_sequence_len = 0
        else:
            event_window = (
                hist.tail(self.history_len)
                .reindex(columns=self.history_cols, fill_value=0.0)
                .fillna(0.0)
            )

            all_history = (
                event_window - self.history_mean
            ) / self.history_std

            all_history_tensor = torch.tensor(
                all_history.to_numpy(dtype="float32"),
                dtype=torch.float32,
            )

            pass_mask = (
                event_window[list(self.pass_type_cols)].sum(axis=1) > 0.5
            )
            pass_history = event_window.loc[pass_mask]
            pass_sequence_len = len(pass_history)

            if pass_sequence_len == 0:
                pass_history_tensor = torch.zeros(
                    0,
                    self.history_dim,
                    dtype=torch.float32,
                )
            else:
                pass_history = (
                    pass_history - self.pass_history_mean
                ) / self.pass_history_std

                pass_history_tensor = torch.tensor(
                    pass_history.to_numpy(dtype="float32"),
                    dtype=torch.float32,
                )

        has_pass_history = float(pass_sequence_len > 0)
        pass_history_len_norm = min(pass_sequence_len, 50) / 50.0

        return (
            all_history_tensor,
            pass_history_tensor,
            has_history,
            stored_history_len_norm,
            has_pass_history,
            pass_history_len_norm,
            stored_len,
            pass_sequence_len,
        )

    def __getitem__(self, idx):
        sample = self.base[idx]

        (
            all_history,
            pass_history,
            has_history,
            stored_history_len_norm,
            has_pass_history,
            pass_history_len_norm,
            stored_len,
            pass_sequence_len,
        ) = self._get_dual_history(
            sample.game_id,
            sample.action_id,
        )

        sample.possession_history = all_history
        sample.pass_only_history = pass_history

        sample.has_history = torch.tensor(
            [has_history],
            dtype=torch.float32,
        )
        sample.stored_history_len_norm = torch.tensor(
            [stored_history_len_norm],
            dtype=torch.float32,
        )

        sample.has_pass_history = torch.tensor(
            [has_pass_history],
            dtype=torch.float32,
        )
        sample.pass_history_len_norm = torch.tensor(
            [pass_history_len_norm],
            dtype=torch.float32,
        )

        sample.stored_history_len = stored_len
        sample.pass_sequence_len = pass_sequence_len

        return sample
    
# gnn74
# gnn74
class GraphPassSelectionDatasetWithEventContextAndDisjointDualHistory(
    GraphPassSelectionDatasetWithEventContextAndDualHistory
):
    """Split one recent-event window into non-pass and pass-only sequences."""

    def __init__(
        self,
        path,
        history_len=10,
        context_mean=None,
        context_std=None,
        history_cols=None,
        history_mean=None,
        history_std=None,
        pass_history_mean=None,
        pass_history_std=None,
        non_pass_history_mean=None,
        non_pass_history_std=None,
    ):
        super().__init__(
            path=path,
            history_len=history_len,
            context_mean=context_mean,
            context_std=context_std,
            history_cols=history_cols,
            history_mean=history_mean,
            history_std=history_std,
            pass_history_mean=pass_history_mean,
            pass_history_std=pass_history_std,
        )

        pass_mask = (
            self.history_df[list(self.pass_type_cols)].sum(axis=1) > 0.5
        )
        non_pass_rows = self.history_df.loc[~pass_mask]

        if len(non_pass_rows) == 0:
            raise ValueError("No historical non-pass actions were found.")

        if non_pass_history_mean is None:
            non_pass_history_mean = non_pass_rows.mean(axis=0)

        if non_pass_history_std is None:
            non_pass_history_std = (
                non_pass_rows.std(axis=0)
                .replace(0, 1.0)
            )

        self.non_pass_history_mean = (
            pd.Series(non_pass_history_mean, index=self.history_cols)
            .fillna(0.0)
        )
        self.non_pass_history_std = (
            pd.Series(non_pass_history_std, index=self.history_cols)
            .replace(0, 1.0)
            .fillna(1.0)
        )

    def _get_disjoint_history(self, game_id, action_id):
        key = (game_id, action_id)

        try:
            hist = self.history_df.loc[key]
        except KeyError:
            hist = pd.DataFrame(columns=self.history_cols)

        if isinstance(hist, pd.Series):
            hist = hist.to_frame().T

        hist = hist.sort_index()

        stored_len = len(hist)

        # Keep these two metadata fields consistent with gnn73.
        # They describe whether stored possession history exists.
        has_history = float(stored_len > 0)
        stored_history_len_norm = min(stored_len, 50) / 50.0

        if self.history_len == 0 or stored_len == 0:
            event_window_len = 0
            non_pass_sequence_len = 0
            pass_sequence_len = 0

            non_pass_history_tensor = torch.zeros(
                0,
                self.history_dim,
                dtype=torch.float32,
            )
            pass_history_tensor = torch.zeros(
                0,
                self.history_dim,
                dtype=torch.float32,
            )
        else:
            event_window = (
                hist.tail(self.history_len)
                .reindex(columns=self.history_cols, fill_value=0.0)
                .fillna(0.0)
            )

            event_window_len = len(event_window)

            pass_mask = (
                event_window[list(self.pass_type_cols)].sum(axis=1) > 0.5
            )

            pass_history = event_window.loc[pass_mask]
            non_pass_history = event_window.loc[~pass_mask]

            pass_sequence_len = len(pass_history)
            non_pass_sequence_len = len(non_pass_history)

            if non_pass_sequence_len == 0:
                non_pass_history_tensor = torch.zeros(
                    0,
                    self.history_dim,
                    dtype=torch.float32,
                )
            else:
                non_pass_history = (
                    non_pass_history - self.non_pass_history_mean
                ) / self.non_pass_history_std

                non_pass_history_tensor = torch.tensor(
                    non_pass_history.to_numpy(dtype="float32"),
                    dtype=torch.float32,
                )

            if pass_sequence_len == 0:
                pass_history_tensor = torch.zeros(
                    0,
                    self.history_dim,
                    dtype=torch.float32,
                )
            else:
                pass_history = (
                    pass_history - self.pass_history_mean
                ) / self.pass_history_std

                pass_history_tensor = torch.tensor(
                    pass_history.to_numpy(dtype="float32"),
                    dtype=torch.float32,
                )

        has_non_pass_history = float(non_pass_sequence_len > 0)
        non_pass_history_len_norm = min(non_pass_sequence_len, 50) / 50.0

        has_pass_history = float(pass_sequence_len > 0)
        pass_history_len_norm = min(pass_sequence_len, 50) / 50.0

        return (
            non_pass_history_tensor,
            pass_history_tensor,
            has_history,
            stored_history_len_norm,
            has_non_pass_history,
            non_pass_history_len_norm,
            has_pass_history,
            pass_history_len_norm,
            stored_len,
            event_window_len,
            non_pass_sequence_len,
            pass_sequence_len,
        )

    def __getitem__(self, idx):
        sample = self.base[idx]

        (
            non_pass_history,
            pass_history,
            has_history,
            stored_history_len_norm,
            has_non_pass_history,
            non_pass_history_len_norm,
            has_pass_history,
            pass_history_len_norm,
            stored_len,
            event_window_len,
            non_pass_sequence_len,
            pass_sequence_len,
        ) = self._get_disjoint_history(
            sample.game_id,
            sample.action_id,
        )

        # The gnn73 model reads possession_history as its global branch input.
        # In gnn74 this global branch now receives non-pass history only.
        sample.possession_history = non_pass_history
        sample.pass_only_history = pass_history

        # Keep model-consumed global metadata identical in definition to gnn73
        # so the controlled change is the global token sequence content.
        sample.has_history = torch.tensor(
            [has_history],
            dtype=torch.float32,
        )
        sample.stored_history_len_norm = torch.tensor(
            [stored_history_len_norm],
            dtype=torch.float32,
        )

        sample.has_pass_history = torch.tensor(
            [has_pass_history],
            dtype=torch.float32,
        )
        sample.pass_history_len_norm = torch.tensor(
            [pass_history_len_norm],
            dtype=torch.float32,
        )

        # Extra fields for checking and reporting only.
        sample.has_non_pass_history = torch.tensor(
            [has_non_pass_history],
            dtype=torch.float32,
        )
        sample.non_pass_history_len_norm = torch.tensor(
            [non_pass_history_len_norm],
            dtype=torch.float32,
        )
        sample.stored_history_len = stored_len
        sample.event_window_len = event_window_len
        sample.non_pass_sequence_len = non_pass_sequence_len
        sample.pass_sequence_len = pass_sequence_len

        return sample
    
# gnn75
class GraphPassSelectionDatasetWithEventContextAndContextualizedPassHistory(
    GraphPassSelectionDatasetWithEventContextAndHistory
):
    """Return all-event history plus an aligned pass/cross mask."""

    pass_type_cols = (
        "hist_type_name_pass",
        "hist_type_name_cross",
    )

    def _get_history_and_pass_mask(self, game_id, action_id):
        key = (game_id, action_id)

        try:
            hist = self.history_df.loc[key]
        except KeyError:
            hist = pd.DataFrame(columns=self.history_cols)

        if isinstance(hist, pd.Series):
            hist = hist.to_frame().T

        hist = hist.sort_index()

        stored_len = len(hist)
        has_history = float(stored_len > 0)
        stored_history_len_norm = min(stored_len, 50) / 50.0

        if self.history_len == 0 or stored_len == 0:
            event_window_len = 0
            pass_sequence_len = 0

            history_tensor = torch.zeros(
                0,
                self.history_dim,
                dtype=torch.float32,
            )

            history_pass_mask = torch.zeros(
                0,
                dtype=torch.bool,
            )
        else:
            event_window = (
                hist.tail(self.history_len)
                .reindex(columns=self.history_cols, fill_value=0.0)
                .fillna(0.0)
            )

            event_window_len = len(event_window)

            pass_mask = (
                event_window[list(self.pass_type_cols)].sum(axis=1) > 0.5
            )

            pass_sequence_len = int(pass_mask.sum())

            normalized_history = (
                event_window - self.history_mean
            ) / self.history_std

            history_tensor = torch.tensor(
                normalized_history.to_numpy(dtype="float32"),
                dtype=torch.float32,
            )

            history_pass_mask = torch.tensor(
                pass_mask.to_numpy(dtype="bool"),
                dtype=torch.bool,
            )

        has_pass_history = float(pass_sequence_len > 0)
        pass_history_len_norm = min(pass_sequence_len, 50) / 50.0

        return (
            history_tensor,
            history_pass_mask,
            has_history,
            stored_history_len_norm,
            has_pass_history,
            pass_history_len_norm,
            stored_len,
            event_window_len,
            pass_sequence_len,
        )

    def __getitem__(self, idx):
        sample = self.base[idx]

        (
            history,
            history_pass_mask,
            has_history,
            stored_history_len_norm,
            has_pass_history,
            pass_history_len_norm,
            stored_len,
            event_window_len,
            pass_sequence_len,
        ) = self._get_history_and_pass_mask(
            sample.game_id,
            sample.action_id,
        )

        sample.possession_history = history
        sample.history_pass_mask = history_pass_mask

        sample.has_history = torch.tensor(
            [has_history],
            dtype=torch.float32,
        )
        sample.stored_history_len_norm = torch.tensor(
            [stored_history_len_norm],
            dtype=torch.float32,
        )

        sample.has_pass_history = torch.tensor(
            [has_pass_history],
            dtype=torch.float32,
        )
        sample.pass_history_len_norm = torch.tensor(
            [pass_history_len_norm],
            dtype=torch.float32,
        )

        sample.stored_history_len = stored_len
        sample.event_window_len = event_window_len
        sample.pass_sequence_len = pass_sequence_len

        return sample
    
# gnn75-4
class GraphPassSelectionDatasetWithCurrentContextAndContextualizedPassHistory(
    GraphPassSelectionDatasetWithEventContextAndContextualizedPassHistory
):
    """gnn75-4: remove fixed a1/a2 features from event context."""

    def __init__(
        self,
        path,
        history_len=40,
        context_mean=None,
        context_std=None,
        context_cols=None,
        history_cols=None,
        history_mean=None,
        history_std=None,
    ):
        super().__init__(
            path=path,
            history_len=history_len,
            context_mean=None,
            context_std=None,
            history_cols=history_cols,
            history_mean=history_mean,
            history_std=history_std,
        )

        context_by_action = self.base.context_by_action.copy()

        if context_cols is None:
            context_cols = [
                c for c in context_by_action.columns
                if not (
                    "_a1" in c
                    or "_a2" in c
                )
            ]

        self.context_cols = list(context_cols)

        self.base.context_by_action = context_by_action.reindex(
            columns=self.context_cols,
            fill_value=0.0,
        )

        if context_mean is None:
            self.context_mean = self.base.context_by_action.mean(axis=0)
        else:
            self.context_mean = context_mean

        if context_std is None:
            self.context_std = (
                self.base.context_by_action
                .std(axis=0)
                .replace(0, 1.0)
            )
        else:
            self.context_std = context_std

        self.base.context_mean = pd.Series(
            self.context_mean,
            index=self.context_cols,
        ).fillna(0.0)

        self.base.context_std = (
            pd.Series(
                self.context_std,
                index=self.context_cols,
            )
            .replace(0, 1.0)
            .fillna(1.0)
        )

    @property
    def context_dim(self):
        return len(self.context_cols)