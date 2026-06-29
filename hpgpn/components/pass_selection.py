"""Implements the pass selection component."""
import copy
import json
import math
from typing import Any, Dict, List, Optional

import random
import hydra
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from rich.progress import track
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
#from mamba_ssm import Mamba
from torch_geometric.nn import RGCNConv

from hpgpn.components.soccermap import SoccerMap, pixel
from hpgpn.config import logger as log
from hpgpn.datasets import GraphPassSelectionDataset, GraphSample

from .base import UnxpassComponent, UnxPassPytorchComponent, UnxPassXGBoostComponent


class PassSelectionComponent(UnxpassComponent):
    """The pass selection component.

    From any given game situation where a player controls the ball, the model
    estimates the most likely destination of a potential pass.
    """

    component_name = "pass_selection"

    def _get_metrics(self, y, y_hat):
        return {
            "log_loss": log_loss(y, y_hat, labels=[0, 1]),
            "brier": brier_score_loss(y, y_hat),
        }


class ClosestPlayerBaseline(PassSelectionComponent):
    """A baseline model that predicts the closest player to the ball as the most likely receiver."""

    def __init__(self):
        super().__init__(
            features={
                "pass_options": ["distance"],
            },
            label=["receiver"],
        )

    def train(self, dataset, optimized_metric=None):
        # No training required
        # Return metric score for hyperparameter optimization
        if optimized_metric is not None:
            metrics = self.test(dataset)
            return metrics[optimized_metric]

        return None

    def test(self, dataset) -> Dict:
        data = self.initialize_dataset(dataset)
        X_test, y = data.features, data.labels
        y["pred"] = X_test.distance
        y["pred_rank"] = (
            y.groupby(["game_id", "action_id"])["pred"].rank("dense", ascending=True).astype(int)
        )
        return {"acc": len(y[y.receiver & (y.pred_rank == 1)]) / y.receiver.sum()}

    def predict(self, dataset) -> pd.Series:
        data = self.initialize_dataset(dataset)
        pred = data.features[["distance"]]
        pred["rank"] = (
            pred.groupby(["game_id", "action_id"])["distance"]
            .rank("dense", ascending=True)
            .astype(int)
        )
        return pd.Series(
            (pred["rank"] == 1).astype(float).values.tolist(),
            index=data.features.index,
        )


class XGBoostComponent(PassSelectionComponent, UnxPassXGBoostComponent):
    """A XGBoost model based on handcrafted features."""

    def __init__(
        self,
        model: xgb.XGBRanker,
        features: Dict[str, List[str]],
        label: List[str] = ["receiver"],
    ):
        super().__init__(
            model=model,
            features=features,
            label=label,
        )

    def train(self, dataset, optimized_metric=None, **train_cfg) -> Optional[float]:
        mlflow.xgboost.autolog()

        # Load data
        data = self.initialize_dataset(dataset)
        X_train, X_val, y_train, y_val = train_test_split(
            data.features, data.labels, test_size=0.2
        )
        group_train = X_train.groupby(["game_id", "action_id"]).size()
        group_val = X_val.groupby(["game_id", "action_id"]).size()

        # Train the model
        log.info("Fitting model on train set")
        self.model.fit(
            X_train,
            y_train,
            group=group_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            **train_cfg,
        )

        # Return metric score for hyperparameter optimization
        if optimized_metric is not None:
            idx = self.model.best_iteration
            return self.model.evals_result()["validation_0"][optimized_metric][idx]

        return None

    def test(self, dataset) -> Dict[str, float]:
        data = self.initialize_dataset(dataset)
        X_test, y = data.features, data.labels
        y["pred"] = self.model.predict(X_test)
        y["pred_rank"] = (
            y.groupby(["game_id", "action_id"])["pred"].rank("dense", ascending=False).astype(int)
        )
        return {"acc": len(y[y.receiver & (y.pred_rank == 1)]) / y.receiver.sum()}


class PytorchSoccerMapModel(pl.LightningModule):
    """A pass selection model based on the SoccerMap architecture."""

    def __init__(
        self,
        lr: float = 1e-5,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SoccerMap(in_channels=9)
        self.softmax = nn.Softmax(2)

        # loss function
        self.criterion = torch.nn.BCELoss()

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.softmax(x.view(*x.size()[:2], -1)).view_as(x)
        return x

    def step(self, batch: Any):
        x, mask, y = batch
        surface = self.forward(x)
        y_hat = pixel(surface, mask)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        x, _, _ = batch
        surface = self(x)
        return surface

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)


class ToSoccerMapTensor:
    """Convert inputs to a spatial representation.

    Parameters
    ----------
    dim : tuple(int), default=(68, 104)
        The dimensions of the pitch in the spatial representation.
        The original pitch dimensions are 105x68, but even numbers are easier
        to work with.
    """

    def __init__(self, dim=(68, 104)):
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim

    def _get_cell_indexes(self, x, y):
        x_bin = np.clip(x / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip(y / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def __call__(self, sample):
        start_x, start_y, end_x, end_y = (
            sample["start_x_a0"],
            sample["start_y_a0"],
            sample["end_x_a0"],
            sample["end_y_a0"],
        )
        speed_x, speed_y = sample["speedx_a02"], sample["speedy_a02"]
        frame = pd.DataFrame.from_records(sample["freeze_frame_360_a0"])

        # Location of the player that passes the ball
        # passer_coo = frame.loc[frame.actor, ["x", "y"]].fillna(1e-10).values.reshape(-1, 2)
        # Location of the ball
        ball_coo = np.array([[start_x, start_y]])
        # Location of the goal
        goal_coo = np.array([[105, 34]])
        # Locations of the passing player's teammates
        players_att_coo = frame.loc[~frame.actor & frame.teammate, ["x", "y"]].values.reshape(
            -1, 2
        )
        # Locations and speed vector of the defending players
        players_def_coo = frame.loc[~frame.teammate, ["x", "y"]].values.reshape(-1, 2)

        # Output
        matrix = np.zeros((9, self.y_bins, self.x_bins))

        # CH 1: Locations of attacking team
        x_bin_att, y_bin_att = self._get_cell_indexes(
            players_att_coo[:, 0],
            players_att_coo[:, 1],
        )
        matrix[0, y_bin_att, x_bin_att] = 1

        # CH 2: Locations of defending team
        x_bin_def, y_bin_def = self._get_cell_indexes(
            players_def_coo[:, 0],
            players_def_coo[:, 1],
        )
        matrix[1, y_bin_def, x_bin_def] = 1

        # CH 3: Distance to ball
        yy, xx = np.ogrid[0.5 : self.y_bins, 0.5 : self.x_bins]

        x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        matrix[2, :, :] = np.sqrt((xx - x0_ball) ** 2 + (yy - y0_ball) ** 2)

        # CH 4: Distance to goal
        x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
        matrix[3, :, :] = np.sqrt((xx - x0_goal) ** 2 + (yy - y0_goal) ** 2)

        # CH 5: Cosine of the angle between the ball and goal
        coords = np.dstack(np.meshgrid(xx, yy))
        goal_coo_bin = np.concatenate((x0_goal, y0_goal))
        ball_coo_bin = np.concatenate((x0_ball, y0_ball))
        a = goal_coo_bin - coords
        b = ball_coo_bin - coords
        matrix[4, :, :] = np.clip(
            np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1
        )

        # CH 6: Sine of the angle between the ball and goal
        # sin = np.cross(a,b) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2))
        matrix[5, :, :] = np.sqrt(1 - matrix[4, :, :] ** 2)  # This is much faster

        # CH 7: Angle (in radians) to the goal location
        matrix[6, :, :] = np.abs(
            np.arctan((y0_goal - coords[:, :, 1]) / (x0_goal - coords[:, :, 0]))
        )

        # CH 8-9: Ball speed
        matrix[7, y0_ball, x0_ball] = speed_x
        matrix[8, y0_ball, x0_ball] = speed_y

        # Mask
        mask = np.zeros((1, self.y_bins, self.x_bins))
        end_ball_coo = np.array([[end_x, end_y]])
        if np.isnan(end_ball_coo).any():
            raise ValueError("End coordinates not known.")
        x0_ball_end, y0_ball_end = self._get_cell_indexes(end_ball_coo[:, 0], end_ball_coo[:, 1])
        mask[0, y0_ball_end, x0_ball_end] = 1

        if "receiver" in sample:
            target = int(sample["receiver"]) if not math.isnan(sample["receiver"]) else -1
            return (
                torch.from_numpy(matrix).float(),
                torch.from_numpy(mask).float(),
                torch.tensor([target]).float(),
            )
        return (
            torch.from_numpy(matrix).float(),
            torch.from_numpy(mask).float(),
            torch.tensor([1]).float(),
        )


class SoccerMapComponent(PassSelectionComponent, UnxPassPytorchComponent):
    """A SoccerMap deep-learning model."""

    def __init__(self, model: PytorchSoccerMapModel):
        super().__init__(
            model=model,
            features={
                "startlocation": ["start_x_a0", "start_y_a0"],
                "endlocation": ["end_x_a0", "end_y_a0"],
                "speed": ["speedx_a02", "speedy_a02"],
                "freeze_frame_360": ["freeze_frame_360_a0"],
            },
            label=["success"],  # just a dummy lalel
            transform=ToSoccerMapTensor(dim=(68, 104)),
        )

    def test(self, dataset, batch_size=1, num_workers=0, pin_memory=False, **test_cfg) -> Dict:
        # Load dataset
        data = self.initialize_dataset(dataset)
        dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model.eval()

        # Apply model on test set
        all_preds, all_targets = [], []
        for batch in track(dataloader):
            x, mask, _ = batch
            surface = self.model(x)
            for i in range(x.shape[0]):
                teammate_locations = torch.nonzero(x[i, 0, :, :])
                if len(teammate_locations) > 0:
                    p_teammate_selection = surface[
                        i, 0, teammate_locations[:, 0], teammate_locations[:, 1]
                    ]
                    selected_teammate = torch.argmin(
                        torch.cdist(torch.nonzero(mask[i, 0]).float(), teammate_locations.float())
                    )
                    all_targets.append(
                        (torch.argmax(p_teammate_selection) == selected_teammate).item()
                    )
                else:
                    all_targets.append(True)
            y_hat = pixel(surface, mask)
            all_preds.append(y_hat)
        all_preds = torch.cat(all_preds, dim=0).detach().numpy()[:, 0]

        # Compute metrics
        y = np.ones(len(all_preds))
        return {
            "log_loss": log_loss(y, all_preds, labels=[0, 1]),
            "brier": brier_score_loss(y, all_preds),
            "acc": sum(all_targets) / len(all_targets),
        }

 
    

class GraphSAGELayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lin_self = nn.Linear(hidden_dim, hidden_dim)
        self.lin_neigh = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        # x: [N, H]
        # edge_index: [2, E]
        src, dst = edge_index

        # aggregate neighbor embeddings by mean
        neigh_sum = torch.zeros_like(x)                    # [N, H]
        neigh_sum.index_add_(0, dst, x[src])              # sum over incoming neighbors

        deg = torch.zeros(x.size(0), device=x.device)     # [N]
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)            # [N, 1]

        neigh_mean = neigh_sum / deg                      # [N, H]

        out = self.lin_self(x) + self.lin_neigh(neigh_mean)
        return F.relu(out)

    
# gnn64
class EventHistoryPointerHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.init_query = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.glimpse_candidate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.glimpse_query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.glimpse_v = nn.Linear(hidden_dim, 1, bias=False)

        self.query_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pointer_candidate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pointer_query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pointer_v = nn.Linear(hidden_dim, 1, bias=False)

    def _attention_logits(self, candidate_x, query, cand_proj, query_proj, v):
        cand_term = cand_proj(candidate_x)
        query_term = query_proj(query).unsqueeze(0)
        return v(torch.tanh(cand_term + query_term)).squeeze(-1)

    def forward(
        self,
        candidate_x,
        passer_x,
        graph_context,
        event_context_x,
        history_context,
        history_meta,
    ):
        query_input = torch.cat(
            [
                passer_x,
                graph_context,
                event_context_x,
                history_context,
                history_meta,
            ],
            dim=-1,
        )

        query = self.init_query(query_input)

        glimpse_logits = self._attention_logits(
            candidate_x,
            query,
            self.glimpse_candidate_proj,
            self.glimpse_query_proj,
            self.glimpse_v,
        )

        glimpse_alpha = torch.softmax(glimpse_logits, dim=0)
        glimpse = (glimpse_alpha.unsqueeze(-1) * candidate_x).sum(dim=0)

        refined_query = self.query_update(
            torch.cat([query, glimpse], dim=-1)
        )

        pointer_logits = self._attention_logits(
            candidate_x,
            refined_query,
            self.pointer_candidate_proj,
            self.pointer_query_proj,
            self.pointer_v,
        )

        return pointer_logits

    



class EventContextAndDynamicHistoryGatedCrossAttentionGraphSAGE(nn.Module):
    def __init__(
        self,
        node_dim=13,
        context_dim=134,
        history_dim=42,
        hidden_dim=64,
        time_delta_idx=None,
        num_heads=4,
    ):
        super().__init__()

        if time_delta_idx is None:
            raise ValueError("time_delta_idx must be provided.")

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.time_delta_idx = time_delta_idx

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.sage = GraphSAGELayer(hidden_dim)

        self.event_context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.context_value_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.context_feature_embedding = nn.Embedding(context_dim, hidden_dim)

        self.candidate_context_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.context_norm = nn.LayerNorm(hidden_dim)

        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.time_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.history_query = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.history_key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.history_query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.history_v = nn.Linear(hidden_dim, 1, bias=False)

        self.candidate_history_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.history_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.history_norm = nn.LayerNorm(hidden_dim)

        self.pointer_head = EventHistoryPointerHead(hidden_dim)

    def encode_event_context_tokens(self, sample):
        context_values = sample.event_context.unsqueeze(-1)
        context_tokens = self.context_value_encoder(context_values)

        feature_ids = torch.arange(
            self.context_dim,
            device=sample.event_context.device,
        )

        context_tokens = context_tokens + self.context_feature_embedding(feature_ids)

        return context_tokens

    def apply_candidate_context_attention(self, candidate_x, context_tokens):
        context_attended, _ = self.candidate_context_attn(
            query=candidate_x.unsqueeze(0),
            key=context_tokens.unsqueeze(0),
            value=context_tokens.unsqueeze(0),
        )

        context_attended = context_attended.squeeze(0)

        gate = self.context_gate(
            torch.cat([candidate_x, context_attended], dim=-1)
        )

        candidate_x = self.context_norm(
            candidate_x + gate * context_attended
        )

        return candidate_x

    def _temporal_features(self, history):
        L = history.shape[0]

        time_delta = history[:, self.time_delta_idx]
        time_delta = time_delta - time_delta.min()
        time_delta = time_delta / (time_delta.max() + 1e-6)

        if L == 1:
            recency_pos = torch.zeros(
                1,
                device=history.device,
                dtype=history.dtype,
            )
        else:
            recency_pos = torch.arange(
                L - 1,
                -1,
                -1,
                device=history.device,
                dtype=history.dtype,
            )
            recency_pos = recency_pos / (L - 1)

        return torch.stack([time_delta, recency_pos], dim=-1)

    def encode_history_tokens(self, sample):
        history = sample.possession_history

        if history.shape[0] == 0:
            return torch.zeros(
                0,
                self.hidden_dim,
                device=sample.node_features.device,
            )

        history_x = self.history_encoder(history)
        temporal_feat = self._temporal_features(history)
        time_x = self.time_encoder(temporal_feat)

        return history_x + time_x

    def pool_history_context(
        self,
        history_x,
        passer_x,
        graph_context,
        event_context_x,
        history_meta,
    ):
        if history_x.shape[0] == 0:
            return torch.zeros(
                self.hidden_dim,
                device=passer_x.device,
            )

        query = self.history_query(
            torch.cat(
                [
                    passer_x,
                    graph_context,
                    event_context_x,
                    history_meta,
                ],
                dim=-1,
            )
        )

        hist_term = self.history_key_proj(history_x)
        query_term = self.history_query_proj(query).unsqueeze(0)

        logits = self.history_v(
            torch.tanh(hist_term + query_term)
        ).squeeze(-1)

        alpha = torch.softmax(logits, dim=0)

        return (alpha.unsqueeze(-1) * history_x).sum(dim=0)

    def apply_candidate_history_attention(self, candidate_x, history_x):
        if history_x.shape[0] == 0:
            return candidate_x

        history_attended, _ = self.candidate_history_attn(
            query=candidate_x.unsqueeze(0),
            key=history_x.unsqueeze(0),
            value=history_x.unsqueeze(0),
        )

        history_attended = history_attended.squeeze(0)

        gate = self.history_gate(
            torch.cat([candidate_x, history_attended], dim=-1)
        )

        candidate_x = self.history_norm(
            candidate_x + gate * history_attended
        )

        return candidate_x

    def forward(self, sample):
        x = self.node_encoder(sample.node_features)
        x = self.sage(x, sample.edge_index)

        candidate_x = x[sample.candidate_mask]
        passer_x = x[0]
        graph_context = x.mean(dim=0)

        event_context_x = self.event_context_encoder(sample.event_context)

        context_tokens = self.encode_event_context_tokens(sample)

        candidate_x = self.apply_candidate_context_attention(
            candidate_x=candidate_x,
            context_tokens=context_tokens,
        )

        history_meta = torch.cat(
            [
                sample.has_history,
                sample.stored_history_len_norm,
            ],
            dim=-1,
        )

        history_x = self.encode_history_tokens(sample)

        history_context = self.pool_history_context(
            history_x=history_x,
            passer_x=passer_x,
            graph_context=graph_context,
            event_context_x=event_context_x,
            history_meta=history_meta,
        )

        candidate_x = self.apply_candidate_history_attention(
            candidate_x=candidate_x,
            history_x=history_x,
        )

        scores = self.pointer_head(
            candidate_x=candidate_x,
            passer_x=passer_x,
            graph_context=graph_context,
            event_context_x=event_context_x,
            history_context=history_context,
            history_meta=history_meta,
        )

        return scores
    



    
# gnn73
class DualHistoryPointerHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.init_query = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.glimpse_candidate_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.glimpse_query_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.glimpse_v = nn.Linear(
            hidden_dim,
            1,
            bias=False,
        )

        self.query_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pointer_candidate_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.pointer_query_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.pointer_v = nn.Linear(
            hidden_dim,
            1,
            bias=False,
        )

    def _attention_logits(
        self,
        candidate_x,
        query,
        candidate_proj,
        query_proj,
        score_layer,
    ):
        candidate_term = candidate_proj(candidate_x)
        query_term = query_proj(query).unsqueeze(0)

        return score_layer(
            torch.tanh(candidate_term + query_term)
        ).squeeze(-1)

    def forward(
        self,
        candidate_x,
        passer_x,
        graph_context,
        event_context_x,
        all_history_context,
        pass_history_context,
        dual_history_meta,
    ):
        query = self.init_query(
            torch.cat(
                [
                    passer_x,
                    graph_context,
                    event_context_x,
                    all_history_context,
                    pass_history_context,
                    dual_history_meta,
                ],
                dim=-1,
            )
        )

        glimpse_logits = self._attention_logits(
            candidate_x=candidate_x,
            query=query,
            candidate_proj=self.glimpse_candidate_proj,
            query_proj=self.glimpse_query_proj,
            score_layer=self.glimpse_v,
        )

        glimpse_alpha = torch.softmax(glimpse_logits, dim=0)

        glimpse = (
            glimpse_alpha.unsqueeze(-1) * candidate_x
        ).sum(dim=0)

        refined_query = self.query_update(
            torch.cat([query, glimpse], dim=-1)
        )

        scores = self._attention_logits(
            candidate_x=candidate_x,
            query=refined_query,
            candidate_proj=self.pointer_candidate_proj,
            query_proj=self.pointer_query_proj,
            score_layer=self.pointer_v,
        )

        return scores
    



# gnn75
class ContextualizedPassHistoryHierarchicalAttentionGraphSAGE(
    EventContextAndDynamicHistoryGatedCrossAttentionGraphSAGE
):
    def __init__(
        self,
        node_dim=13,
        context_dim=134,
        history_dim=42,
        hidden_dim=64,
        time_delta_idx=None,
        num_heads=4,
        temporal_layers=1,
        dropout=0.1,
    ):
        super().__init__(
            node_dim=node_dim,
            context_dim=context_dim,
            history_dim=history_dim,
            hidden_dim=hidden_dim,
            time_delta_idx=time_delta_idx,
            num_heads=num_heads,
        )

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )

        self.history_temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=temporal_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        self.pass_history_query = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pass_history_key_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.pass_history_query_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.pass_history_v = nn.Linear(
            hidden_dim,
            1,
            bias=False,
        )

        self.candidate_pass_history_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.pass_history_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.pass_history_norm = nn.LayerNorm(hidden_dim)

        self.all_history_scale = nn.Parameter(torch.tensor(0.0))
        self.pass_history_scale = nn.Parameter(torch.tensor(0.0))

        self.dual_history_pointer_head = DualHistoryPointerHead(hidden_dim)

    def encode_contextualized_history_tokens(self, sample):
        history = sample.possession_history

        if history.shape[0] == 0:
            return torch.zeros(
                0,
                self.hidden_dim,
                device=sample.node_features.device,
            )

        history_x = self.history_encoder(history)

        temporal_features = self._temporal_features(history)
        history_x = history_x + self.time_encoder(temporal_features)

        contextualized_history_x = self.history_temporal_encoder(
            history_x.unsqueeze(0)
        ).squeeze(0)

        return contextualized_history_x

    def pool_pass_history_context(
        self,
        pass_history_x,
        passer_x,
        graph_context,
        event_context_x,
        dual_history_meta,
    ):
        if pass_history_x.shape[0] == 0:
            return torch.zeros(
                self.hidden_dim,
                device=passer_x.device,
            )

        query = self.pass_history_query(
            torch.cat(
                [
                    passer_x,
                    graph_context,
                    event_context_x,
                    dual_history_meta,
                ],
                dim=-1,
            )
        )

        history_term = self.pass_history_key_proj(pass_history_x)
        query_term = self.pass_history_query_proj(query).unsqueeze(0)

        logits = self.pass_history_v(
            torch.tanh(history_term + query_term)
        ).squeeze(-1)

        alpha = torch.softmax(logits, dim=0)

        return (
            alpha.unsqueeze(-1) * pass_history_x
        ).sum(dim=0)

    def apply_candidate_pass_history_attention(
        self,
        candidate_x,
        pass_history_x,
    ):
        if pass_history_x.shape[0] == 0:
            return candidate_x

        pass_attended, _ = self.candidate_pass_history_attn(
            query=candidate_x.unsqueeze(0),
            key=pass_history_x.unsqueeze(0),
            value=pass_history_x.unsqueeze(0),
        )

        pass_attended = pass_attended.squeeze(0)

        gate = self.pass_history_gate(
            torch.cat(
                [
                    candidate_x,
                    pass_attended,
                ],
                dim=-1,
            )
        )

        branch_scale = torch.tanh(self.pass_history_scale)

        return self.pass_history_norm(
            candidate_x
            + branch_scale * gate * pass_attended
        )

    def forward(self, sample):
        # Graph backbone.
        x = self.node_encoder(sample.node_features)
        x = self.sage(x, sample.edge_index)

        candidate_x = x[sample.candidate_mask]
        passer_x = x[0]
        graph_context = x.mean(dim=0)

        # Event-context branch, unchanged from gnn73.
        event_context_x = self.event_context_encoder(
            sample.event_context
        )

        context_tokens = self.encode_event_context_tokens(sample)

        candidate_x = self.apply_candidate_context_attention(
            candidate_x=candidate_x,
            context_tokens=context_tokens,
        )

        all_history_meta = torch.cat(
            [
                sample.has_history,
                sample.stored_history_len_norm,
            ],
            dim=-1,
        )

        pass_history_meta = torch.cat(
            [
                sample.has_pass_history,
                sample.pass_history_len_norm,
            ],
            dim=-1,
        )

        dual_history_meta = torch.cat(
            [
                all_history_meta,
                pass_history_meta,
            ],
            dim=-1,
        )

        # Full history sequence is first contextualized jointly.
        contextualized_history_x = (
            self.encode_contextualized_history_tokens(sample)
        )

        all_history_context = self.pool_history_context(
            history_x=contextualized_history_x,
            passer_x=passer_x,
            graph_context=graph_context,
            event_context_x=event_context_x,
            history_meta=all_history_meta,
        )

        all_history_context = (
            torch.tanh(self.all_history_scale)
            * all_history_context
        )

        # Historical pass tokens are extracted after contextualization.
        if contextualized_history_x.shape[0] == 0:
            contextualized_pass_x = torch.zeros(
                0,
                self.hidden_dim,
                device=sample.node_features.device,
            )
        else:
            contextualized_pass_x = contextualized_history_x[
                sample.history_pass_mask
            ]

        pass_history_context = self.pool_pass_history_context(
            pass_history_x=contextualized_pass_x,
            passer_x=passer_x,
            graph_context=graph_context,
            event_context_x=event_context_x,
            dual_history_meta=dual_history_meta,
        )

        pass_history_context = (
            torch.tanh(self.pass_history_scale)
            * pass_history_context
        )

        candidate_x = self.apply_candidate_pass_history_attention(
            candidate_x=candidate_x,
            pass_history_x=contextualized_pass_x,
        )

        scores = self.dual_history_pointer_head(
            candidate_x=candidate_x,
            passer_x=passer_x,
            graph_context=graph_context,
            event_context_x=event_context_x,
            all_history_context=all_history_context,
            pass_history_context=pass_history_context,
            dual_history_meta=dual_history_meta,
        )

        return scores
    