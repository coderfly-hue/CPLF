#!/usr/bin/env python3
"""
CPLF-LLM: complete executable reference implementation for a protocols.io
computational protocol.

Scope
-----
This file implements the full computational workflow described by the manuscript:
1) standardized learner-interaction loading and fixed learner-level splitting,
2) cognition-aware learner profiling,
3) dynamic profile memory,
4) next-step correctness prediction,
5) concept-gap construction,
6) pedagogical-resource retrieval,
7) Qwen2.5-7B-Instruct + LoRA feedback generation,
8) same-backbone Direct LLM control,
9) independent rubric-based feedback verification,
10) FQS computation,
11) human-evaluation reliability utilities,
12) paired cluster bootstrap and Holm correction,
13) export of machine-readable results.

Important implementation note
-----------------------------
The manuscript defines the temporal Encoder_theta abstractly rather than naming
a specific neural cell. To make this reference protocol executable, this file
instantiates Encoder_theta as a single-layer GRU. All other explicitly reported
settings are implemented directly. If an author's archived implementation uses
a different Encoder_theta, that archived source should be treated as the
authoritative implementation.

The script expects standardized tabular data rather than redistributing the six
third-party datasets. A standardized interaction table uses these columns:

student_id, exercise_id, concepts, correct, timestamp,
time_spent, attempts, hints, interval,
question_text, response_text, reference_feedback, misconception_label

Only student_id, exercise_id, concepts, correct, and timestamp are required for
prediction. Text columns are used by feedback generation when present.

"concepts" is a "|" separated string, for example:
algebra|linear_equation

This reference implementation is designed to be uploaded as an attachment to
protocols.io alongside the protocol text.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

# Heavy packages are imported lazily where practical.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    log_loss,
    brier_score_loss,
)
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Reproducible constants
# ---------------------------------------------------------------------------

SEEDS = [2024, 2025, 2026, 2027, 2028]

QWEN_REPO = "Qwen/Qwen2.5-7B-Instruct"
QWEN_REVISION = "acbd96531cda22292a3ceaa67e984955d3965282"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EVALUATOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

DATASET_SOURCES = {
    "OULAD": "https://analyse.kmi.open.ac.uk/open_dataset",
    "ASSIST2009": (
        "https://sites.google.com/site/assistmentsdata/home/"
        "2009-2010-assistment-data/skill-builder-data-2009-2010"
    ),
    "EdNet": "https://github.com/riiid/ednet",
    "MathDial": "https://github.com/eth-nlped/mathdial",
    "CodeWorkout": "https://pslcdatashop.web.cmu.edu/Files?datasetId=3458",
    "CodeStream": "https://data.mendeley.com/datasets/n77t7z9zcr/1",
}

DEFAULT_CONFIG = {
    "prediction": {
        "hidden_dim": 128,
        "concept_embedding_dim": 128,
        "exercise_embedding_dim": 64,
        "correct_embedding_dim": 16,
        "aux_embedding_dim": 16,
        "time_embedding_dim": 16,
        "engagement_dim": 16,
        "uncertainty_dim": 16,
        "misconception_classes": 8,
        "max_sequence_length": 200,
        "dropout": 0.2,
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "epochs": 50,
        "early_stopping_patience": 10,
        "ece_bins": 10,
    },
    "profile": {
        "mastery_threshold_gamma": 0.70,
    },
    "retrieval": {
        "top_k": 5,
        "lambda_semantic": 0.50,
        "lambda_gap": 0.35,
        "lambda_difficulty": 0.15,
    },
    "generator": {
        "repo_id": QWEN_REPO,
        "revision": QWEN_REVISION,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "epochs": 3,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "effective_batch_size": 32,
        "train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "max_sequence_length": 2048,
        "temperature": 0.3,
        "top_p": 0.90,
        "repetition_penalty": 1.05,
        "max_new_tokens": 256,
    },
    "verification": {
        "evaluator_model": EVALUATOR_MODEL,
        "temperature": 0.0,
        "max_new_tokens": 256,
        "eta": 0.70,
        "delta": 0.20,
        "weights": {
            "relevance": 0.25,
            "diagnostic": 0.25,
            "actionability": 0.25,
            "personalization": 0.25,
            "leakage_penalty": 0.40,
        },
    },
    "statistics": {
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 2024,
    },
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_config(path: Optional[str]) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is None:
        return cfg
    import yaml
    user_cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    deep_update(cfg, user_cfg)
    return cfg


def deep_update(base: dict, update: dict) -> dict:
    for k, v in (update or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ---------------------------------------------------------------------------
# Standardized interaction schema and split management
# ---------------------------------------------------------------------------

REQUIRED_INTERACTION_COLUMNS = [
    "student_id", "exercise_id", "concepts", "correct", "timestamp"
]

OPTIONAL_DEFAULTS = {
    "time_spent": 0.0,
    "attempts": 1.0,
    "hints": 0.0,
    "interval": 0.0,
    "question_text": "",
    "response_text": "",
    "reference_feedback": "",
    "misconception_label": -1,
}


def normalize_interaction_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_INTERACTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Standardized interaction file is missing required columns: "
            + ", ".join(missing)
        )
    out = df.copy()
    for c, default in OPTIONAL_DEFAULTS.items():
        if c not in out.columns:
            out[c] = default

    out["student_id"] = out["student_id"].astype(str)
    out["exercise_id"] = out["exercise_id"].astype(str)
    out["concepts"] = out["concepts"].fillna("").astype(str)
    out["correct"] = pd.to_numeric(out["correct"], errors="coerce").fillna(0).astype(int)
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(0).astype(np.int64)

    for c in ["time_spent", "attempts", "hints", "interval"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(OPTIONAL_DEFAULTS[c]).astype(float)

    out["misconception_label"] = (
        pd.to_numeric(out["misconception_label"], errors="coerce")
        .fillna(-1).astype(int)
    )
    out = out.sort_values(["student_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    return out


def make_fixed_student_split(
    df: pd.DataFrame,
    output_dir: str | Path,
    seed: int = 2024,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> dict[str, list[str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    students = np.array(sorted(df["student_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(students)

    n = len(students)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_ids = students[:n_train]
    val_ids = students[n_train:n_train + n_val]
    test_ids = students[n_train + n_val:]

    split = {
        "train": train_ids.tolist(),
        "validation": val_ids.tolist(),
        "test": test_ids.tolist(),
    }
    for name, ids in split.items():
        (output_dir / f"{name}_ids.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8"
        )

    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))
    return split


def load_split_ids(split_dir: str | Path) -> dict[str, set[str]]:
    split_dir = Path(split_dir)
    result = {}
    for name in ["train", "validation", "test"]:
        path = split_dir / f"{name}_ids.txt"
        result[name] = {
            x.strip() for x in path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        }
    return result


def apply_split(df: pd.DataFrame, split_ids: dict[str, set[str]]) -> dict[str, pd.DataFrame]:
    out = {}
    for name, ids in split_ids.items():
        out[name] = df[df["student_id"].isin(ids)].copy().reset_index(drop=True)
    return out


def dataset_statistics(df: pd.DataFrame) -> dict[str, int]:
    concepts = set()
    for x in df["concepts"].astype(str):
        concepts.update([c for c in x.split("|") if c])
    return {
        "learners": int(df["student_id"].nunique()),
        "interactions": int(len(df)),
        "tasks": int(df["exercise_id"].nunique()),
        "concepts": int(len(concepts)),
    }


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Vocabulary:
    exercise_to_id: dict[str, int]
    concept_to_id: dict[str, int]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Vocabulary":
        exercises = sorted(df["exercise_id"].astype(str).unique())
        concepts = sorted({
            c
            for value in df["concepts"].astype(str)
            for c in value.split("|")
            if c
        })
        return cls(
            exercise_to_id={x: i + 1 for i, x in enumerate(exercises)},  # 0=UNK/PAD
            concept_to_id={x: i + 1 for i, x in enumerate(concepts)},
        )

    @property
    def n_exercises(self) -> int:
        return len(self.exercise_to_id) + 1

    @property
    def n_concepts(self) -> int:
        return len(self.concept_to_id) + 1

    def encode_exercise(self, x: str) -> int:
        return self.exercise_to_id.get(str(x), 0)

    def encode_concepts(self, x: str) -> list[int]:
        ids = [
            self.concept_to_id.get(c, 0)
            for c in str(x).split("|")
            if c
        ]
        return ids or [0]

    def save(self, path: str | Path) -> None:
        save_json(dataclasses.asdict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**obj)


# ---------------------------------------------------------------------------
# Sequence dataset
# ---------------------------------------------------------------------------

class StudentSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocabulary,
        max_sequence_length: int = 200,
        max_concepts_per_item: int = 8,
    ):
        self.samples = []
        self.vocab = vocab
        self.max_sequence_length = max_sequence_length
        self.max_concepts_per_item = max_concepts_per_item

        for sid, g in df.groupby("student_id", sort=False):
            g = g.sort_values("timestamp")
            if len(g) < 2:
                continue
            rows = g.to_dict("records")
            # Consecutive chunks preserve chronology and avoid leakage.
            for start in range(0, len(rows), max_sequence_length):
                chunk = rows[start:start + max_sequence_length]
                if len(chunk) >= 2:
                    self.samples.append((str(sid), chunk))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        student_id, rows = self.samples[idx]
        T = len(rows)
        exercise = np.array(
            [self.vocab.encode_exercise(r["exercise_id"]) for r in rows],
            dtype=np.int64,
        )

        concepts = np.zeros((T, self.max_concepts_per_item), dtype=np.int64)
        for t, r in enumerate(rows):
            ids = self.vocab.encode_concepts(r["concepts"])[:self.max_concepts_per_item]
            concepts[t, :len(ids)] = ids

        correct = np.array([int(r["correct"]) for r in rows], dtype=np.int64)
        aux = np.array(
            [[r["time_spent"], r["attempts"], r["hints"], r["interval"]] for r in rows],
            dtype=np.float32,
        )
        timestamp = np.array([r["timestamp"] for r in rows], dtype=np.float32)
        # Relative log-time feature avoids huge raw timestamp magnitudes.
        timestamp = np.log1p(np.maximum(0.0, timestamp - timestamp[0]))[:, None]

        misconception = np.array(
            [int(r.get("misconception_label", -1)) for r in rows],
            dtype=np.int64,
        )

        return {
            "student_id": student_id,
            "exercise": torch.from_numpy(exercise),
            "concepts": torch.from_numpy(concepts),
            "correct": torch.from_numpy(correct),
            "aux": torch.from_numpy(aux),
            "time": torch.from_numpy(timestamp),
            "misconception": torch.from_numpy(misconception),
            "length": T,
        }


def collate_sequences(batch: list[dict]) -> dict:
    B = len(batch)
    max_t = max(x["length"] for x in batch)
    max_c = max(x["concepts"].shape[1] for x in batch)

    exercise = torch.zeros(B, max_t, dtype=torch.long)
    concepts = torch.zeros(B, max_t, max_c, dtype=torch.long)
    correct = torch.zeros(B, max_t, dtype=torch.long)
    aux = torch.zeros(B, max_t, 4, dtype=torch.float32)
    time = torch.zeros(B, max_t, 1, dtype=torch.float32)
    misconception = torch.full((B, max_t), -1, dtype=torch.long)
    mask = torch.zeros(B, max_t, dtype=torch.bool)

    student_ids = []
    lengths = []
    for i, item in enumerate(batch):
        t = item["length"]
        exercise[i, :t] = item["exercise"]
        concepts[i, :t] = item["concepts"]
        correct[i, :t] = item["correct"]
        aux[i, :t] = item["aux"]
        time[i, :t] = item["time"]
        misconception[i, :t] = item["misconception"]
        mask[i, :t] = True
        student_ids.append(item["student_id"])
        lengths.append(t)

    return {
        "student_id": student_ids,
        "exercise": exercise,
        "concepts": concepts,
        "correct": correct,
        "aux": aux,
        "time": time,
        "misconception": misconception,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# CPLF learner model
# ---------------------------------------------------------------------------

class InteractionEncoder(nn.Module):
    def __init__(self, vocab: Vocabulary, cfg: dict):
        super().__init__()
        self.exercise_emb = nn.Embedding(
            vocab.n_exercises, cfg["exercise_embedding_dim"], padding_idx=0
        )
        self.concept_emb = nn.Embedding(
            vocab.n_concepts, cfg["concept_embedding_dim"], padding_idx=0
        )
        self.correct_emb = nn.Embedding(2, cfg["correct_embedding_dim"])
        self.aux_proj = nn.Sequential(
            nn.Linear(4, cfg["aux_embedding_dim"]),
            nn.ReLU(),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(1, cfg["time_embedding_dim"]),
            nn.ReLU(),
        )

        in_dim = (
            cfg["exercise_embedding_dim"]
            + cfg["concept_embedding_dim"]
            + cfg["correct_embedding_dim"]
            + cfg["aux_embedding_dim"]
            + cfg["time_embedding_dim"]
        )
        self.proj = nn.Sequential(
            nn.Linear(in_dim, cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
        )

    def aggregate_concepts(self, concept_ids: torch.Tensor) -> torch.Tensor:
        emb = self.concept_emb(concept_ids)  # [..., C, D]
        mask = (concept_ids != 0).unsqueeze(-1)
        denom = mask.sum(dim=-2).clamp_min(1)
        return (emb * mask).sum(dim=-2) / denom

    def forward(
        self,
        exercise: torch.Tensor,
        concepts: torch.Tensor,
        correct: torch.Tensor,
        aux: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        q = self.exercise_emb(exercise)
        c = self.aggregate_concepts(concepts)
        r = self.correct_emb(correct.clamp(0, 1))
        a = self.aux_proj(aux)
        tau = self.time_proj(time)
        return self.proj(torch.cat([q, c, r, a, tau], dim=-1))


class CPLFModel(nn.Module):
    def __init__(self, vocab: Vocabulary, cfg: dict):
        super().__init__()
        self.vocab = vocab
        self.cfg = cfg
        h = cfg["hidden_dim"]
        self.encoder = InteractionEncoder(vocab, cfg)
        self.temporal = nn.GRU(h, h, batch_first=True)

        self.mastery_head = nn.Linear(h, vocab.n_concepts)
        self.misconception_head = nn.Linear(
            h + cfg["exercise_embedding_dim"] + cfg["concept_embedding_dim"] + cfg["correct_embedding_dim"],
            cfg["misconception_classes"],
        )
        self.engagement_head = nn.Linear(h + 4, cfg["engagement_dim"])
        self.uncertainty_head = nn.Linear(h, cfg["uncertainty_dim"])

        self.profile_dim = (
            vocab.n_concepts
            + cfg["misconception_classes"]
            + cfg["engagement_dim"]
            + cfg["uncertainty_dim"]
        )
        self.profile_project = nn.Sequential(
            nn.Linear(self.profile_dim, h),
            nn.Tanh(),
        )
        self.memory_gate = nn.Linear(h + h + h, h)

        pred_in = h + cfg["exercise_embedding_dim"] + cfg["concept_embedding_dim"]
        self.pred_head = nn.Linear(pred_in, 1)

    def profile_components(
        self,
        hidden: torch.Tensor,
        exercise: torch.Tensor,
        concepts: torch.Tensor,
        correct: torch.Tensor,
        aux: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        q = self.encoder.exercise_emb(exercise)
        c = self.encoder.aggregate_concepts(concepts)
        r = self.encoder.correct_emb(correct.clamp(0, 1))

        mastery = torch.sigmoid(self.mastery_head(hidden))
        misconception = torch.softmax(
            self.misconception_head(torch.cat([hidden, q, c, r], dim=-1)),
            dim=-1,
        )
        engagement = torch.sigmoid(
            self.engagement_head(torch.cat([hidden, aux], dim=-1))
        )
        uncertainty = torch.nn.functional.softplus(
            self.uncertainty_head(hidden)
        )
        profile = torch.cat(
            [mastery, misconception, engagement, uncertainty], dim=-1
        )
        return {
            "mastery": mastery,
            "misconception": misconception,
            "engagement": engagement,
            "uncertainty": uncertainty,
            "profile": profile,
        }

    def dynamic_memory(
        self,
        profile: torch.Tensor,
        interaction_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.profile_project(profile)
        B, T, H = projected.shape
        prev = torch.zeros(B, H, device=profile.device)
        outputs = []
        for t in range(T):
            cur = projected[:, t]
            e = interaction_emb[:, t]
            gate = torch.sigmoid(self.memory_gate(torch.cat([prev, cur, e], dim=-1)))
            updated = gate * cur + (1.0 - gate) * prev
            valid = mask[:, t].float().unsqueeze(-1)
            prev = valid * updated + (1.0 - valid) * prev
            outputs.append(prev)
        return torch.stack(outputs, dim=1)

    def forward(self, batch: dict) -> dict:
        e = self.encoder(
            batch["exercise"],
            batch["concepts"],
            batch["correct"],
            batch["aux"],
            batch["time"],
        )
        h, _ = self.temporal(e)
        comp = self.profile_components(
            h,
            batch["exercise"],
            batch["concepts"],
            batch["correct"],
            batch["aux"],
        )
        memory = self.dynamic_memory(comp["profile"], e, batch["mask"])

        # Predict next response: memory at t predicts interaction at t+1.
        next_q = self.encoder.exercise_emb(batch["exercise"][:, 1:])
        next_c = self.encoder.aggregate_concepts(batch["concepts"][:, 1:])
        pred_x = torch.cat([memory[:, :-1], next_q, next_c], dim=-1)
        logits = self.pred_head(pred_x).squeeze(-1)
        probs = torch.sigmoid(logits)
        target = batch["correct"][:, 1:].float()
        target_mask = batch["mask"][:, 1:] & batch["mask"][:, :-1]

        return {
            **comp,
            "memory": memory,
            "logits": logits,
            "probs": probs,
            "target": target,
            "target_mask": target_mask,
        }


def prediction_loss(
    output: dict,
    batch: dict,
    lambda_mis: float = 0.2,
    lambda_unc: float = 0.1,
) -> torch.Tensor:
    mask = output["target_mask"]
    logits = output["logits"][mask]
    target = output["target"][mask]
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target)

    # Misconception classification when labels are available.
    labels = batch["misconception"]
    valid_mis = (labels >= 0) & batch["mask"]
    if valid_mis.any():
        pred = output["misconception"][valid_mis]
        y = labels[valid_mis]
        loss = loss + lambda_mis * nn.functional.nll_loss(
            torch.log(pred.clamp_min(1e-8)), y
        )

    # Uncertainty regularization: align mean uncertainty with squared error.
    if mask.any():
        unc = output["uncertainty"][:, :-1].mean(dim=-1)[mask]
        err2 = (output["probs"][mask].detach() - target).pow(2)
        loss = loss + lambda_unc * nn.functional.mse_loss(unc, err2)

    return loss


# ---------------------------------------------------------------------------
# Predictor training and evaluation
# ---------------------------------------------------------------------------

def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (y_prob >= lo) & (y_prob <= hi)
        else:
            m = (y_prob >= lo) & (y_prob < hi)
        if not m.any():
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += m.mean() * abs(acc - conf)
    return float(ece)


def binary_metrics(y_true, y_prob, ece_bins=10) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "nll": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob, ece_bins),
    }


@torch.no_grad()
def collect_predictions(
    model: CPLFModel,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        raw_student_ids = batch["student_id"]
        batch = move_batch(batch, device)
        out = model(batch)
        probs = out["probs"].detach().cpu().numpy()
        targets = out["target"].detach().cpu().numpy()
        mask = out["target_mask"].detach().cpu().numpy()
        mastery = out["mastery"].detach().cpu().numpy()

        for b, sid in enumerate(raw_student_ids):
            valid_positions = np.where(mask[b])[0]
            for t in valid_positions:
                rows.append({
                    "student_id": sid,
                    "step": int(t + 1),
                    "y_true": int(targets[b, t]),
                    "y_prob": float(probs[b, t]),
                    "mean_mastery": float(mastery[b, t].mean()),
                })
    return pd.DataFrame(rows)


def train_predictor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    output_dir: str | Path,
    seed: int,
) -> dict:
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p_cfg = cfg["prediction"]
    vocab = Vocabulary.fit(train_df)
    vocab.save(output_dir / "vocab.json")

    train_ds = StudentSequenceDataset(
        train_df, vocab, p_cfg["max_sequence_length"]
    )
    val_ds = StudentSequenceDataset(
        val_df, vocab, p_cfg["max_sequence_length"]
    )
    test_ds = StudentSequenceDataset(
        test_df, vocab, p_cfg["max_sequence_length"]
    )

    train_loader = DataLoader(
        train_ds, batch_size=p_cfg["batch_size"], shuffle=True,
        collate_fn=collate_sequences
    )
    val_loader = DataLoader(
        val_ds, batch_size=p_cfg["batch_size"], shuffle=False,
        collate_fn=collate_sequences
    )
    test_loader = DataLoader(
        test_ds, batch_size=p_cfg["batch_size"], shuffle=False,
        collate_fn=collate_sequences
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CPLFModel(vocab, p_cfg).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=p_cfg["learning_rate"],
        weight_decay=p_cfg["weight_decay"],
    )

    best_auc = -np.inf
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, p_cfg["epochs"] + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            out = model(batch)
            loss = prediction_loss(out, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_pred = collect_predictions(model, val_loader, device)
        val_metrics = binary_metrics(
            val_pred["y_true"], val_pred["y_prob"], p_cfg["ece_bins"]
        )
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else np.nan,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["auc"] > best_auc + 1e-6:
            best_auc = val_metrics["auc"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= p_cfg["early_stopping_patience"]:
                break

    if best_state is None:
        raise RuntimeError("Predictor training produced no checkpoint.")

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "config": p_cfg,
            "vocab": dataclasses.asdict(vocab),
            "seed": seed,
        },
        output_dir / f"predictor_seed_{seed}.pt",
    )

    test_pred = collect_predictions(model, test_loader, device)
    test_pred["seed"] = seed
    write_table(test_pred, output_dir / f"test_predictions_seed_{seed}.csv")

    metrics = binary_metrics(
        test_pred["y_true"], test_pred["y_prob"], p_cfg["ece_bins"]
    )
    metrics["seed"] = seed
    pd.DataFrame(history).to_csv(
        output_dir / f"training_history_seed_{seed}.csv", index=False
    )
    save_json(metrics, output_dir / f"test_metrics_seed_{seed}.json")
    return metrics


# ---------------------------------------------------------------------------
# Profile validity utilities
# ---------------------------------------------------------------------------

def doa_score(df: pd.DataFrame) -> float:
    """
    Expected columns:
      student_id, concept_id, mastery, correctness
    DOA is averaged across concepts.
    """
    scores = []
    for _, g in df.groupby("concept_id"):
        per_student = (
            g.groupby("student_id")
            .agg(mastery=("mastery", "mean"), correctness=("correctness", "mean"))
            .reset_index()
        )
        num = 0
        den = 0
        vals = per_student.to_dict("records")
        for i in range(len(vals)):
            for j in range(len(vals)):
                if i == j:
                    continue
                if vals[i]["correctness"] == vals[j]["correctness"]:
                    continue
                den += 1
                if (
                    vals[i]["mastery"] > vals[j]["mastery"]
                    and vals[i]["correctness"] > vals[j]["correctness"]
                ):
                    num += 1
        if den > 0:
            scores.append(num / den)
    return float(np.mean(scores)) if scores else float("nan")


def mastery_spearman(df: pd.DataFrame) -> dict:
    """
    Expected columns:
      mastery, heldout_correctness
    """
    rho, p = spearmanr(
        df["mastery"].to_numpy(float),
        df["heldout_correctness"].to_numpy(float),
        nan_policy="omit",
    )
    return {"rho": float(rho), "p": float(p), "n": int(len(df))}


# ---------------------------------------------------------------------------
# Feedback plan and retrieval
# ---------------------------------------------------------------------------

def concept_gap(
    mastery_by_concept: dict[str, float],
    required_concepts: list[str],
    gamma: float = 0.70,
) -> dict[str, float]:
    return {
        c: max(0.0, gamma - float(mastery_by_concept.get(c, 0.0)))
        for c in required_concepts
    }


class PedagogicalRetriever:
    """
    Resource CSV/Parquet schema:
      resource_id, text, concepts, difficulty

    concepts: "|" separated
    difficulty: numeric in [0,1]
    """
    def __init__(
        self,
        resources: pd.DataFrame,
        embedding_model: str = EMBEDDING_MODEL,
        lambda_semantic: float = 0.50,
        lambda_gap: float = 0.35,
        lambda_difficulty: float = 0.15,
    ):
        from sentence_transformers import SentenceTransformer
        self.resources = resources.copy().reset_index(drop=True)
        self.model = SentenceTransformer(embedding_model)
        self.embeddings = self.model.encode(
            self.resources["text"].astype(str).tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.lambda_semantic = lambda_semantic
        self.lambda_gap = lambda_gap
        self.lambda_difficulty = lambda_difficulty

    def retrieve(
        self,
        query_text: str,
        gaps: dict[str, float],
        learner_ability: float,
        top_k: int = 5,
    ) -> list[dict]:
        q_emb = self.model.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        semantic = self.embeddings @ q_emb

        scores = []
        for i, row in self.resources.iterrows():
            rc = {x for x in str(row.get("concepts", "")).split("|") if x}
            gap_cov = sum(gaps.get(c, 0.0) for c in rc)
            difficulty = float(row.get("difficulty", 0.5))
            diff_mismatch = abs(difficulty - learner_ability)
            score = (
                self.lambda_semantic * float(semantic[i])
                + self.lambda_gap * gap_cov
                - self.lambda_difficulty * diff_mismatch
            )
            scores.append(score)

        idx = np.argsort(scores)[::-1][:top_k]
        out = []
        for i in idx:
            row = self.resources.iloc[int(i)].to_dict()
            row["retrieval_score"] = float(scores[int(i)])
            out.append(row)
        return out


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CPLF_SYSTEM_PROMPT = """You are an educational feedback assistant.
Your task is to help the learner make the next productive step without revealing
the final answer. Use the supplied learner profile only as evidence, not as a
label or judgment. Feedback must be concise, specific, actionable, and aligned
with the current task. Do not claim certainty when the profile is uncertain.
Do not reveal a complete solution or final answer."""

DIRECT_SYSTEM_PROMPT = """You are an educational feedback assistant.
Give concise, useful feedback on the learner's current response. Help the learner
make the next productive step without revealing the final answer or a complete
solution."""

EVALUATOR_SYSTEM_PROMPT = """You are an independent evaluator of educational
feedback. You do not know which method generated the feedback. Score the feedback
only using the supplied task, learner evidence, and rubric. Return valid JSON and
nothing else."""

REVISION_SYSTEM_PROMPT = """You revise educational feedback after a quality-control
failure. Correct the specified failed dimension while preserving useful parts of
the original feedback. Do not reveal the final answer or a complete solution."""


def build_cplf_prompt(case: dict) -> str:
    evidence_lines = []
    for r in case.get("retrieved_evidence", []):
        evidence_lines.append(
            f"- {r.get('text', '')}"
        )
    evidence = "\n".join(evidence_lines) if evidence_lines else "- None"

    gaps = case.get("concept_gaps", {})
    gap_text = ", ".join(
        f"{k}: {v:.2f}" for k, v in sorted(gaps.items(), key=lambda x: -x[1])
    ) or "none identified"

    misconceptions = case.get("misconceptions", [])
    mis_text = ", ".join(map(str, misconceptions)) or "none identified"

    return f"""TASK
{case.get('question_text', '')}

LEARNER RESPONSE
{case.get('response_text', '')}

RESPONSE CORRECTNESS
{case.get('correct', '')}

LEARNER PROFILE SUMMARY
Concept mastery: {case.get('mastery_summary', '')}
Engagement: {case.get('engagement_summary', '')}
Uncertainty: {case.get('uncertainty_summary', '')}

DIAGNOSED LEARNING GAP
{gap_text}

LIKELY MISCONCEPTION CUES
{mis_text}

RETRIEVED PEDAGOGICAL EVIDENCE
{evidence}

FEEDBACK CONSTRAINTS
1. Address the most important learning gap first.
2. Explain the likely reasoning issue without stigmatizing the learner.
3. Give one or two concrete next actions.
4. Do not reveal the final answer or a complete solution.
5. Keep the response within 150 words.

Write only the feedback to the learner."""


def build_direct_prompt(case: dict) -> str:
    return f"""TASK
{case.get('question_text', '')}

LEARNER RESPONSE
{case.get('response_text', '')}

RESPONSE CORRECTNESS
{case.get('correct', '')}

Give concise, actionable feedback that helps the learner improve the response.
Do not reveal the final answer or a complete solution. Keep the response within
150 words."""


def build_evaluator_prompt(case: dict, feedback: str) -> str:
    return f"""TASK
{case.get('question_text', '')}

LEARNER RESPONSE
{case.get('response_text', '')}

LEARNER PROFILE / DIAGNOSTIC EVIDENCE
{case.get('diagnostic_evidence', '')}

FEEDBACK TO SCORE
{feedback}

RUBRIC
Give an integer from 1 to 5 for each dimension.

diagnostic:
1 = contradicts or ignores the diagnosed learner difficulty
3 = partially aligned but generic or incomplete
5 = precisely aligned with the identified concept gap/misconception

actionability:
1 = no usable next step
3 = somewhat usable but vague
5 = concrete next step(s) the learner can immediately apply

personalization:
1 = generic and interchangeable across learners
3 = uses limited learner-specific information
5 = clearly tailored to this learner's current state without stereotyping

leakage:
1 = no answer/solution leakage
3 = provides substantial solution structure that may over-scaffold
5 = directly reveals the final answer or essentially complete solution

Return exactly:
{{"diagnostic": INT, "actionability": INT, "personalization": INT, "leakage": INT}}"""


def build_revision_prompt(
    case: dict,
    feedback: str,
    failed_dimensions: list[str],
) -> str:
    return f"""TASK
{case.get('question_text', '')}

LEARNER RESPONSE
{case.get('response_text', '')}

ORIGINAL FEEDBACK
{feedback}

FAILED QUALITY DIMENSION(S)
{", ".join(failed_dimensions)}

Revise the feedback so that it addresses the failed dimension(s), remains
specific and actionable, and does not reveal the final answer or a complete
solution. Return only the revised feedback."""


# ---------------------------------------------------------------------------
# Qwen generator + LoRA
# ---------------------------------------------------------------------------

class QwenFeedbackGenerator:
    def __init__(
        self,
        cfg: dict,
        adapter_path: Optional[str] = None,
        for_training: bool = False,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, PeftModel

        self.cfg = cfg
        repo = cfg["repo_id"]
        revision = cfg["revision"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            repo, revision=revision, trust_remote_code=True
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            repo,
            revision=revision,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        elif for_training:
            lora = LoraConfig(
                r=cfg["lora_rank"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg["lora_dropout"],
                target_modules=cfg["lora_target_modules"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=self.cfg["temperature"] > 0,
                temperature=max(self.cfg["temperature"], 1e-5),
                top_p=self.cfg["top_p"],
                repetition_penalty=self.cfg["repetition_penalty"],
                max_new_tokens=self.cfg["max_new_tokens"],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class FeedbackSFTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        direct: bool,
        max_length: int,
    ):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        system = DIRECT_SYSTEM_PROMPT if direct else CPLF_SYSTEM_PROMPT

        for _, row in df.iterrows():
            case = row.to_dict()
            user = build_direct_prompt(case) if direct else build_cplf_prompt(case)
            answer = str(row["reference_feedback"])
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            self.examples.append(text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.examples[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_lora_generator(
    training_file: str,
    output_dir: str,
    cfg: dict,
    direct: bool,
) -> None:
    from transformers import Trainer, TrainingArguments

    df = read_table(training_file)
    generator = QwenFeedbackGenerator(cfg, for_training=True)
    ds = FeedbackSFTDataset(
        df,
        generator.tokenizer,
        direct=direct,
        max_length=cfg["max_sequence_length"],
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        bf16=torch.cuda.is_available(),
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=generator.model, args=args, train_dataset=ds)
    trainer.train()
    generator.model.save_pretrained(output_dir)
    generator.tokenizer.save_pretrained(output_dir)


# ---------------------------------------------------------------------------
# Evaluator, relevance, verification and FQS
# ---------------------------------------------------------------------------

class FeedbackEvaluator:
    def __init__(self, cfg: dict):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["evaluator_model"], trust_remote_code=True
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["evaluator_model"],
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    def relevance(self, task_text: str, feedback: str) -> float:
        emb = self.embedder.encode(
            [task_text, feedback],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        cos = float(np.dot(emb[0], emb[1]))
        # Convert cosine [-1,1] to [0,1].
        return float(np.clip((cos + 1.0) / 2.0, 0.0, 1.0))

    def rubric(self, case: dict, feedback: str) -> dict[str, float]:
        prompt = build_evaluator_prompt(case, feedback)
        messages = [
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.cfg["max_new_tokens"],
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        obj = parse_json_object(raw)

        scores = {}
        for key in ["diagnostic", "actionability", "personalization", "leakage"]:
            val = int(obj[key])
            if not 1 <= val <= 5:
                raise ValueError(f"Evaluator score outside 1..5: {key}={val}")
            # Normalize rubric 1..5 to 0..1.
            scores[key] = (val - 1) / 4.0
        return scores


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError(f"Evaluator did not return JSON: {text}")
        return json.loads(m.group(0))


def fqs(
    relevance: float,
    diagnostic: float,
    actionability: float,
    personalization: float,
    leakage: float,
) -> float:
    value = (
        0.25 * relevance
        + 0.25 * diagnostic
        + 0.25 * actionability
        + 0.25 * personalization
        - 0.40 * leakage
    )
    return float(max(0.0, value))


def verification_score(scores: dict) -> float:
    return fqs(
        scores["relevance"],
        scores["diagnostic"],
        scores["actionability"],
        scores["personalization"],
        scores["leakage"],
    )


def failed_dimensions(scores: dict, eta: float, delta: float) -> list[str]:
    failed = []
    if scores["diagnostic"] < eta:
        failed.append("diagnostic consistency")
    if scores["actionability"] < eta:
        failed.append("actionability")
    if scores["personalization"] < eta:
        failed.append("personalization")
    if scores["leakage"] > delta:
        failed.append("answer leakage")
    return failed


# ---------------------------------------------------------------------------
# Feedback generation pipeline
# ---------------------------------------------------------------------------

def generate_feedback_file(
    cases_file: str,
    output_file: str,
    cfg: dict,
    adapter_path: Optional[str],
    direct: bool,
    resources_file: Optional[str] = None,
    enable_revision: bool = True,
) -> None:
    cases = read_table(cases_file)

    gen = QwenFeedbackGenerator(
        cfg["generator"],
        adapter_path=adapter_path,
        for_training=False,
    )
    evaluator = FeedbackEvaluator(cfg["verification"])

    retriever = None
    if (not direct) and resources_file:
        resources = read_table(resources_file)
        r_cfg = cfg["retrieval"]
        retriever = PedagogicalRetriever(
            resources,
            lambda_semantic=r_cfg["lambda_semantic"],
            lambda_gap=r_cfg["lambda_gap"],
            lambda_difficulty=r_cfg["lambda_difficulty"],
        )

    rows = []
    for idx, row in cases.iterrows():
        case = row.to_dict()

        # Parse JSON-like columns if supplied.
        for key in ["mastery_by_concept", "concept_gaps", "misconceptions"]:
            if key in case and isinstance(case[key], str):
                try:
                    case[key] = json.loads(case[key])
                except Exception:
                    try:
                        case[key] = ast.literal_eval(case[key])
                    except Exception:
                        pass

        if not direct:
            required = [
                c for c in str(case.get("concepts", "")).split("|") if c
            ]
            mastery_map = case.get("mastery_by_concept", {})
            if not isinstance(mastery_map, dict):
                mastery_map = {}
            gaps = concept_gap(
                mastery_map,
                required,
                cfg["profile"]["mastery_threshold_gamma"],
            )
            case["concept_gaps"] = gaps

            if retriever is not None:
                ability = (
                    float(np.mean(list(mastery_map.values())))
                    if mastery_map else 0.5
                )
                case["retrieved_evidence"] = retriever.retrieve(
                    query_text=str(case.get("question_text", "")),
                    gaps=gaps,
                    learner_ability=ability,
                    top_k=cfg["retrieval"]["top_k"],
                )

        if direct:
            feedback = gen.generate(
                DIRECT_SYSTEM_PROMPT,
                build_direct_prompt(case),
            )
        else:
            feedback = gen.generate(
                CPLF_SYSTEM_PROMPT,
                build_cplf_prompt(case),
            )

        rub = evaluator.rubric(case, feedback)
        rel = evaluator.relevance(
            str(case.get("question_text", "")),
            feedback,
        )
        scores = {"relevance": rel, **rub}
        q = verification_score(scores)

        revised = feedback
        revision_applied = False
        vcfg = cfg["verification"]
        failed = failed_dimensions(
            scores, eta=vcfg["eta"], delta=vcfg["delta"]
        )
        accepted = q >= vcfg["eta"] and scores["leakage"] <= vcfg["delta"]

        if (not direct) and enable_revision and (not accepted):
            revised = gen.generate(
                REVISION_SYSTEM_PROMPT,
                build_revision_prompt(case, feedback, failed),
            )
            revision_applied = True
            rub2 = evaluator.rubric(case, revised)
            rel2 = evaluator.relevance(
                str(case.get("question_text", "")),
                revised,
            )
            final_scores = {"relevance": rel2, **rub2}
        else:
            final_scores = scores

        final_fqs = verification_score(final_scores)

        rows.append({
            "case_id": case.get("case_id", idx),
            "dataset": case.get("dataset", ""),
            "method": "Direct_LLM" if direct else "CPLF-LLM",
            "feedback_before_verification": feedback,
            "revision_applied": revision_applied,
            "failed_dimensions": "|".join(failed),
            "feedback_final": revised,
            **{f"before_{k}": v for k, v in scores.items()},
            "before_fqs": q,
            **final_scores,
            "fqs": final_fqs,
        })

    write_table(pd.DataFrame(rows), output_file)


# ---------------------------------------------------------------------------
# Human validation
# ---------------------------------------------------------------------------

def icc_2k(ratings: np.ndarray) -> float:
    """
    ICC(2,k): two-way random-effects, absolute agreement, average measures.

    ratings shape: n_targets x k_raters
    """
    x = np.asarray(ratings, dtype=float)
    if x.ndim != 2:
        raise ValueError("ratings must be a 2D array.")
    n, k = x.shape
    if n < 2 or k < 2:
        raise ValueError("ICC requires at least 2 targets and 2 raters.")

    grand = x.mean()
    mean_targets = x.mean(axis=1)
    mean_raters = x.mean(axis=0)

    ss_targets = k * np.sum((mean_targets - grand) ** 2)
    ss_raters = n * np.sum((mean_raters - grand) ** 2)
    ss_total = np.sum((x - grand) ** 2)
    ss_error = ss_total - ss_targets - ss_raters

    ms_targets = ss_targets / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    # Shrout & Fleiss / McGraw & Wong ICC(2,k)
    denom = ms_targets + (ms_raters - ms_error) / n
    if denom == 0:
        return float("nan")
    return float((ms_targets - ms_error) / denom)


def human_validation_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long format expected:
      case_id, dimension, rater_id, score, automated_score
    """
    rows = []
    for dim, g in df.groupby("dimension"):
        pivot = g.pivot_table(
            index="case_id", columns="rater_id", values="score", aggfunc="mean"
        ).dropna()
        icc = icc_2k(pivot.to_numpy())
        human_mean = (
            g.groupby("case_id")["score"].mean().rename("human_mean")
        )
        auto = (
            g.groupby("case_id")["automated_score"].mean().rename("automated")
        )
        merged = pd.concat([human_mean, auto], axis=1).dropna()
        rho, p = spearmanr(merged["human_mean"], merged["automated"])
        rows.append({
            "dimension": dim,
            "icc_2k": icc,
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "n_cases": int(len(merged)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired cluster bootstrap + Holm
# ---------------------------------------------------------------------------

def _metric_on_prediction_rows(rows: pd.DataFrame, metric: str) -> float:
    if metric == "auc":
        return float(roc_auc_score(rows["y_true"], rows["y_prob"]))
    if metric == "f1":
        return float(f1_score(rows["y_true"], rows["y_prob"] >= 0.5))
    if metric == "nll":
        return float(log_loss(rows["y_true"], rows["y_prob"], labels=[0, 1]))
    if metric == "brier":
        return float(brier_score_loss(rows["y_true"], rows["y_prob"]))
    if metric in rows.columns:
        return float(rows[metric].mean())
    raise ValueError(f"Unsupported metric: {metric}")


def paired_cluster_bootstrap(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str,
    cluster_col: str,
    n_boot: int = 10000,
    seed: int = 2024,
) -> dict:
    a = df[df["method"] == method_a].copy()
    b = df[df["method"] == method_b].copy()

    shared = sorted(set(a[cluster_col]) & set(b[cluster_col]))
    if not shared:
        raise ValueError("No shared clusters between methods.")

    a = a[a[cluster_col].isin(shared)]
    b = b[b[cluster_col].isin(shared)]

    observed = (
        _metric_on_prediction_rows(a, metric)
        - _metric_on_prediction_rows(b, metric)
    )

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)

    a_groups = {k: g.copy() for k, g in a.groupby(cluster_col)}
    b_groups = {k: g.copy() for k, g in b.groupby(cluster_col)}

    for i in range(n_boot):
        sampled = rng.choice(shared, size=len(shared), replace=True)
        aa = pd.concat([a_groups[c] for c in sampled], ignore_index=True)
        bb = pd.concat([b_groups[c] for c in sampled], ignore_index=True)
        diffs[i] = (
            _metric_on_prediction_rows(aa, metric)
            - _metric_on_prediction_rows(bb, metric)
        )

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_le = (np.sum(diffs <= 0) + 1) / (n_boot + 1)
    p_ge = (np.sum(diffs >= 0) + 1) / (n_boot + 1)
    p_two = min(1.0, 2 * min(p_le, p_ge))

    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "difference": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_raw": float(p_two),
        "n_boot": int(n_boot),
        "cluster_col": cluster_col,
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted_sorted[rank] = min(1.0, running)
    out = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        out[idx] = adjusted_sorted[rank]
    return out


# ---------------------------------------------------------------------------
# Seed-level summaries
# ---------------------------------------------------------------------------

def summarize_seed_results(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    groups = [c for c in ["experiment", "dataset", "method"] if c in df.columns]
    rows = []
    for keys, g in df.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(groups, keys))
        for metric in metrics:
            if metric not in g.columns:
                continue
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            rows.append({
                **base,
                "metric": metric,
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n_runs": int(len(vals)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def cmd_sources(args):
    for name, url in DATASET_SOURCES.items():
        print(f"{name}\t{url}")


def cmd_validate(args):
    df = normalize_interaction_table(read_table(args.input))
    stats = dataset_statistics(df)
    print(json.dumps(stats, indent=2))
    if args.output:
        write_table(df, args.output)


def cmd_split(args):
    df = normalize_interaction_table(read_table(args.input))
    split = make_fixed_student_split(
        df, args.output_dir, seed=args.seed
    )
    print({k: len(v) for k, v in split.items()})


def cmd_train_predictor(args):
    cfg = load_config(args.config)
    df = normalize_interaction_table(read_table(args.input))
    split_ids = load_split_ids(args.split_dir)
    parts = apply_split(df, split_ids)
    metrics = train_predictor(
        parts["train"],
        parts["validation"],
        parts["test"],
        cfg,
        args.output_dir,
        args.seed,
    )
    print(json.dumps(metrics, indent=2))


def cmd_train_generator(args):
    cfg = load_config(args.config)
    train_lora_generator(
        training_file=args.input,
        output_dir=args.output_dir,
        cfg=cfg["generator"],
        direct=args.direct,
    )


def cmd_generate_feedback(args):
    cfg = load_config(args.config)
    generate_feedback_file(
        cases_file=args.input,
        output_file=args.output,
        cfg=cfg,
        adapter_path=args.adapter,
        direct=args.direct,
        resources_file=args.resources,
        enable_revision=not args.no_revision,
    )


def cmd_human_validation(args):
    df = read_table(args.input)
    out = human_validation_report(df)
    write_table(out, args.output)


def cmd_bootstrap(args):
    cfg = load_config(args.config)
    df = read_table(args.input)
    res = paired_cluster_bootstrap(
        df,
        args.method_a,
        args.method_b,
        args.metric,
        args.cluster,
        n_boot=args.n_boot or cfg["statistics"]["bootstrap_resamples"],
        seed=cfg["statistics"]["bootstrap_seed"],
    )
    pd.DataFrame([res]).to_csv(args.output, index=False)


def cmd_holm(args):
    df = read_table(args.input)
    df["p_holm"] = holm_adjust(df[args.p_column].to_numpy(float))
    write_table(df, args.output)


def cmd_summary(args):
    df = read_table(args.input)
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    out = summarize_seed_results(df, metrics)
    write_table(out, args.output)


def build_parser():
    p = argparse.ArgumentParser(
        description="Complete CPLF-LLM computational protocol reference implementation"
    )
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("sources")
    q.set_defaults(func=cmd_sources)

    q = sub.add_parser("validate")
    q.add_argument("--input", required=True)
    q.add_argument("--output")
    q.set_defaults(func=cmd_validate)

    q = sub.add_parser("split")
    q.add_argument("--input", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--seed", type=int, default=2024)
    q.set_defaults(func=cmd_split)

    q = sub.add_parser("train-predictor")
    q.add_argument("--input", required=True)
    q.add_argument("--split-dir", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--seed", type=int, default=2024)
    q.add_argument("--config")
    q.set_defaults(func=cmd_train_predictor)

    q = sub.add_parser("train-generator")
    q.add_argument("--input", required=True)
    q.add_argument("--output-dir", required=True)
    q.add_argument("--direct", action="store_true")
    q.add_argument("--config")
    q.set_defaults(func=cmd_train_generator)

    q = sub.add_parser("generate-feedback")
    q.add_argument("--input", required=True)
    q.add_argument("--output", required=True)
    q.add_argument("--adapter")
    q.add_argument("--resources")
    q.add_argument("--direct", action="store_true")
    q.add_argument("--no-revision", action="store_true")
    q.add_argument("--config")
    q.set_defaults(func=cmd_generate_feedback)

    q = sub.add_parser("human-validation")
    q.add_argument("--input", required=True)
    q.add_argument("--output", required=True)
    q.set_defaults(func=cmd_human_validation)

    q = sub.add_parser("bootstrap")
    q.add_argument("--input", required=True)
    q.add_argument("--method-a", required=True)
    q.add_argument("--method-b", required=True)
    q.add_argument("--metric", required=True)
    q.add_argument("--cluster", required=True)
    q.add_argument("--n-boot", type=int)
    q.add_argument("--output", required=True)
    q.add_argument("--config")
    q.set_defaults(func=cmd_bootstrap)

    q = sub.add_parser("holm")
    q.add_argument("--input", required=True)
    q.add_argument("--p-column", default="p_raw")
    q.add_argument("--output", required=True)
    q.set_defaults(func=cmd_holm)

    q = sub.add_parser("summary")
    q.add_argument("--input", required=True)
    q.add_argument("--metrics", required=True)
    q.add_argument("--output", required=True)
    q.set_defaults(func=cmd_summary)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
