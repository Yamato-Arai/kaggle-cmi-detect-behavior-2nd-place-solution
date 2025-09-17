import os
import sys
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Any, Sequence

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import math  # Used in PositionalEncoding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from transformers.optimization import get_cosine_schedule_with_warmup
from scipy.spatial.transform import Rotation as R

from utils import FEATURE_NAMES

try:
    import tyro  # for CLI                           
except ImportError:  # Optional dependency
    tyro = None  # type: ignore

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autograd.anomaly_mode.set_detect_anomaly(False)

# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set Lightning seed (deterministic algorithms are manually controlled)
    L.seed_everything(seed, workers=True)  # type: ignore
    
    # Avoid adaptive_max_pool deterministic errors entirely
    # Disable deterministic algorithms; use cudnn.deterministic to ensure reproducibility
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------
# Dataset utilities
# -------------------------------------------------------------

# Updated feature columns to include all engineered features
_FEATURE_COLUMNS = FEATURE_NAMES

_LABEL_NON_TARGET = "Non-Target"


def behavior_to_phase(behavior):
    """Convert behavior string to phase index"""
    if behavior in ['Moves hand to target location', 'Relaxes and moves hand to target location']:
        return 0  # phase1
    elif behavior == 'Hand at target location':
        return 1  # phase2
    elif behavior == 'Performs gesture':
        return 2  # phase3
    else:
        return -1  # unknown phase

from utils import quaternion_to_6d_rotation, remove_gravity_from_acc, calculate_angular_velocity_from_quat

def quaternion_slerp(q1, q2, t):
    """
    Spherical linear interpolation (SLERP) for quaternions
    
    Args:
        q1, q2: Quaternion [w, x, y, z]
        t: Interpolation parameter (0.0 to 1.0)
        
    Returns:
        Interpolated quaternion
    """
    # Compute inner product
    dot = np.dot(q1, q2)
    
    # Choose shortest path (flip one if dot < 0)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # Threshold: if nearly identical quaternions, use linear interpolation
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    # SLERP computation
    theta_0 = np.arccos(np.abs(dot))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return (s0 * q1) + (s1 * q2)

def make_feature_from_np(data, tof):
    acc = data[:, :3].copy()
    rot = data[:, 3:7].copy()
    handedness = data[0, 7]

    # acc
    feat = acc.copy()

    # 6D
    rot_6d = quaternion_to_6d_rotation(rot)
    feat = np.concatenate([feat, rot_6d], axis=1)

    # angular velocity
    angular_velocity = calculate_angular_velocity_from_quat(rot)
    feat = np.concatenate([feat, angular_velocity], axis=1)

    # linear acc
    linear_acc = remove_gravity_from_acc(acc, rot)
    feat = np.concatenate([feat, linear_acc], axis=1)

    # fillna
    feat = np.nan_to_num(feat, nan=0.0).astype(np.float32)

    # handedness
    if handedness == 0:
        feat[:, 0] *= -1.0
        feat[:, 3] *= -1.0
        feat[:, 7] *= -1.0
        feat[:, 10] *= -1.0
        feat[:, 11] *= -1.0
        feat[:, 12] *= -1.0

        tof3 = tof[:,-64*3:-64*2]
        tof3 = tof3.reshape(tof3.shape[0], 8, 8)
        tof3 = tof3[:,::-1,:]
        tof3 = tof3.reshape(tof3.shape[0], -1)
        tof5 = tof[:,-64:]
        tof5 = tof5.reshape(tof5.shape[0], 8, 8)
        tof5 = tof5[:,::-1,:]
        tof5 = tof5.reshape(tof5.shape[0], -1)
        tof[:, -64*3:-64*2] = tof5
        tof[:, -64:] = tof3

    return feat, tof

def process_sequence(sid, df, label2idx):
    """Process one sequence_id and return a sample tuple"""
    grp = (
        df.filter(pl.col("sequence_id") == sid)
        .sort("sequence_counter")
    )

    # Features [T, F]
    subject = grp.select("subject").to_numpy().flatten()[0]
    feat = grp.select(["acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_z", "rot_w", "handedness"]).to_numpy()
    tof_cols = [f"tof_{i}_v{j}" for i in range(1,6) for j in range(64)]
    tof = grp.select(tof_cols).to_numpy()

    # Extract and convert phase information
    behaviors = grp.select("behavior").to_numpy().flatten()
    phases = np.array([behavior_to_phase(b) for b in behaviors])

    # Determine label as a class index of (orientation, gesture, phase1_behavior)
    row0 = grp.row(0)
    gesture = row0[grp.columns.index("gesture")]
    orientation = row0[grp.columns.index("orientation")]
    phase1_behavior = get_phase1_behavior(df, sid)
    label_idx = label2idx[(orientation, gesture, phase1_behavior)]

    feat, tof = make_feature_from_np(feat, tof)

    return (feat, tof, len(feat), label_idx, phases, subject)

class GestureDataset(Dataset):
    """Dataset holding one sequence = one sample.

    Parameters
    ----------
    df: pl.DataFrame
        Filtered dataframe containing required columns.
    seq_ids: Sequence[str]
        Sequence IDs belonging to this split.
    label2idx: Dict[str, int]
        Mapping from gesture string to integer label.
    """

    def __init__(self, df: pl.DataFrame, seq_ids: Sequence[Any], label2idx: Dict[str, int],
                 use_tof: bool = False,
                 use_tof_mask_augmentation_prob: float = 0, is_train: bool = False,
                 rot_zero: bool = False):
        super().__init__()
        self.seq_ids = list(seq_ids)
        self.label2idx = label2idx
        self.use_tof_mask_augmentation_prob = use_tof_mask_augmentation_prob
        # Convert all sequences into numpy arrays here and cache
        self.samples: list[tuple[np.ndarray, np.ndarray, int, int, np.ndarray, str]] = []  # (feat, tof, length, label_idx, phase, subject)
        self.use_tof = use_tof
        self.is_train = is_train
        self.rot_zero = rot_zero
        # Run per-sequence processing in parallel
        # Important: samples order must match seq_ids order
        # (to preserve correspondence between validation logits and seq_ids)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=36) as executor:
            # Submit each sequence_id processing in parallel
            futures = [executor.submit(process_sequence, sid, df, self.label2idx) for sid in self.seq_ids]
            
            # Retrieve results in the original order and append to samples
            # Do not use as_completed(); iterate futures in order to preserve seq_ids ordering
            for future in futures:
                self.samples.append(future.result())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feat_np, tof_np, length_int, label_idx, phases_np, subject = self.samples[idx]

        if self.rot_zero:
            feat_np[:, 3:15] = 0

        # Special subjects: SUBJ_019262, SUBJ_045235
        if subject in ["SUBJ_019262", "SUBJ_045235"]:
            feat_np[:, 0] *= -1
            feat_np[:, 1] *= -1
            feat_np[:, 3] *= -1
            feat_np[:, 4] *= -1
            feat_np[:, 5] *= -1
            feat_np[:, 6] *= -1
            feat_np[:, 7] *= -1
            feat_np[:, 8] *= -1
            feat_np[:, 9] *= -1
            feat_np[:, 10] *= -1
            feat_np[:, 11] *= -1
            feat_np[:, 12] *= -1
            feat_np[:, 13] *= -1

            tof_np[:,:] = 0

        if self.use_tof:
            if random.random() < self.use_tof_mask_augmentation_prob:
                tof_np[:,:] = 0
            feat_np = np.concatenate([feat_np, tof_np], axis=1)
        x = torch.from_numpy(feat_np).clone()
        length = torch.tensor(length_int, dtype=torch.long)
        
        y = torch.tensor(label_idx, dtype=torch.long)
            
        phases = torch.from_numpy(phases_np).clone().long()

        if x.shape[0] > 200:
            x = x[-200:]
            length = torch.tensor(200, dtype=torch.long)
            phases = phases[-200:]

        return x, length, y, phases


class MixupDataset(Dataset):
    """Dataset wrapper implementing phase-wise mixup"""
    
    def __init__(self, dataset: GestureDataset, alpha: float = 0.2, num_classes: int = 72):
        self.dataset = dataset
        self.alpha = alpha
        self.num_classes = num_classes
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        # Fetch the original sample
        x1, length1, y1, phases1 = self.dataset[idx]
        
        # Randomly select another sample
        idx2 = random.randint(0, len(self.dataset) - 1)
        x2, length2, y2, phases2 = self.dataset[idx2]
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            if lam < 0.5:
                lam = 1.0 - lam
        else:
            lam = 1.0
        
        # Perform mixup per phase
        mixed_x, mixed_length, mixed_phases = self._mixup_by_phase(
            x1, phases1, x2, phases2, lam
        )
        
        # Mixup labels (GestureDataset already returns the correct labels)
        if y1.dim() > 0:  # soft label (tensor)
            mixed_y = lam * y1 + (1 - lam) * y2
        else:  # hard label (scalar) → make one-hot then mix
            y1_onehot = torch.zeros(self.num_classes)
            y2_onehot = torch.zeros(self.num_classes)
            y1_onehot[y1] = 1.0
            y2_onehot[y2] = 1.0
            mixed_y = lam * y1_onehot + (1 - lam) * y2_onehot
        
        return mixed_x, mixed_length, mixed_y, mixed_phases
    
    def _mixup_by_phase(self, x1, phases1, x2, phases2, lam):
        """Perform mixup per phase"""
        # Compute indices for each phase
        phase_indices = {}
        for phase in [0, 1, 2]:  # phase0, phase1, phase2
            idx1 = (phases1 == phase).nonzero(as_tuple=True)[0]
            idx2 = (phases2 == phase).nonzero(as_tuple=True)[0]
            phase_indices[phase] = (idx1, idx2)
        
        mixed_segments = []
        mixed_phase_segments = []
        
        for phase in [0, 1, 2]:
            idx1, idx2 = phase_indices[phase]
            
            if len(idx1) == 0 and len(idx2) == 0:
                # Skip if neither sample contains this phase
                continue
            elif len(idx1) == 0:
                # If x1 has no such phase, use x2 phase as is
                seg2 = x2[idx2]
                mixed_segments.append(seg2)
                mixed_phase_segments.append(torch.full((len(idx2),), phase, dtype=torch.long))
            elif len(idx2) == 0:
                # If x2 has no such phase, use x1 phase as is
                seg1 = x1[idx1]
                mixed_segments.append(seg1)
                mixed_phase_segments.append(torch.full((len(idx1),), phase, dtype=torch.long))
            else:
                # Mixup if both contain the phase
                seg1 = x1[idx1]
                seg2 = x2[idx2]
                
                # Match lengths to seg1
                if len(seg1) < len(seg2):
                    seg2 = seg2[:len(seg1)]
                elif len(seg1) > len(seg2):
                    padding = seg1[len(seg2):]
                    seg2 = torch.cat([seg2, padding], dim=0)
                
                # Run mixup (with element-wise conditional rules)
                # Standard mixup
                mixed_seg = lam * seg1 + (1 - lam) * seg2
                # If an element in seg1 is zero, set to zero
                mixed_seg = torch.where(seg1 == 0, torch.zeros_like(seg1), mixed_seg)
                # If an element in seg2 is zero, keep seg1 value
                mixed_seg = torch.where(seg2 == 0, seg1, mixed_seg)
                
                mixed_segments.append(mixed_seg)
                mixed_phase_segments.append(torch.full((len(seg1),), phase, dtype=torch.long))
        
        # Concatenate all phases
        if mixed_segments:
            mixed_x = torch.cat(mixed_segments, dim=0)
            mixed_phases = torch.cat(mixed_phase_segments, dim=0)
            mixed_length = torch.tensor(len(mixed_x), dtype=torch.long)
        else:
            # Fallback when all phases are empty (avoid errors)
            mixed_x = torch.zeros(1, x1.shape[1])
            mixed_phases = torch.tensor([-1], dtype=torch.long)
            mixed_length = torch.tensor(1, dtype=torch.long)
        
        return mixed_x, mixed_length, mixed_phases


# -------------------------------------------------------------
# Collate function for variable-length sequences
# -------------------------------------------------------------

def collate_fn(batch):
    xs, lengths, ys, phases = zip(*batch)
    lengths = torch.stack(lengths)
    max_len = lengths.max().item()
    feat_dim = xs[0].shape[1]

    padded = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    padded_phases = torch.full((len(batch), max_len), -1, dtype=torch.long)  # -1 for padding
    
    for i, (seq, l, phase) in enumerate(zip(xs, lengths, phases)):
        padded[i, : l.item()] = seq
        padded_phases[i, : l.item()] = phase

    # Stack labels (support both soft and hard labels)
    if ys[0].dim() > 0 and ys[0].shape[0] > 1:  # soft label (one-hot)
        stacked_ys = torch.stack(ys)
    else:  # hard label (scalar)
        stacked_ys = torch.stack(ys)

    return padded, lengths, stacked_ys, padded_phases




# -------------------------------------------------------------
# Model
# -------------------------------------------------------------

from model import IMUModel, ALLModel, IMUSimpleModel, ALLSimpleModel, IMUDeepModel, ALLDeepModel, ALL25DModel

# -------------------------------------------------------------
# LightningModule
# -------------------------------------------------------------

class GestureLitModel(L.LightningModule):
    def __init__(
        self,
        input_size: int = 14,  # Updated to match expanded feature set
        hidden_size: int = 128,
        num_layers: int = 4,
        num_classes: int = 144,  # number of classes for (orientation × gesture × phase1_behavior)
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        model_type: str = "squeezeformer",  # "squeezeformer" or "gru"
        eval_mapping: Dict[str, Any] | None = None,  # mapping used for evaluation
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['eval_mapping'])  # exclude eval_mapping from saved hparams
        
        if model_type == "imu":
            self.model = IMUModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        elif model_type == "all":
            self.model = ALLModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        elif model_type == "imu_simple":
            self.model = IMUSimpleModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        elif model_type == "all_simple":
            self.model = ALLSimpleModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        elif model_type == "imu_deep":
            self.model = IMUDeepModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        elif model_type == "all_deep":
            self.model = ALLDeepModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        elif model_type == "all_25d":
            self.model = ALL25DModel(
                input_size=input_size,
                n_classes=num_classes,
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")
            
        self.criterion = nn.CrossEntropyLoss()
        self.eval_mapping = eval_mapping
    # forward not strictly needed but helpful for inference
    def forward(self, x, lengths, phases=None):
        return self.model(x, lengths, phases)

    def _shared_step(self, batch, stage: str):
        x, lengths, y, phases = batch
        outputs = self(x, lengths, phases)
        
        # If outputs are dict (including phase prediction)
        if isinstance(outputs, dict):
            logits = outputs['gesture_logits']
        else:
            logits = outputs
        
        # Support soft label (mixup) vs hard label
        if y.dim() > 1:  # soft label (one-hot) - mixup
            # With mixup: compute cross entropy with soft labels
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
            # Use hard label for accuracy computation
            y_hard = y.argmax(dim=1)
        else:  # hard label (scalar)
            y_hard = y
            # Standard cross entropy loss
            loss = self.criterion(logits, y)
        
        # Add phase prediction loss when available in outputs
        if isinstance(outputs, dict) and 'phase_logits' in outputs:
            phase_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
            phase_loss = phase_criterion(outputs['phase_logits'].view(-1, 3), phases.view(-1))
            loss = loss + phase_loss
            self.log(f"{stage}/phase_loss", phase_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        acc = (logits.argmax(dim=1) == y_hard).float().mean()
        
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        x, lengths, y, phases = batch
        outputs = self(x, lengths, phases)

        # If outputs are dict (including phase prediction)
        if isinstance(outputs, dict):
            logits = outputs['gesture_logits']
        else:
            logits = outputs

        # Support soft label (mixup) vs hard label
        if y.dim() > 1:  # soft label (one-hot) - mixup
            # With mixup: compute cross entropy with soft labels
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
            # Use hard label for accuracy computation
            y_hard = y.argmax(dim=1)
        else:  # hard label
            y_hard = y
            # In validation, always use standard cross entropy
            loss = self.criterion(logits, y)

        # Add phase prediction loss when available in outputs
        if isinstance(outputs, dict) and 'phase_logits' in outputs:
            phase_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
            phase_loss = phase_criterion(outputs['phase_logits'].view(-1, 3), phases.view(-1))
            loss = loss + phase_loss
            self.log("val/phase_loss", phase_loss, prog_bar=True, on_step=False, on_epoch=True)

        acc = (logits.argmax(dim=1) == y_hard).float().mean()
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        preds = logits.argmax(dim=1)
        if not hasattr(self, "_val_preds"):
            self._val_preds = []  # type: ignore[attr-defined]
            self._val_targets = []  # type: ignore[attr-defined]
            self._val_logits = []  # type: ignore[attr-defined]  # logits storage
        self._val_preds.append(preds.cpu())
        self._val_targets.append(y_hard.cpu())  # save hard labels
        self._val_logits.append(logits.cpu())  # save logits
        return loss

    def on_validation_epoch_end(self):
        import numpy as _np
        from sklearn.metrics import f1_score as _f1

        if not hasattr(self, "_val_preds"):
            return  # no validation run (e.g., fast_dev_run)

        preds_72 = torch.cat(self._val_preds).numpy()  # type: ignore[attr-defined]
        targets_72 = torch.cat(self._val_targets).numpy()  # type: ignore[attr-defined]
        val_logits = torch.cat(self._val_logits).numpy()  # type: ignore[attr-defined]

        # Save logits (for ensembling)
        # Note: collected in validation DataLoader order
        # DataLoader uses shuffle=False, so order matches saved val_seq_ids
        if hasattr(self, '_logits_save_path') and self._logits_save_path:
            _np.save(self._logits_save_path, {
                'logits': val_logits,
                'targets': targets_72,
                'preds': preds_72
            })

        # Convert N-class to 18-class for evaluation if mapping is provided
        if self.eval_mapping is not None:
            class72_to_class18 = self.eval_mapping['class72_to_class18']
            class18_to_class9 = self.eval_mapping['class18_to_class9']
            
            # Convert predictions and targets to 18 classes
            preds_18 = np.array([class72_to_class18[p] for p in preds_72])
            targets_18 = np.array([class72_to_class18[t] for t in targets_72])
            
            # Log accuracy for 18-class classification
            acc_18 = (preds_18 == targets_18).astype(float).mean()
            self.log("val/acc_18class", acc_18, prog_bar=True)
            
            # Further convert 18-class to 9-class
            preds_9 = np.array([class18_to_class9[p] for p in preds_18])
            targets_9 = np.array([class18_to_class9[t] for t in targets_18])
            
            # 9-class (macro) F1
            f1_macro = _f1(targets_9, preds_9, average="macro")
            
            # Binary F1: Non-Target (0) vs Others
            bin_targets = (targets_9 != 0).astype(int)
            bin_preds = (preds_9 != 0).astype(int)
            f1_binary = _f1(bin_targets, bin_preds)
            
            # Also log N-class accuracy
            acc_72 = (preds_72 == targets_72).astype(float).mean()
            self.log("val/acc_72class", acc_72, prog_bar=False)
        else:
            # Fallback: evaluate directly in N-class (not perfect)
            f1_macro = _f1(targets_72, preds_72, average="macro")
            # Binary classification with a provisional threshold (not perfect)
            bin_targets = (targets_72 < 36).astype(int)  # provisional threshold
            bin_preds = (preds_72 < 36).astype(int)
            f1_binary = _f1(bin_targets, bin_preds)

        f1_mean = (f1_macro + f1_binary) / 2.0

        # Log
        self.log("val/f1_macro", f1_macro, prog_bar=True)
        self.log("val/f1_binary", f1_binary, prog_bar=True)
        self.log("val/f1_mean", f1_mean, prog_bar=True)

        # Free memory
        del self._val_preds  # type: ignore[attr-defined]
        del self._val_targets  # type: ignore[attr-defined]
        del self._val_logits  # type: ignore[attr-defined]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        
        # Use get_cosine_schedule_with_warmup
        # Estimate total training steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.1)  # 10% warmup steps
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # update every step
                "frequency": 1,
            },
        }


# -------------------------------------------------------------
# DataModule
# -------------------------------------------------------------

class GestureDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_ds: GestureDataset,
        val_ds: GestureDataset,
        batch_size: int = 128,
        num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )


# -------------------------------------------------------------
# Config
# -------------------------------------------------------------

@dataclass
class CFG:
    csv_path: str = "../../input/train.csv"
    cache_dir: str = "./cache"  # cache directory
    project: str = "CMI2025"
    exp_name: str = "imu_102_10"
    seed: int = 42

    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 128
    num_layers: int = 2
    max_epochs: int = 50
    num_workers: int = 32

    # Model type selection
    model_type: str = "imu"  # "imu" or "all" or "imu_simple" or "all_simple" or "imu_deep" or "all_25d"

    # Mixup settings
    use_mixup: bool = True  # whether to use Mixup
    mixup_alpha: float = 0.5  # alpha for Beta distribution

    # Augmentation settings
    use_tof_mask_augmentation_prob : float = 0.1  # probability to apply TOF mask augmentation

    rot_zero: bool = False  # whether to zero out all rotation features

    # Ensemble-related settings
    save_logits: bool = True  # whether to save validation logits for ensembling

    n_folds: int = 10
    fold: int | None = None  # if None → run all folds sequentially

    accelerator: str = "gpu"  # "cpu" or "gpu"
    devices: int | None = 1  # None → auto
    debug: bool = False  # If True, run quick debug on 1 batch and disable WandB

# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def prepare_dataframe(csv_path: str) -> pl.DataFrame:

    df = pl.read_csv(csv_path)
    df_demo = pl.read_csv("../../input/train_demographics.csv")
    df = df.join(df_demo, on="subject", how="left")
    
    # Select base columns (raw columns before feature engineering)
    keep_cols = [
        "row_id",
        "sequence_type",
        "sequence_id",
        "sequence_counter",
        "subject",
        "gesture",
        "behavior",
        "orientation",  # include posture information
        "acc_x", "acc_y", "acc_z",
        "rot_x", "rot_y", "rot_z", "rot_w",
        "handedness",
    ]
    tof_cols = [f"tof_{i}_v{j}" for i in range(1,6) for j in range(64)]
    keep_cols.extend(tof_cols)
    df = df.select(keep_cols)
    import concurrent.futures

    def fill_tof_col(col_name: str, df: pl.DataFrame) -> pl.Series:
        filled = pl.when((pl.col(col_name).is_null()) | (pl.col(col_name) == -1)) \
                   .then(0) \
                   .otherwise(pl.col(col_name)) \
                   .alias(col_name)
        return filled

    # Fill each tof_col in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        filled_exprs = list(executor.map(lambda c: fill_tof_col(c, df), tof_cols))
    df = df.with_columns(filled_exprs)
    
    return df


def get_phase1_behavior(df: pl.DataFrame, sequence_id: str) -> str:
    """Get phase1 behavior for the specified sequence_id"""
    seq_behaviors = (
        df.filter(pl.col("sequence_id") == sequence_id)
        .select("behavior")
        .to_series()
        .to_list()
    )
    
    # Find phase1 behavior
    phase1_behaviors = ['Moves hand to target location', 'Relaxes and moves hand to target location']
    for behavior in seq_behaviors:
        if behavior in phase1_behaviors:
            return behavior
    
    # Raise error if phase1 is not found
    raise ValueError(f"Phase1 behavior not found in sequence {sequence_id}")

def make_label_mapping(df: pl.DataFrame) -> Dict[str, int]:
    orientation_gesture_pairs = (
        df.filter(pl.col("sequence_counter") == 0)
        .select(["orientation", "gesture", "behavior"])
        .unique()
        .sort(["orientation", "gesture", "behavior"])
    )
    
    # Assign labels (0..N-1) for (orientation, gesture, phase1_behavior) triplets
    label2idx = {}
    for i, (orientation, gesture, phase1_behavior) in enumerate(orientation_gesture_pairs.iter_rows()):
        label2idx[(orientation, gesture, phase1_behavior)] = i
    
    print(f"Created {len(label2idx)} orientation-gesture-phase1behavior combinations")
    
    # Debug: print number of gestures per orientation
    orientations = df.select("orientation").unique().sort("orientation").to_series().to_list()
    for orientation in orientations:
        gestures_in_orientation = df.filter(pl.col("orientation") == orientation).select("gesture").unique().to_series().to_list()
        print(f"Orientation '{orientation}': {len(gestures_in_orientation)} gestures")
    
    return label2idx


def create_gesture_mapping_for_evaluation(df: pl.DataFrame, label2idx: Dict[Tuple[str, str, str], int]) -> Dict[str, Any]:
    """Create mapping to convert N-class predictions into 18-class evaluation space"""
    
    # Identify Non-Target gestures
    non_target_gestures = (
        df.filter(pl.col("sequence_type") == _LABEL_NON_TARGET)
        .select("gesture")
        .unique()
        .to_series()
        .to_list()
    )
    
    # Identify Target gestures
    target_gestures = (
        df.filter(pl.col("sequence_type") == "Target")
        .select("gesture")
        .unique()
        .sort("gesture")
        .to_series()
        .to_list()
    )
    
    # Build the full gesture list (18-class for evaluation)
    all_gestures = sorted(set(non_target_gestures + target_gestures))
    gesture_to_idx18 = {gesture: i for i, gesture in enumerate(all_gestures)}
    
    # Build mapping N-class → 18-class (ignore orientation and phase1_behavior; use gesture only)
    classN_to_class18 = {}
    for (orientation, gesture, phase1_behavior), idxN in label2idx.items():
        classN_to_class18[idxN] = gesture_to_idx18[gesture]
    
    # Build mapping 18-class → 9-class
    # All Non-Target gestures map to 0, Target gestures map to 1..8
    class18_to_class9 = {}
    for gesture, idx18 in gesture_to_idx18.items():
        if gesture in non_target_gestures:
            class18_to_class9[idx18] = 0  # Non-Target
        else:
            # Compute target gesture index
            target_idx = target_gestures.index(gesture) + 1
            class18_to_class9[idx18] = target_idx
    
    return {
        'class72_to_class18': classN_to_class18,  # keep old key name for backward compatibility
        'class18_to_class9': class18_to_class9,
        'non_target_gestures': non_target_gestures,
        'target_gestures': target_gestures,
        'non_target_indices': [gesture_to_idx18[g] for g in non_target_gestures],
        'gesture_to_idx18': gesture_to_idx18,
        'all_gestures': all_gestures
    }


def run_fold(cfg: CFG, fold_idx: int, df: pl.DataFrame, label2idx: Dict[Tuple[str, str, str], int]) -> Dict[str, float]:
    print("Preparing data splits ...", flush=True)
    # Get (orientation, gesture) pair per sequence_id
    seq_df = df.select(["sequence_id", "subject", "orientation", "gesture", "handedness"]).unique().sort(["sequence_id"])
    seq_ids = seq_df["sequence_id"].to_list()
    subjects = seq_df["subject"].to_list()
    
    # Use handedness as y for stratification
    y = seq_df["handedness"].to_list()

    # Use StratifiedGroupKFold to balance (orientation, gesture) across folds
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    splits = list(sgkf.split(seq_ids, y=y, groups=subjects))
    train_idx, val_idx = splits[fold_idx]

    train_seq_ids = [seq_ids[i] for i in train_idx]
    val_seq_ids = [seq_ids[i] for i in val_idx]

    print("Creating datasets ...", flush=True)
    
    # Build datasets (apply soft labels only for train)
    use_tof = (cfg.model_type in ["all", "all_simple", "all_deep", "all_rot", "all_25d"])
    train_ds = GestureDataset(df, train_seq_ids, label2idx, 
                             use_tof=use_tof,
                             use_tof_mask_augmentation_prob=cfg.use_tof_mask_augmentation_prob,
                             is_train=True,
                             rot_zero=cfg.rot_zero)
    val_ds = GestureDataset(df, val_seq_ids, label2idx, use_tof=use_tof, rot_zero=cfg.rot_zero)  # validation uses hard labels

    # Wrap with MixupDataset if Mixup is enabled
    if cfg.use_mixup:
        print(f"Using Mixup with alpha={cfg.mixup_alpha}", flush=True)
        train_ds = MixupDataset(train_ds, alpha=cfg.mixup_alpha, num_classes=len(label2idx))

    print("Initializing DataModule ...", flush=True)
    dm = GestureDataModule(
        train_ds, 
        val_ds, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers
    )

    print("Building model ...", flush=True)
    # Create evaluation mapping
    eval_mapping = create_gesture_mapping_for_evaluation(df, label2idx)
    print(f"Eval mapping created: {len(eval_mapping['non_target_gestures'])} Non-Target gestures, {len(eval_mapping['target_gestures'])} Target gestures")
    print(f"Total orientation-gesture combinations: {len(label2idx)}")
    
    model = GestureLitModel(
        input_size=len(_FEATURE_COLUMNS),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_classes=len(label2idx),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        model_type=cfg.model_type,
        eval_mapping=eval_mapping,
    )

    # ---------------------------------------------------------
    # Decide a unique output directory (./output/<exp_name>/<n>/)
    # ---------------------------------------------------------
    base_out_dir = Path("../../output") / cfg.exp_name
    run_idx = 0
    while (base_out_dir / str(run_idx)).exists():
        run_idx += 1
    out_dir = base_out_dir / str(run_idx)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Set save path for validation logits (ensemble)
    logits_save_path = out_dir / f"val_logits_fold{fold_idx}.npy"
    model._logits_save_path = str(logits_save_path) if cfg.save_logits else None

    # Also save validation sequence IDs (for ensemble verification)
    # Important: val_seq_ids must align with validation logits order
    # GestureDataset preserves seq_ids order and DataLoader uses shuffle=False
    if cfg.save_logits:
        val_seq_info_path = out_dir / f"val_seq_ids_fold{fold_idx}.npy"
        np.save(val_seq_info_path, val_seq_ids)

    # Callbacks & logger
    if cfg.debug:
        # Debug mode: run only 1 batch for train/val and disable external logging
        wandb_logger: Any | bool = False
        callbacks: list[Any] = []
    else:
        wandb_logger = WandbLogger(
            project=cfg.project,
            name=f"{cfg.exp_name}_fold{fold_idx}",
            config=asdict(cfg),
            reinit=True,  # start new run per fold
        )

        callbacks = [
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath=out_dir / "checkpoints",
                monitor="val/acc",
                mode="max",
                filename="{fold}-best-acc",  # type: ignore[format-string]
                save_top_k=1,
                save_last=True,  # also save the last-epoch model
            ),
        ]

    print("Setting up trainer ...", flush=True)
    trainer = L.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        max_epochs=1 if cfg.debug else cfg.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=True,
        fast_dev_run=1 if cfg.debug else False,
        default_root_dir=str(out_dir),  # base directory to save artifacts
        enable_progress_bar=sys.stdout.isatty(),
        gradient_clip_val=1.0,
    )

    print("Starting training ...", flush=True)
    trainer.fit(model, dm)

    # ---- Collect validation metrics per fold ----
    metrics: dict[str, float] = {}
    callback_metrics = trainer.callback_metrics  # type: ignore[assignment]
    for base in ["val/f1_macro", "val/f1_binary", "val/f1_mean", "val/acc"]:
        if base in callback_metrics:
            val = callback_metrics[base]
        elif f"{base}_epoch" in callback_metrics:
            val = callback_metrics[f"{base}_epoch"]
        else:
            continue
        # Convert tensor to float
        if isinstance(val, torch.Tensor):
            val = val.item()
        metrics[base] = float(val)

    # ---- Close WandB run ----
    if not cfg.debug and isinstance(wandb_logger, WandbLogger):
        import wandb  # late import is fine
        wandb.finish()

    return metrics


def main(cfg: CFG):
    set_seed(cfg.seed)

    # --- Progress logging ---
    print(f"Loading data from {cfg.csv_path} ...", flush=True)
    df = prepare_dataframe(cfg.csv_path)
    print("Dataframe prepared.", flush=True)

    print("Building label mapping ...", flush=True)
    label2idx = make_label_mapping(df)
    print(f"Label mapping prepared. Num orientation-gesture combinations: {len(label2idx)}", flush=True)

    folds_to_run = range(cfg.n_folds) if cfg.fold is None else [cfg.fold]
    
    fold_metrics_list: list[dict[str, float]] = []
    for fold_idx in folds_to_run:
        print(f"\n===== Fold {fold_idx} / {cfg.n_folds} =====")
        metrics = run_fold(cfg, fold_idx, df, label2idx)
        fold_metrics_list.append(metrics)

    # --- Compute and display mean across folds ---
    if fold_metrics_list:
        avg_metrics: dict[str, float] = {}
        all_keys = set().union(*(m.keys() for m in fold_metrics_list))
        for k in all_keys:
            vals = [m[k] for m in fold_metrics_list if k in m]
            if vals:
                avg_metrics[k] = float(np.mean(vals))

        print("\n===== Average validation metrics across folds =====")
        for k, v in avg_metrics.items():
            print(f"{k}: {v:.6f}")

        # --- Log to WandB (skip in debug) ---
        if not cfg.debug:
            import wandb
            wandb_run = wandb.init(
                project=cfg.project,
                name=f"{cfg.exp_name}_summary",
                config=asdict(cfg),
                reinit=True,
            )
            wandb.log(avg_metrics)
            wandb.finish()


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------

if __name__ == "__main__":
    if tyro is not None:
        cfg = tyro.cli(CFG)  # type: ignore[misc]
    else:
        print("tyro not installed. Falling back to default configuration.")
        cfg = CFG()

    main(cfg)