import polars as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
from collections import deque
from typing import Optional, List, Tuple

from utils import quaternion_to_6d_rotation, calculate_angular_velocity_from_quat, remove_gravity_from_acc
from train import GestureLitModel
from metric import score

tof_cols = [f"tof_{i}_v{j}" for i in range(1,6) for j in range(64)]

def make_data(sequence: pl.DataFrame, demographics: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:

    sequence = sequence.join(demographics, on="subject")
    row_feat = sequence.select(["acc_x", "acc_y", "acc_z", "rot_x", "rot_y", "rot_z", "rot_w", "handedness"]).to_numpy()
    tof = sequence.select(tof_cols).to_numpy()

    acc = row_feat[:, :3].copy()
    rot = row_feat[:, 3:7].copy()
    handedness = row_feat[0, 7]
    subject = sequence.select(["subject"]).to_numpy()[0,0]

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

    # Handle missing ToF values by zero-filling
    tof[tof == -1] = 0
    tof[np.isnan(tof)] = 0

    # handedness
    if handedness == 0:
        # feat
        feat[:, 0] *= -1.0
        feat[:, 3] *= -1.0
        feat[:, 7] *= -1.0
        feat[:, 10] *= -1.0
        feat[:, 11] *= -1.0
        feat[:, 12] *= -1.0
        # tof
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
    
    feat = feat.astype(np.float32)
    tof = tof.astype(np.float32)

    if feat.shape[0] > 200:
        feat = feat[-200:, :]
        tof = tof[-200:, :]
    
    # Special subjects: SUBJ_019262, SUBJ_045235
    if subject in ["SUBJ_019262", "SUBJ_045235"]:
        feat[:, 0] *= -1
        feat[:, 1] *= -1
        feat[:, 3] *= -1
        feat[:, 4] *= -1
        feat[:, 5] *= -1
        feat[:, 6] *= -1
        feat[:, 7] *= -1
        feat[:, 8] *= -1
        feat[:, 9] *= -1
        feat[:, 10] *= -1
        feat[:, 11] *= -1
        feat[:, 12] *= -1
        feat[:, 13] *= -1

        tof[:,:] = 0

    return feat, tof, subject

def load_model(model_path_list: list[str]):
    imu_model_list = []
    all_model_list = []
    imu_rot_model_list = []
    all_rot_model_list = []
    for model_path in tqdm(model_path_list):
        if "imu" in model_path:
            if "rot" in model_path:
                model = GestureLitModel.load_from_checkpoint(model_path)
                model.eval()
                model.to("cuda")
                imu_rot_model_list.append(model)
            else:
                model = GestureLitModel.load_from_checkpoint(model_path)
                model.eval()
                model.to("cuda")
                imu_model_list.append(model)
        elif "all" in model_path:
            if "rot" in model_path:
                model = GestureLitModel.load_from_checkpoint(model_path)
                model.eval()
                model.to("cuda")
                all_rot_model_list.append(model)
            else:
                model = GestureLitModel.load_from_checkpoint(model_path)
                model.eval()
                model.to("cuda")
                all_model_list.append(model)
        else:
            raise ValueError(f"Invalid model path: {model_path}")
    return imu_model_list, all_model_list, imu_rot_model_list, all_rot_model_list

imu_data = []
all_data = []
imu_rot_data = []
all_rot_data = []

def train_model(model_list, data, model_idx: Optional[int] = None):
    if len(data) < 32:
        return
    
    length_list = [data[i][0].shape[0] for i in range(len(data))]
    length_tensor = torch.tensor(length_list, dtype=torch.long, device="cuda")
    max_len = max(length_tensor)
    if len(data[0]) == 2:
        pseudo_x = [torch.from_numpy(data[i][0]).to("cuda") for i in range(len(data))]
        pseudo_label = [data[i][1] for i in range(len(data))]
    else:
        pseudo_x = [torch.from_numpy(np.concatenate([data[i][0], data[i][1]], axis=1)).to("cuda") for i in range(len(data))]
        pseudo_label = [data[i][2] for i in range(len(data))]
    feat_dim = pseudo_x[0].shape[-1]
    padded_x = torch.zeros(len(pseudo_x), max_len, feat_dim, dtype=torch.float32, device="cuda")
    for i, (seq, length) in enumerate(zip(pseudo_x, length_list)):
        padded_x[i, :length] = seq

    pseudo_label = torch.tensor(pseudo_label, dtype=torch.long, device="cuda")
    criterion = nn.CrossEntropyLoss()
    
    for i, model in enumerate(model_list):
        if model_idx is not None and i != model_idx:
            continue
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
        optimizer.zero_grad()
        outputs = model(padded_x, length_tensor)
        pred_logits = outputs["gesture_logits"]
        loss = criterion(pred_logits, pseudo_label)
        loss.backward()
        optimizer.step()
        model.eval()
    
    data.clear()

def training_step(imu_model_list, imu_rot_model_list, all_model_list, all_rot_model_list, imu_data, imu_rot_data, all_data, all_rot_data, fold_idx: Optional[int] = None):
    train_model(imu_model_list, imu_data, model_idx=fold_idx)
    train_model(imu_rot_model_list, imu_rot_data, model_idx=fold_idx)
    train_model(all_model_list, all_data, model_idx=fold_idx)
    train_model(all_rot_model_list, all_rot_data, model_idx=fold_idx)

def avg_predict(imu_model_list, all_model_list, imu_rot_model_list, all_rot_model_list, feat, tof, fold=None):
    rot_null = (feat[:,3:] == 0).all()
    imu_only = (tof == 0).all()
    pred_list = []
    if imu_only:
        if rot_null:
            x = torch.from_numpy(feat).to("cuda").unsqueeze(0)
            lengths = torch.tensor([x.shape[1]], dtype=torch.long).to("cuda")
            for i, model in enumerate(imu_rot_model_list):
                if fold is not None and i != fold:
                    continue
                with torch.no_grad():
                    pred = model(x, lengths)["gesture_logits"].cpu().numpy()[0]
                pred_list.append(pred)
        else:
            x = torch.from_numpy(feat).to("cuda").unsqueeze(0)
            lengths = torch.tensor([x.shape[1]], dtype=torch.long).to("cuda")
            for i, model in enumerate(imu_model_list):
                if fold is not None and i != fold:
                    continue
                with torch.no_grad():
                    pred = model(x, lengths)["gesture_logits"].cpu().numpy()[0]
                pred_list.append(pred)
    else:
        if rot_null:
            x = torch.from_numpy(feat).to("cuda").unsqueeze(0)
            tof = torch.from_numpy(tof).to("cuda").unsqueeze(0)
            x = torch.cat([x, tof], dim=-1)
            lengths = torch.tensor([x.shape[1]], dtype=torch.long).to("cuda")
            for i, model in enumerate(all_rot_model_list):
                if fold is not None and i != fold:
                    continue
                with torch.no_grad():
                    pred = model(x, lengths)["gesture_logits"].cpu().numpy()[0]
                pred_list.append(pred)
        else:
            x = torch.from_numpy(feat).to("cuda").unsqueeze(0)
            tof = torch.from_numpy(tof).to("cuda").unsqueeze(0)
            x = torch.cat([x, tof], dim=-1)
            lengths = torch.tensor([x.shape[1]], dtype=torch.long).to("cuda")
            for i, model in enumerate(all_model_list):
                if fold is not None and i != fold:
                    continue
                with torch.no_grad():
                    pred = model(x, lengths)["gesture_logits"].cpu().numpy()[0]
                pred_list.append(pred)
    
    return np.mean(pred_list, axis=0)

def make_pred_to_label(train_df_path: str) :
    train_df = pl.read_csv(train_df_path)
    # Get all combinations of orientation + gesture + phase1_behavior
    train_df = train_df.filter(pl.col("sequence_counter") == 0)
    orientation_gesture_pairs = (
        train_df.select(["orientation", "gesture", "behavior"])
        .unique()
        .sort(["orientation", "gesture", "behavior"])
    )
    
    pred_to_label = {}
    for i, row in enumerate(orientation_gesture_pairs.iter_rows()):
        orientation, gesture, behavior = row
        pred_to_label[i] = gesture
    
    return pred_to_label

model_path_list = [f"../../output/imu_102_10/{i}/checkpoints/last.ckpt" for i in range(10)] \
                + [f"../../output/all_102_10/{i}/checkpoints/last.ckpt" for i in range(10)] \
                + [f"../../output/imu_rot_102_10/{i}/checkpoints/last.ckpt" for i in range(10)] \
                + [f"../../output/all_rot_102_10/{i}/checkpoints/last.ckpt" for i in range(10)]
imu_model_list, all_model_list, imu_rot_model_list, all_rot_model_list = load_model(model_path_list)
pred_to_label = make_pred_to_label("../../input/train.csv")

import numpy as np
from scipy.optimize import linear_sum_assignment

def solve_capacity1_with_hungarian(S):
    """
    S: 2D array with shape (N, K) representing s_{i,j}
    Returns:
        assign: array of length N. For each i, the selected label j (0..K-1)
        value: objective value sum_i s_{i, assign[i]}
    """
    S = np.asarray(S)
    N, K = S.shape
    assert N <= K, "Feasibility requires N <= K"

    # Convert maximization to minimization (keep non-negative costs)
    cost = S.max() - S                   # shape: (N, K)

    # Assign each row to a unique column (when N <= 2K, all rows can be assigned)
    rows, cols = linear_sum_assignment(cost)

    # Fold columns back to original labels (0..K-1)
    assign = (cols % K)

    value = float(S[np.arange(N), assign].sum())
    # Check (usage count of each label ≤ 1)
    counts = np.bincount(assign, minlength=K)
    assert counts.max() <= 1

    return assign.tolist()[-1]

subject_to_idx = {}
subject_logits_list = []

def predict_fold(sequence: pl.DataFrame, demographics: pl.DataFrame, fold_idx: Optional[int] = None, use_hungarian: bool = True, use_pseudo_label: bool = False) -> str:
    global subject_to_idx, subject_logits_list, imu_data, all_data, imu_rot_data, all_rot_data
    feat, tof, subject = make_data(sequence, demographics)
    logits = avg_predict(imu_model_list, all_model_list, imu_rot_model_list, all_rot_model_list, feat, tof, fold_idx)
    if subject not in subject_to_idx:
        subject_to_idx[subject] = len(subject_to_idx)
        subject_logits_list.append([])
    subject_idx = subject_to_idx[subject]
    subject_logits_list[subject_idx].append(logits)
    if use_hungarian:
        pred = solve_capacity1_with_hungarian(subject_logits_list[subject_idx])
    else:
        pred = np.argmax(logits)
    if (tof == 0).all():
        if (feat[:,3:] == 0).all():
            imu_rot_data.append([feat, pred])
        else:
            imu_data.append([feat, pred])
            feat_rot = feat.copy()
            feat_rot[:,3:] = 0
            imu_rot_data.append([feat_rot, pred])
    else:
        if (feat[:,3:] == 0).all():
            all_rot_data.append([feat, tof, pred])
            imu_rot_data.append([feat, pred])
        else:
            all_data.append([feat, tof, pred])
            imu_data.append([feat, pred])
            feat_rot = feat.copy()
            feat_rot[:,3:] = 0
            all_rot_data.append([feat_rot, tof, pred])
            imu_rot_data.append([feat_rot, pred])
    if use_pseudo_label:
        training_step(imu_model_list, imu_rot_model_list, all_model_list, all_rot_model_list, imu_data, imu_rot_data, all_data, all_rot_data, fold_idx)
    gesture = pred_to_label[pred]
    return gesture

def fold_test(use_hungarian: bool = True, use_pseudo_label: bool = False):
    global subject_to_idx, subject_logits_list, imu_data, all_data, imu_rot_data, all_rot_data
    df = pl.read_csv("../../input/train.csv")
    # Disable ToF for half of the sequences (odd indices 1,3,5,...) per-sequence
    odd_seq_ids = (
        df.select("sequence_id")
          .unique()
          .sort("sequence_id")
          .with_row_count("seq_ord", offset=1)  # 1-based index
          .filter((pl.col("seq_ord") % 2) == 1)  # odd indices
          .get_column("sequence_id")
    )
    df = df.with_columns([
        pl.when(pl.col("sequence_id").is_in(odd_seq_ids))
          .then(pl.lit(None))
          .otherwise(pl.col(col))
          .alias(col)
        for col in tof_cols
    ])

    df_demographics = pl.read_csv("../../input/train_demographics.csv")
    seq_df = df.select(["sequence_id", "subject", "orientation", "gesture"]).unique().sort(["sequence_id"])
    seq_ids = seq_df["sequence_id"].to_list()
    subjects = seq_df["subject"].to_list()

    seq_df = seq_df.join(df_demographics, on="subject")
    # Use handedness as y for stratification
    y = seq_df["handedness"].to_list()

    # Use StratifiedGroupKFold to balance (orientation, gesture) across folds
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=10)
    splits = list(sgkf.split(seq_ids, y=y, groups=subjects))

    all_seq_ids = []
    all_gesture_pred = []
    all_gesture_label = []

    for fold_idx in range(10):
        subject_to_idx.clear()
        subject_logits_list.clear()
        imu_data.clear()
        all_data.clear()
        imu_rot_data.clear()
        all_rot_data.clear()
        train_idx, val_idx = splits[fold_idx]
        val_seq_ids = [seq_ids[i] for i in val_idx]
        val_df = df.filter(pl.col("sequence_id").is_in(val_seq_ids))
        gesture_pred = []
        gesture_label = []
        for seq_id in tqdm(val_seq_ids):
            seq_df = val_df.filter(pl.col("sequence_id") == seq_id)
            demographics = df_demographics.filter(pl.col("subject") == seq_df["subject"].to_list()[0])
            label = seq_df["gesture"].to_list()[0]
            gesture = predict_fold(seq_df, demographics, fold_idx, use_hungarian, use_pseudo_label)
            gesture_pred.append(gesture)
            gesture_label.append(label)
        all_seq_ids.extend(val_seq_ids)
        all_gesture_pred.extend(gesture_pred)
        all_gesture_label.extend(gesture_label)

    solution = pd.DataFrame({
        "sequence_id": all_seq_ids,
        "gesture": all_gesture_label
    })
    submission = pd.DataFrame({
        "sequence_id": all_seq_ids,
        "gesture": all_gesture_pred
    })

    f1_mean = score(solution, submission, "sequence_id")
    print(f"F1 mean: {f1_mean}")

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    global subject_to_idx, subject_logits_list, imu_data, all_data, imu_rot_data, all_rot_data
    feat, tof, subject = make_data(sequence, demographics)
    logits = avg_predict(imu_model_list, all_model_list, imu_rot_model_list, all_rot_model_list, feat, tof)
    if subject not in subject_to_idx:
        subject_to_idx[subject] = len(subject_to_idx)
        subject_logits_list.append([])
    subject_idx = subject_to_idx[subject]
    subject_logits_list[subject_idx].append(logits)
    pred = solve_capacity1_with_hungarian(subject_logits_list[subject_idx])
    if (tof == 0).all():
        if (feat[:,3:] == 0).all():
            imu_rot_data.append([feat, pred])
        else:
            imu_data.append([feat, pred])
            feat_rot = feat.copy()
            feat_rot[:,3:] = 0
            imu_rot_data.append([feat_rot, pred])
    else:
        if (feat[:,3:] == 0).all():
            all_rot_data.append([feat, tof, pred])
            imu_rot_data.append([feat, pred])
        else:
            all_data.append([feat, tof, pred])
            imu_data.append([feat, pred])
            feat_rot = feat.copy()
            feat_rot[:,3:] = 0
            all_rot_data.append([feat_rot, tof, pred])
            imu_rot_data.append([feat_rot, pred])
    training_step(imu_model_list, imu_rot_model_list, all_model_list, all_rot_model_list, imu_data, imu_rot_data, all_data, all_rot_data)
    gesture = pred_to_label[pred]
    return gesture

if __name__ == "__main__":
    fold_test(use_hungarian=True, use_pseudo_label=True)