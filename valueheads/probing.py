"""Linear probes on per-head activations (Sec. 3.2)."""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def feature_path(feature_dir, model_name, name, split):
    return os.path.join(feature_dir, f"{model_name}_{name}_{split}")


def load_features(feature_dir, model_name, name, split, mmap=True):
    """Head activations (N, num_layers, num_heads * head_dim) and labels (N,) or None."""
    prefix = feature_path(feature_dir, model_name, name, split)
    acts = np.load(prefix + "_head_wise.npy", mmap_mode="r" if mmap else None)
    labels = np.load(prefix + "_labels.npy") if os.path.exists(prefix + "_labels.npy") else None
    return acts, labels


def train_head_probes(train_acts, y_train, val_acts, y_val, num_heads, seed=42, layer_bias=True):
    """Trains one logistic-regression probe per (layer, head) and returns test accuracies.

    With `layer_bias`, each head's features are concatenated with the mean activation over the heads
    of its layer. Returns an array of shape (num_layers * num_heads,), indexed by layer * num_heads + head.
    """
    num_layers = train_acts.shape[1]
    accs = np.zeros(num_layers * num_heads)
    for layer in tqdm(range(num_layers), desc="probing layers"):
        x_tr = np.asarray(train_acts[:, layer], dtype=np.float32).reshape(len(train_acts), num_heads, -1)
        x_va = np.asarray(val_acts[:, layer], dtype=np.float32).reshape(len(val_acts), num_heads, -1)
        mean_tr, mean_va = x_tr.mean(axis=1), x_va.mean(axis=1)
        for head in range(num_heads):
            f_tr, f_va = x_tr[:, head], x_va[:, head]
            if layer_bias:
                f_tr = np.concatenate([f_tr, mean_tr], axis=-1)
                f_va = np.concatenate([f_va, mean_va], axis=-1)
            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(f_tr, y_train)
            accs[layer * num_heads + head] = accuracy_score(y_val, clf.predict(f_va))
    return accs


def save_head_scores(accs, path):
    """Saves {flat_head_idx: accuracy}, sorted by accuracy (descending)."""
    order = np.argsort(accs)[::-1]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({int(i): float(accs[i]) for i in order}, f)


def head_scores_path(score_dir, model_name, value):
    return os.path.join(score_dir, f"{model_name}_{value}_heads.json")


def load_head_scores(score_dir, model_name, value):
    """{flat_head_idx: accuracy}, sorted by accuracy (descending)."""
    with open(head_scores_path(score_dir, model_name, value)) as f:
        return {int(k): v for k, v in json.load(f).items()}


def top_heads(scores, k):
    """Flat indices of the k highest-scoring heads."""
    return list(scores)[:k]


def group_by_layer(flat_heads, num_heads):
    """[flat_idx, ...] -> {layer: [head, ...]}."""
    groups = {}
    for idx in flat_heads:
        groups.setdefault(idx // num_heads, []).append(idx % num_heads)
    return groups
