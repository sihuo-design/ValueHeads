"""Value directions and steering vectors (Sec. 5).

Single value:  dir_l^h = E_pos[x_l^h] - E_neg[x_l^h]                                     (Eq. 2)
Multi value:   dir_{l,V}^h = sum_k lambda_k dir_{l,k}^h, over the values whose top heads contain h
Both are normalized to unit length and scaled by sigma_l^h, the std of the activations projected onto
the direction; at inference x_l^h <- x_l^h + alpha * sigma_l^h * dir_l^h.
"""
import numpy as np
import torch


def mass_mean_directions(acts_list, labels_list, num_heads, chunk=512):
    """Positive-minus-negative mean activation of every head.

    acts_list / labels_list: activation arrays (N, L, H * D) and labels, pooled together
    (we pool the probing train and val splits). Returns (L * H, D).
    """
    pos_sum = neg_sum = None
    n_pos = n_neg = 0
    for acts, labels in zip(acts_list, labels_list):
        labels = np.asarray(labels)
        for start in range(0, len(acts), chunk):
            x = np.asarray(acts[start:start + chunk], dtype=np.float64)
            y = labels[start:start + chunk]
            p, n = x[y == 1].sum(axis=0), x[y == 0].sum(axis=0)
            pos_sum = p if pos_sum is None else pos_sum + p
            neg_sum = n if neg_sum is None else neg_sum + n
            n_pos += int((y == 1).sum())
            n_neg += int((y == 0).sum())
    direction = pos_sum / n_pos - neg_sum / n_neg              # (L, H * D)
    num_layers = direction.shape[0]
    return direction.reshape(num_layers * num_heads, -1).astype(np.float32)


def steering_vectors(value_heads, value_dirs, weights, std_acts, num_heads, head_dim):
    """Builds per-layer steering vectors for one or several values.

    value_heads: {value: [flat_idx, ...]}   top heads of each value
    value_dirs:  {value: (L * H, D) array}  mass-mean directions of each value
    weights:     {value: lambda}
    std_acts:    (N, L, H * D) activations used to estimate sigma along each direction
    Returns {layer: tensor(H * D)} with sigma * unit direction on intervened heads, 0 elsewhere.
    """
    head_values = {}
    for value, heads in value_heads.items():
        if weights[value] == 0:
            continue
        for idx in heads:
            head_values.setdefault(idx, []).append(value)

    vectors = {}
    for idx, values in sorted(head_values.items()):
        layer, head = divmod(idx, num_heads)
        direction = sum(weights[v] * value_dirs[v][idx] for v in values)
        direction = direction / np.linalg.norm(direction)
        acts = np.asarray(std_acts[:, layer, head * head_dim:(head + 1) * head_dim], dtype=np.float32)
        sigma = np.std(acts @ direction, ddof=1)
        vec = vectors.setdefault(layer, torch.zeros(num_heads * head_dim))
        vec[head * head_dim:(head + 1) * head_dim] = torch.from_numpy(sigma * direction)
    return vectors
