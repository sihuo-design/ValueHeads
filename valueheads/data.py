"""Datasets for probing (labeled QA pairs) and for evaluation (questions to answer).

Value -> probing data (Sec. 3.1):
  harmless                       HH-RLHF, single-turn dialogues; chosen = 1, rejected = 0
  honesty                        TruthfulQA (generation), 80/20 split; correct = 1, incorrect = 0
  helpfulness / coherence        HelpSteer2; rating 4 = 1, rating 0-1 = 0
  complexity / verbosity         HelpSteer2; rating 3-4 = 1, rating 0 = 0
"""
import os
import re

import numpy as np
import pandas as pd
from datasets import load_dataset

VALUES = ["harmless", "helpfulness", "honesty", "complexity", "verbosity", "coherence"]
HELPSTEER_VALUES = ["helpfulness", "complexity", "verbosity", "coherence"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Seed of the TruthfulQA train/test split. Probing (head identification) and evaluation must use the
# same split so that evaluation questions are held out from head identification.
SPLIT_SEED = 42
VAL_RATIO = 0.2

# Evaluation sets, and the value the LLM judge scores on each single-value set.
EVAL_SETS = {
    "jailbreak": "harmless",
    "malicious": "harmless",
    "honesty": "honesty",
    **{v: v for v in HELPSTEER_VALUES},
    # multi-value sets: judged on every value
    "hh-rlhf": None,
    "beavertails": None,
}


# ----------------------------------------------------------------------------- HH-RLHF
def _num_human_turns(text):
    return len(re.findall(r"\n\nHuman:\s*(.*?)(?=\n\nAssistant:|\Z)", text, re.S))


def single_turn(example):
    return _num_human_turns(example["chosen"]) == 1 and _num_human_turns(example["rejected"]) == 1


def first_qa(text):
    """(question, answer) of the first turn of an HH-RLHF dialogue."""
    humans = re.findall(r"\n\nHuman:\s*(.*?)(?=\n\nAssistant:|\Z)", text, re.S)
    assistants = re.findall(r"\n\nAssistant:\s*(.*?)(?=\n\nHuman:|\Z)", text, re.S)
    if not humans or not assistants:
        return None, None
    return humans[0].strip(), assistants[0].strip()


def load_hhrlhf(split):
    return load_dataset("Anthropic/hh-rlhf")[split]


# ----------------------------------------------------------------------------- TruthfulQA
def truthfulqa_split(split):
    """Seeded 80/20 split of TruthfulQA (generation); split in {"train", "val"}."""
    ds = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    idxs = np.arange(len(ds))
    train_idxs = np.random.RandomState(SPLIT_SEED).choice(
        idxs, size=int(len(idxs) * (1 - VAL_RATIO)), replace=False)
    train_set = set(train_idxs.tolist())
    val_idxs = np.array([i for i in idxs if i not in train_set])
    return ds.select(train_idxs if split == "train" else val_idxs)


# ----------------------------------------------------------------------------- HelpSteer2
def helpsteer_label(value, rating):
    """Binarized HelpSteer2 label (1 / 0), or None if the rating is discarded."""
    if value in ("helpfulness", "coherence"):
        return 1 if rating == 4 else 0 if rating in (0, 1) else None
    return 1 if rating in (3, 4) else 0 if rating == 0 else None


def load_helpsteer(split):
    return load_dataset("nvidia/HelpSteer2")["train" if split == "train" else "validation"]


# ----------------------------------------------------------------------------- probing data
def probing_set(value, split):
    """Labeled texts (question + answer concatenated) for probing value heads.

    Returns (texts, labels); split in {"train", "val"}.
    """
    texts, labels = [], []
    if value == "harmless":
        for ex in load_hhrlhf("train" if split == "train" else "test"):
            if single_turn(ex):
                texts += [ex["chosen"], ex["rejected"]]
                labels += [1, 0]
    elif value == "honesty":
        for ex in truthfulqa_split(split):
            for answer in ex["correct_answers"]:
                texts.append(f"Question: {ex['question']} Answer: {answer}")
                labels.append(1)
            for answer in ex["incorrect_answers"]:
                texts.append(f"Question: {ex['question']} Answer: {answer}")
                labels.append(0)
    elif value in HELPSTEER_VALUES:
        for ex in load_helpsteer(split):
            label = helpsteer_label(value, ex[value])
            if label is not None:
                texts.append(f"Question: {ex['prompt']}\nAnswer: {ex['response']}")
                labels.append(label)
    else:
        raise ValueError(f"Unknown value: {value}")
    return texts, labels


# ----------------------------------------------------------------------------- evaluation data
def eval_set(name, num_eval=None):
    """Questions of an evaluation set (see EVAL_SETS)."""
    if name == "jailbreak":
        questions = pd.read_csv(os.path.join(DATA_DIR, "jailbreakbench.csv"))["input"].tolist()
    elif name == "malicious":
        questions = pd.read_csv(os.path.join(DATA_DIR, "maliciousinstruct.csv"))["input"].tolist()
    elif name == "honesty":
        questions = [ex["question"] for ex in truthfulqa_split("val")]
    elif name in HELPSTEER_VALUES:
        # validation prompts that carry a binarized label, consecutive duplicates removed
        questions = []
        for ex in load_helpsteer("val"):
            if questions and ex["prompt"] == questions[-1]:
                continue
            if helpsteer_label(name, ex[name]) is not None:
                questions.append(ex["prompt"])
    elif name == "hh-rlhf":
        # single-turn test dialogues, skipping the first 100 raw entries
        ds = load_hhrlhf("test")
        questions = []
        for i in range(100, len(ds)):
            if single_turn(ds[i]):
                q, _ = first_qa(ds[i]["chosen"])
                if q is not None:
                    questions.append(q)
            if len(questions) == (num_eval or 100):
                break
    elif name == "beavertails":
        ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
        questions = [ds[i]["prompt"] for i in range(num_eval or 100)]
    else:
        raise ValueError(f"Unknown eval set: {name}")
    return questions[:num_eval] if num_eval else questions


def prompt_set(name, num_prompts=300):
    """Unlabeled raw prompts of an evaluation set, used to estimate activation std along a direction."""
    if name == "beavertails":
        ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
        return [ds[i]["prompt"] for i in range(num_prompts)]
    if name in ("jailbreak", "malicious"):
        return eval_set(name)
    raise ValueError(f"Unknown prompt set: {name}")
