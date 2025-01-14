# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import random

import pandas as pd

from experiments.summarization.example import SummarizationExample


def load_clinical_note_sum(split: str):
    examples = []
    assert split in ["test", "train", "validation"]
    if split == "test":
        data = pd.read_csv("data/datasets/aci-bench/clinicalnlp_taskB_test1.csv")
    elif split == "train":
        data = pd.read_csv("data/datasets/aci-bench/train.csv")

    elif split == "validation":
        data = pd.read_csv("data/datasets/aci-bench/valid.csv")

    for i, row in data.iterrows():
        examples.append(SummarizationExample(x=row["dialogue"], y=row["note"]))
    # print(split, len(examples))
    return examples


def load_clinical_note_sum_standard_train_dev_test(seed=42):
    random.seed(seed)
    train = load_clinical_note_sum(split="train")
    dev = load_clinical_note_sum(split="validation")
    test = load_clinical_note_sum(split="test")
    return train, dev, test
