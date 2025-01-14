# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import random

import tqdm
from datasets import load_dataset

from experiments.summarization.example import SummarizationExample

MEETING_BANK_X_LEN = 5000
MEETING_BANK_Y_LEN = 512


def load_pubmed(split: str, trunc: bool = False):
    examples = []
    dataset = load_dataset("scientific_papers", "pubmed", trust_remote_code=True)[split]
    for example in tqdm.tqdm(dataset):
        if example["article"].strip() and example["abstract"].strip():
            x = example["article"]
            y = example["abstract"]
            if trunc:
                x = " ".join(x.split()[:MEETING_BANK_X_LEN])
                y = " ".join(y.split()[:MEETING_BANK_Y_LEN])

            examples.append(SummarizationExample(x=x, y=y))
    return examples


def load_pubmed_quick_train_dev_test(seed=42, trunc=False):
    random.seed(seed)
    train = load_pubmed(split="train", trunc=trunc)
    train = random.sample(train, 50)
    dev = load_pubmed(split="validation", trunc=trunc)
    dev = random.sample(dev, 50)
    test = load_pubmed(split="test", trunc=trunc)
    test = random.sample(test, 100)
    return train, dev, test


def load_pubmed_standard_train_dev_test(seed=42, trunc=False):
    random.seed(seed)
    train = load_pubmed(split="train", trunc=trunc)
    train = random.sample(train, 100)
    dev = load_pubmed(split="validation", trunc=trunc)
    dev = random.sample(dev, 100)
    test = load_pubmed(split="test", trunc=trunc)
    test = random.sample(test, 500)
    return train, dev, test
