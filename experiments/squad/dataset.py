# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import random

from datasets import load_dataset

from experiments.nq.example import RAGExample


def load_squad(split: str, num_samples: int):
    assert split in ["test", "train", "dev"]
    examples = []
    dataset = load_dataset("rajpurkar/squad")

    for example in dataset.select(range(num_samples)):
        examples.append(
            RAGExample(
                x=(example["question"], [example["context"]]),
                y=example["answers"]["text"],
            )
        )
    return examples


def load_squad(split: str):
    examples = []
    assert split in ["train", "validation"]
    dataset = load_dataset("rajpurkar/squad")[split]
    for example in dataset:
        examples.append(
            RAGExample(
                x=(example["question"], [example["context"]]), y=example["answers"]
            )
        )

    return examples


def load_squad_shuffled_standard_train_dev_test(random_seed=42):
    random.seed(random_seed)
    train = random.sample(load_squad(split="train"), 100)
    train, dev = train[:50], train[50:]
    test = random.sample(load_squad(split="validation"), 500)
    return train, dev, test


def load_squad_debug_train_dev_test(random_seed=42):
    random.seed(random_seed)
    train = random.sample(load_squad(split="train"), 10)
    train, dev = train[:5], train[5:]
    test = random.sample(load_squad(split="validation"), 10)
    return train, dev, test
