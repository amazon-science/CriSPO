# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import random

from experiments.nq.example import RAGExample


def load_natural_questions(split: str):
    examples = []
    assert split in ["test", "train", "dev"]
    with open(f"data/NQ/{split}.json") as f:
        data = json.load(f)

    for item in data:
        examples.append(
            RAGExample(
                x=(item["question"], [x["text"] for x in item["ctxs"]]),
                y=item["answers"],
            )
        )
    print(split, len(examples))

    return examples


def load_natural_questions_debug_train_dev_test(seed=42):
    random.seed(seed)
    train = load_natural_questions(split="train")
    train = random.sample(train, 10)
    dev = load_natural_questions(split="dev")
    dev = random.sample(dev, 10)
    test = load_natural_questions(split="test")
    test = random.sample(test, 20)

    return train, dev, test


def load_natural_questions_quick_train_dev_test(seed=42):
    random.seed(seed)
    train = load_natural_questions(split="train")
    train = random.sample(train, 50)
    dev = load_natural_questions(split="dev")
    dev = random.sample(dev, 50)
    test = load_natural_questions(split="test")
    test = random.sample(test, 100)

    return train, dev, test


def load_natural_questions_standard_train_dev_test(seed=42):
    random.seed(seed)
    train = load_natural_questions(split="train")
    train = random.sample(train, 100)
    dev = load_natural_questions(split="validation")
    dev = random.sample(dev, 100)
    test = load_natural_questions(split="test")
    test = random.sample(test, 500)

    return train, dev, test
