# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments.summarization.example import SummarizationExample
from datasets import load_dataset
import random


def load_samsum_full_train_dev_test():
    train = load_samsum(split="train")
    dev = load_samsum(split="validation")
    test = load_samsum(split="test")
    return train, dev, test


def load_samsum_shuffled_quick_train_dev_test(random_seed=42):
    random.seed(random_seed)
    train = random.sample(load_samsum(split="train"), 50)
    dev = random.sample(load_samsum(split="validation"), 50)
    test = random.sample(load_samsum(split="test"), 100)
    return train, dev, test


def load_samsum_shuffled_sample_train_dev_test(
    random_seed=42, sample_size=50, random_seed_train_dev=42
):
    random.seed(random_seed)
    train = random.sample(load_samsum(split="train"), sample_size)
    dev = random.sample(load_samsum(split="validation"), sample_size)
    test = random.sample(load_samsum(split="test"), 100)

    random.seed(random_seed_train_dev)
    train = random.sample(load_samsum(split="train"), sample_size)
    dev = random.sample(load_samsum(split="validation"), sample_size)
    return train, dev, test


def load_samsum_shuffled_standard_train_dev_test(random_seed=42):
    random.seed(random_seed)
    train = random.sample(load_samsum(split="train"), 50)
    dev = random.sample(load_samsum(split="validation"), 50)
    test = random.sample(load_samsum(split="test"), 500)
    return train, dev, test


def load_samsum_debug_train_dev_test(random_seed=42):
    random.seed(random_seed)
    train = random.sample(load_samsum(split="train"), 2)
    dev = random.sample(load_samsum(split="validation"), 2)
    test = random.sample(load_samsum(split="test"), 2)
    return train, dev, test


def load_samsum(split: str):
    examples = []
    dataset = load_dataset("samsum", trust_remote_code=True)[split]
    for example in dataset:
        examples.append(
            SummarizationExample(x=example["dialogue"], y=example["summary"])
        )
    return examples


def main():
    from experiments import cdroot

    cdroot()
    load_samsum_shuffled_quick_train_dev_test()


if __name__ == "__main__":
    main()
