# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import random

from datasets import load_dataset

from experiments.summarization.example import SummarizationExample

random.seed(42)


def load_cnn(split: str):
    examples = []
    dataset = load_dataset("cnn_dailymail", "2.0.0", split=split)
    for example in dataset:
        examples.append(
            SummarizationExample(x=example["article"], y=example["highlights"])
        )
    return examples


def load_cnn_standard_train_dev_test():
    train = load_cnn(split="train[:100]")
    dev = load_cnn(split="validation[:100]")
    test = load_cnn(split="test[:500]")
    return train, dev, test


def load_cnn_standard_train_dev_test_shuffled(seed=42):
    random.seed(seed)
    train = load_cnn(split="train")
    train = random.sample(train, 100)
    dev = load_cnn(split="validation")
    dev = random.sample(dev, 100)
    test = load_cnn(split="test")
    test = random.sample(test, 500)
    return train, dev, test


def load_cnn_quick_train_dev_test():
    train = load_cnn(split="train[:60]")
    dev = load_cnn(split="validation[:50]")
    test = load_cnn(split="test[:100]")
    return train, dev, test


def load_cnn_debug_train_dev_test():
    train = load_cnn(split="train[:3]")
    dev = load_cnn(split="validation[:3]")
    test = load_cnn(split="test[:3]")
    return train, dev, test


def main():
    from experiments import cdroot

    cdroot()
    dataset = load_cnn("test[:10]")
    print(len(dataset))


if __name__ == "__main__":
    main()
