# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import random

from datasets import load_dataset

from experiments.summarization.example import SummarizationExample

MEETING_BANK_X_LEN = 5000
MEETING_BANK_Y_LEN = 256


def load_meetingbank_full_train_dev_test(trunc=False):
    train = load_meetingbank(split="train", trunc=trunc)
    dev = load_meetingbank(split="validation", trunc=trunc)
    test = load_meetingbank(split="test", trunc=trunc)
    return train, dev, test


def load_meetingbank_shuffled_quick_train_dev_test(random_seed=42, trunc=False):
    random.seed(random_seed)
    train = random.sample(load_meetingbank(split="train", trunc=trunc), 50)
    dev = random.sample(load_meetingbank(split="validation", trunc=trunc), 50)
    test = random.sample(load_meetingbank(split="test", trunc=trunc), 100)
    return train, dev, test


def load_meetingbank_shuffled_standard_train_dev_test(random_seed=42, trunc=False):
    random.seed(random_seed)
    train = random.sample(load_meetingbank(split="train", trunc=trunc), 50)
    dev = random.sample(load_meetingbank(split="validation", trunc=trunc), 50)
    test = random.sample(load_meetingbank(split="test", trunc=trunc), 500)
    return train, dev, test


def load_meetingbank_debug_train_dev_test(random_seed=42, trunc=False):
    random.seed(random_seed)
    train = random.sample(load_meetingbank(split="train", trunc=trunc), 2)
    dev = random.sample(load_meetingbank(split="validation", trunc=trunc), 2)
    test = random.sample(load_meetingbank(split="test", trunc=trunc), 2)
    return train, dev, test


def load_meetingbank(split: str, trunc: bool = False):
    examples = []
    dataset = load_dataset("huuuyeah/meetingbank")[split]
    for example in dataset:
        x = example["transcript"]
        y = example["summary"]
        if trunc:
            x = " ".join(x.split()[:MEETING_BANK_X_LEN])
            y = " ".join(y.split()[:MEETING_BANK_Y_LEN])

        examples.append(SummarizationExample(x=x, y=y))
    return examples


def main():
    from experiments import cdroot

    cdroot()
    load_meetingbank_shuffled_quick_train_dev_test()


if __name__ == "__main__":
    main()
