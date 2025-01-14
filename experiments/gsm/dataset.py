# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments.gsm.example import GsmExample, Output
from datasets import load_dataset
import re

_CALC = re.compile("<<.*?>>")


def load_gsm8k(split: str, num_samples: int):
    examples = []
    if split == "validation":
        split = "train"
        num_samples = -num_samples
    dataset = load_dataset("gsm8k", name="main", split=split)
    dataset = dataset.shuffle(seed=233)
    indices = (
        range(num_samples)
        if num_samples > 0
        else range(len(dataset) + num_samples, len(dataset))
    )
    for example in dataset.select(indices):
        y = example["answer"]
        reasoning, answer = y.split("####", 1)
        answer: str = answer.strip()
        answer = answer.replace(",", "")
        reasoning = reasoning.strip()
        reasoning = _CALC.sub("", reasoning)  # Remove calculator
        examples.append(
            GsmExample(
                x=example["question"], y=Output(label=answer, reasoning=reasoning)
            )
        )
    return examples


def load_gsm8k_standard_train_dev_test():
    # OPRO paper train: 261, test 7212, no dev
    train = load_gsm8k(split="train", num_samples=200)
    dev = load_gsm8k(split="validation", num_samples=200)
    test = load_gsm8k(split="test", num_samples=500)
    return train, dev, test


def load_gsm8k_quick_train_dev_test():
    train = load_gsm8k(split="train", num_samples=60)
    dev = load_gsm8k(split="validation", num_samples=50)
    test = load_gsm8k(split="test", num_samples=100)
    return train, dev, test


def main():
    for ds in load_gsm8k_standard_train_dev_test():
        for each in ds:
            each: GsmExample = each
            if not each.y.label.isdigit():
                print(each.y.label)


if __name__ == "__main__":
    main()
