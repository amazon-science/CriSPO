# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List

from experiments.narrativeqa.example import NarrativeQaExample, Input
from datasets import load_dataset


def load_narrativeqa(split: str, num_samples: int) -> List[NarrativeQaExample]:
    examples = []
    ds = load_dataset("narrativeqa", split=split)
    ds = ds.shuffle(seed=233)
    indices = (
        range(num_samples) if num_samples > 0 else range(len(ds) + num_samples, len(ds))
    )
    ds = ds.select(indices)
    for example in ds:
        examples.append(
            NarrativeQaExample(
                x=Input(
                    question=example["question"]["text"],
                    context=example["document"]["summary"]["text"],
                ),
                y=[x["text"] for x in example["answers"]],
            )
        )
    return examples


def load_narrativeqa_standard_train_dev_test():
    train = load_narrativeqa(split="train", num_samples=100)
    dev = load_narrativeqa(split="validation", num_samples=50)
    test = load_narrativeqa(split="test", num_samples=500)
    return train, dev, test


def load_narrativeqa_debug_train_dev_test():
    train = load_narrativeqa(split="train", num_samples=5)
    dev = load_narrativeqa(split="validation", num_samples=3)
    test = load_narrativeqa(split="test", num_samples=3)
    return train, dev, test


def main():
    import statistics
    from experiments.summarization.utilities.tok_eos import tokenize

    def length_of(text: str):
        return len(sum(tokenize(text).tokens, []))

    train, dev, test = load_narrativeqa_standard_train_dev_test()
    inputs = []
    outputs = []
    for each in test:
        each: NarrativeQaExample
        inputs.append(length_of("\n".join([each.x.question, each.x.context])))
        outputs.append(length_of(each.y[0]))
    print(f"Inputs: {statistics.mean(inputs):.1f}")
    print(f"Outputs: {statistics.mean(outputs):.1f}")


if __name__ == "__main__":
    main()
