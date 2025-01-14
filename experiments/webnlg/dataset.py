# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from datasets import load_dataset

from experiments.webnlg.example import WebNLGExample


def load_webnlg(split: str, num_samples: int):
    examples = []
    ds = load_dataset(
        "web_nlg", "webnlg_challenge_2017", trust_remote_code=True, split=split
    )
    ds = ds.filter(lambda x: x["lex"]["text"], num_proc=8)
    ds = ds.shuffle(seed=233)
    indices = (
        range(num_samples) if num_samples > 0 else range(len(ds) + num_samples, len(ds))
    )
    ds = ds.select(indices)
    for example in ds:
        examples.append(
            WebNLGExample(
                x=example["modified_triple_sets"]["mtriple_set"][0],
                y=example["lex"]["text"],
            )
        )
    return examples


def load_webnlg_standard_train_dev_test():
    train = load_webnlg(split="train", num_samples=100)
    dev = load_webnlg(split="dev", num_samples=100)
    test = load_webnlg(split="test", num_samples=500)
    return train, dev, test


def load_webnlg_debug_train_dev_test():
    train = load_webnlg(split="train", num_samples=10)
    dev = load_webnlg(split="dev", num_samples=10)
    test = load_webnlg(split="test", num_samples=50)
    return train, dev, test


def main():
    from experiments import cdroot

    cdroot()
    train, dev, test = load_webnlg_standard_train_dev_test()
    print(len(train), len(dev), len(test))


if __name__ == "__main__":
    main()
