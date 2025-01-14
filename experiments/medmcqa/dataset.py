# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments.medmcqa.example import MedMcqaExample, Output
from datasets import load_dataset
import logging


def _format_options(example: dict, template: str = "({option}) {content}"):
    example["options"] = "\n".join(
        template.format(option=k, content=v) for k, v in example["options_dict"].items()
    )
    return example


def _format_prompt(example: dict, template: str = "{question}\n{options}"):
    example["prompt"] = template.format(
        question=example["question"], options=example["options"]
    )
    return example


def process_to_mcqa(example: dict):
    example["options_dict"] = {
        "A": example["opa"],
        "B": example["opb"],
        "C": example["opc"],
        "D": example["opd"],
    }
    if isinstance(example["cop"], int) and 0 <= example["cop"] <= 3:
        example["correct_option"] = "ABCD"[example["cop"]]
    else:
        logging.warning(f"Unable to parse cop: {example['cop']}")
    example["explanation"] = example["exp"]
    return example


def load_medmcqa(split: str, num_samples: int, num_proc=None):
    if split == "test":
        split = "validation"
        num_samples = -num_samples
    examples = []
    ds = load_dataset("openlifescienceai/medmcqa", split=split)
    ds = ds.filter(
        lambda x: x["exp"] and "," in x["exp"],
        num_proc=num_proc,
        desc="Filtering out no explanation",
    )
    ds = ds.shuffle(seed=233)
    indices = (
        range(num_samples) if num_samples > 0 else range(len(ds) + num_samples, len(ds))
    )
    ds = ds.select(indices)
    ds = ds.map(process_to_mcqa, num_proc=num_proc, desc="Processing")
    ds = ds.map(
        _format_options,
        num_proc=num_proc,
        desc="Templating Options",
    )
    ds = ds.map(
        _format_prompt,
        num_proc=num_proc,
        desc="Templating Question",
    )
    for example in ds:
        y = example["correct_option"]
        reasoning = example["explanation"]
        reasoning = reasoning.strip()
        examples.append(
            MedMcqaExample(x=example["prompt"], y=Output(label=y, reasoning=reasoning))
        )
    return examples


def load_medmcqa_standard_train_dev_test():
    train = load_medmcqa(split="train", num_samples=100)
    dev = load_medmcqa(split="validation", num_samples=50)
    test = load_medmcqa(split="test", num_samples=500)
    return train, dev, test


def main():
    import statistics
    from experiments.summarization.utilities.tok_eos import tokenize

    def length_of(text: str):
        return len(sum(tokenize(text).tokens, []))

    train, dev, test = load_medmcqa_standard_train_dev_test()
    inputs = []
    outputs = []
    for each in test:
        each: MedMcqaExample
        inputs.append(length_of(each.x))
        outputs.append(length_of("\n".join([each.y.reasoning, each.y.label])))
    print(f"Inputs: {statistics.mean(inputs):.1f}")
    print(f"Outputs: {statistics.mean(outputs):.1f}")


if __name__ == "__main__":
    main()
