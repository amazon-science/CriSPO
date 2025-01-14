# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments.medmcqa.example import MedMcqaExample


def _input_of_example(example: MedMcqaExample):
    return example.x


def _output_of_example(example: MedMcqaExample):
    return f"{example.y.reasoning}\n<answer>{example.y.label}</answer>"


def _score_of(prompt: dict):
    return prompt["score"]


def example_template_func(example: MedMcqaExample):
    return (
        f"<question>{example.x}</question>\n"
        f"<reasoning>{example.y.reasoning}</reasoning>\n"
        f"<answer>{example.y.label}</answer>"
    )
