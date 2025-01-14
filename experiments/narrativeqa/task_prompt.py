# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

from experiments.narrativeqa.example import Input
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag

QUESTION_PLACEHOLDER = "QUESTION_PLACEHOLDER"
CONTEXT_PLACEHOLDER = "CONTEXT_PLACEHOLDER"

_P = f"""
Read the following context and question then write your answer in <answer> XML tags.

Context: {CONTEXT_PLACEHOLDER}
Question: {QUESTION_PLACEHOLDER}
"""


@dataclass(eq=True, frozen=True)
class NarrativeQaTaskPrompt(TaskPrompt):
    prompt: str = _P

    def fill(self, x: Input) -> str:
        return self.prompt.replace(QUESTION_PLACEHOLDER, x.question).replace(
            CONTEXT_PLACEHOLDER, x.context
        )

    def parse(self, generation: str):
        return extract_xml_tag(generation, "answer") or generation

    def __str__(self) -> str:
        return self.prompt
