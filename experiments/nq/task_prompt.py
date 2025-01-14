# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Tuple, List

from experiments.nq.constants import (
    QUESTION_PLACEHOLDER,
    CONTEXT_PLACEHOLDER,
    EXAMPLES_PLACEHOLDER,
)
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag
from experiments.nq.example import encode_context


@dataclass(eq=True, frozen=True)
class RAGTaskPrompt(TaskPrompt):
    def __init__(self, prompt: str):
        if "<answer>" not in prompt:
            prompt += "Write your answer in <answer> tags."
        super().__init__(prompt)

    def fill(self, x: Tuple[str, List[str]]) -> str:
        filled_prompt = (
            f"<question>\n{x[0]}\n</question>\n\n"
            + encode_context(x)
            + "\n\n"
            + self.prompt
        )
        return filled_prompt

    def parse(self, generation: str):
        found = extract_xml_tag(generation, "answer")
        if found:
            return found
        return generation

    def __str__(self) -> str:
        return self.prompt


@dataclass(eq=True, frozen=True)
class RAGTaskPromptUniversalTemplate(TaskPrompt):

    def __init__(self, prompt: str):
        if "INSERT_CONTEXTS_HERE" in prompt:
            prompt = prompt.replace("INSERT_CONTEXTS_HERE", CONTEXT_PLACEHOLDER)
        if CONTEXT_PLACEHOLDER not in prompt:
            prompt = f"{CONTEXT_PLACEHOLDER}\n\n{prompt}"
        if QUESTION_PLACEHOLDER not in prompt:
            prompt = f"{QUESTION_PLACEHOLDER}\n\n{prompt}"
        if "<answer>" not in prompt:
            prompt += "Write your answer in <answer> tags."
        super().__init__(prompt)

    def fill(self, x: Tuple[str, List[str]]) -> str:
        filled_prompt = self.prompt.replace(
            QUESTION_PLACEHOLDER, f"\n<question>\n{x[0]}\n</question>\n"
        )
        filled_prompt = filled_prompt.replace(CONTEXT_PLACEHOLDER, encode_context(x))
        return filled_prompt

    def parse(self, generation: str):
        found = extract_xml_tag(generation, "answer")
        if found:
            return found
        return generation

    def __str__(self) -> str:
        return self.prompt
