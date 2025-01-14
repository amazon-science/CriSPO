# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Union, Any
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag
from experiments.summarization.constants import ARTICLE_PLACEHOLDER


@dataclass(eq=True, frozen=True)
class SummarizationTaskPromptNoPlaceholder(TaskPrompt):
    def fill(self, x: Union[str, Any]) -> str:
        buffer = [self.prompt, x]
        if "<summary>" not in self.prompt:
            buffer.append("Enclose your summary within <summary> tags.")
        return "\n".join(buffer)

    def parse(self, generation: str, x=None):
        found = extract_xml_tag(generation, "summary")
        if found:
            return found
        return generation

    def __str__(self) -> str:
        return self.prompt


@dataclass(eq=True, frozen=True)
class SummarizationTaskPromptWithPlaceholder(TaskPrompt):
    def fill(self, x: Union[str, Any]) -> str:
        assert ARTICLE_PLACEHOLDER in self.prompt
        filled_prompt = self.prompt.replace(
            ARTICLE_PLACEHOLDER,
            f"<input>\n{x}\n</input>" if "<input>" not in self.prompt else x,
        )
        return filled_prompt

    def parse(self, generation: str):
        found = extract_xml_tag(generation, "summary")
        if found:
            return found
        return generation

    def __str__(self) -> str:
        return self.prompt
