# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# -*- coding:utf-8 -*-
from dataclasses import dataclass
from typing import Union, Any

from experiments.summarization.constants import ARTICLE_PLACEHOLDER
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder


@dataclass(eq=True, frozen=True)
class SummarizationTaskPromptSuffix(SummarizationTaskPromptWithPlaceholder):
    main_prompt: str

    def fill(self, x: Union[str, Any]) -> str:
        prompt = self.main_prompt.replace(
            ARTICLE_PLACEHOLDER,
            f"<input>\n{x}\n</input>" if "<input>" not in self.prompt else x,
        )
        prompt = "\n\n".join([prompt, self.prompt])
        return prompt
