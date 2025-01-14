# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple

from crispo.optimizer.meta_prompt import MetaPrompt
from crispo.task.example import Example
from crispo.task.prompt import TaskPrompt


@dataclass
class OproMetaPrompt(MetaPrompt, ABC):
    prompt: str = "Your task is to generate the instruction <INS>."

    def fill(
        self,
        prompt_score_pairs: List[Tuple[TaskPrompt, float]],
        few_shot_examples: List[Example],
        **kwargs,
    ) -> str:
        meta_prompt = [self.prompt, "\n"]

        meta_prompt.append(
            "Below are some previous instructions with their scores. The score ranges from 0 to 100.\n"
        )
        for prompt, score in prompt_score_pairs:
            meta_prompt.append(
                f"\ntext:\n{prompt.prompt}\nscore:\n{round(score * 100)}\n"
            )

        if few_shot_examples:
            meta_prompt.append("Below are some problems.\n")
            for example in few_shot_examples:
                meta_prompt.append(f"\nProblem:\n<INS>\n{example.x}\n")
                meta_prompt.append(f"\nGround truth answer:\n{example.y}\n")
        meta_prompt.append(
            "\n\nGenerate an instruction that"
            " is different from all the instructions <INS> above,"
            " and has a higher score than all the instructions <INS> above."
            " The instruction should begin with <INS> and end with </INS>."
            " The instruction should be concise, effective,"
            " and generally applicable to all problems above."
        )
        meta_prompt_text = "".join(meta_prompt)
        return meta_prompt_text
