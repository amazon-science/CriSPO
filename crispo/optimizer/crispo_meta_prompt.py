# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import abc
from dataclasses import dataclass
from typing import List, Tuple

from crispo.optimizer.meta_prompt import MetaPrompt
from crispo.task.example import Example
from crispo.task.prompt import TaskPrompt


@dataclass
class CriSPOMetaPrompt(MetaPrompt, abc.ABC):
    prompt: str = (
        """
Your task is to optimize the instruction for a task.

Below are some examples:
{examples}

Below are some previous instructions with their scores and critiques.
{instructions}

Generate an instruction that is different from all the instructions above, and has a higher score than all the instructions above.
It should be concise, effective, and generally applicable to all examples above.

Draft your new instruction step by step:

1. Compare high-score instructions to low-score ones, identify what suggestions could have improved them. List them in <suggestion> tags.
2. Apply the suggestions and draft a new instruction aiming for a higher score.
3. Be creative and vary the wording, paraphrase, position of placeholders, phrase order, grammar, sentence order, which specific example summaries to give, etc.
4. Write your final new instruction in <instruction> tags.
""".strip()
    )

    # noinspection PyMethodOverriding
    def fill(
        self,
        prompt_score_pairs: List[Tuple[TaskPrompt, float]],
        few_shot_examples: List[Example],
        critiques,
        **kwargs,
    ) -> str:
        examples = []
        if few_shot_examples:
            for i, example in enumerate(few_shot_examples):
                examples.append(
                    f"""<example>
<instruction>?</instruction>
{self.format_example(example)}
</example>"""
                )

        instructions = []
        for (prompt, score), critique in zip(prompt_score_pairs, critiques):
            instructions.append(self.format_instruction(prompt, score, critique))

        meta_prompt_text = self.prompt.format(
            examples="\n".join(examples),
            instructions="\n".join(instructions),
        )
        return meta_prompt_text

    def format_example(self, example: Example) -> str:
        return example.to_xml()

    def format_instruction(
        self, prompt: TaskPrompt, score: float, critique: str
    ) -> str:
        return """<rated_instruction>
<instruction>{instruction}</instruction>
<score>{score}</score>
<critique>
{critique}
</critique>
</rated_instruction>""".format(
            instruction=prompt,
            score=score if isinstance(score, str) else f"{score:.2%}",
            critique=critique,
        )
