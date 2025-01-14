# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import abc
from dataclasses import dataclass
from typing import List, Tuple

from crispo.optimizer.crispo_meta_prompt import CriSPOMetaPrompt
from crispo.task.example import Example
from crispo.task.prompt import TaskPrompt


@dataclass
class CriSPOASTMetaPrompt(CriSPOMetaPrompt, abc.ABC):
    prompt: str = (
        f"""
Your task is to append postscript instruction to the main instruction for a task.

The main instruction is:

<instruction>{{main_prompt}}</instruction>

Below are some examples:
{{examples}}

Below are some previous postscripts with their scores and critiques.
{{postscripts}}

Generate a short postscript that is different from all the postscripts above, and has a higher score than all the postscripts above.
It should be concise, effective, and generally applicable to all examples above.

Draft your new postscript step by step:

1. Compare high-score postscripts to low-score ones, identify what suggestions could have improved them. List them in <suggestion> tags.
2. Apply the suggestions and draft a new postscript aiming for a higher score.
3. Be creative and vary the wording, paraphrase, phrase order, grammar, sentence order, etc.
4. Write your final new short postscript in <postscript> tags.
""".strip()
    )
    main_prompt: str = ""

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
                examples.append(self.format_example(example))

        instructions = []
        for (prompt, score), critique in zip(prompt_score_pairs, critiques):
            instructions.append(
                """
<rated_postscript>
<postscript>{postscript}</postscript>
<score>{score}</score>
<critique>
{critique}
</critique>
</rated_postscript>
                """.strip().format(
                    postscript=prompt,
                    score=score if isinstance(score, str) else f"{score:.2%}",
                    critique=critique,
                )
            )

        meta_prompt_text = self.prompt.format(
            main_prompt=self.main_prompt,
            examples="\n".join(examples),
            postscripts="\n".join(instructions),
        )
        return meta_prompt_text
