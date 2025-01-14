# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import List, Tuple

from experiments.narrativeqa.example import NarrativeQaExample
from experiments.narrativeqa.meta_prompt_opro import NarrativeQaOproMetaPrompt
from experiments.narrativeqa.task_prompt import (
    QUESTION_PLACEHOLDER,
    CONTEXT_PLACEHOLDER,
)
from crispo.task.prompt import TaskPrompt

_P = """
Your task is to generate the instruction <instruction> for {task}.

Below are some examples:
{examples}

Below are some previous instructions with their scores and critiques.
{instructions}

Generate an instruction that is different from all the instructions above, and has a higher score than all the instructions above.
It should be concise, effective, and generally applicable to all examples above.

Draft your new instruction step by step:

1. Compare high-score instructions to low-score ones, identify what suggestions could have improved them. List them in <suggestion> tags.
2. Apply the suggestions and draft a new instruction aiming for a higher score.
3. Be creative and vary the wording, paraphrase, position of {CONTEXT_PLACEHOLDER} and {QUESTION_PLACEHOLDER}, phrase order, grammar, sentence order, which specific examples to give, etc.
4. Write your final new instruction in <instruction> tags.
""".strip()

_E = """
<example>
<instruction>?</instruction>
<input>
{input}
</input>
<output>
{output}
</output>
</example>
""".strip()

_I = """
<rated_instruction>
<instruction>{instruction}</instruction>
<score>{score}</score>
<critique>
{critique}
</critique>
</rated_instruction>
""".strip()


@dataclass
class NarrativeQaCritiqueMetaPrompt(NarrativeQaOproMetaPrompt):
    prompt = _P
    task = "a reading comprehension task where a reader answers a question by integrating information and reasoning across a context document"

    # noinspection PyMethodOverriding
    def fill(
        self,
        prompt_score_pairs: List[Tuple[TaskPrompt, float]],
        few_shot_examples: List[NarrativeQaExample],
        critiques,
        **kwargs,
    ) -> str:
        examples = self.format_examples(few_shot_examples)

        instructions = []
        for (prompt, score), critique in zip(prompt_score_pairs, critiques):
            instructions.append(
                _I.format(
                    instruction=prompt,
                    score=score if isinstance(score, str) else f"{score:.2%}",
                    critique=critique,
                )
            )

        meta_prompt_text = self.prompt.format(
            examples="\n".join(examples),
            instructions="\n".join(instructions),
            task=self.task,
            CONTEXT_PLACEHOLDER=CONTEXT_PLACEHOLDER,
            QUESTION_PLACEHOLDER=QUESTION_PLACEHOLDER,
        )
        return meta_prompt_text
