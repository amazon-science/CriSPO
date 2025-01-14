# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Optional, List, Tuple

from experiments.gsm.example import GsmExample
from experiments.gsm.task_prompt import GsmTaskPrompt, QUESTION_PLACEHOLDER
from crispo.optimizer.opro_meta_prompt import OproMetaPrompt
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag

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
3. Be creative and vary the wording, paraphrase, position of {QUESTION_PLACEHOLDER}, phrase order, grammar, sentence order, which specific examples to give, etc.
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
class GsmCritiqueMetaPrompt(OproMetaPrompt):
    prompt = _P
    task = "grade school math word problems"

    # noinspection PyMethodOverriding
    def fill(
        self,
        prompt_score_pairs: List[Tuple[TaskPrompt, float]],
        few_shot_examples: List[GsmExample],
        critiques,
        **kwargs,
    ) -> str:
        examples = []
        if few_shot_examples:
            for i, example in enumerate(few_shot_examples):
                examples.append(_E.format(input=example.x, output=example.y.label))

        instructions = []
        for (prompt, score), critique in zip(prompt_score_pairs, critiques):
            instructions.append(
                _I.format(
                    instruction=prompt,
                    score=score if isinstance(score, str) else f"{score:.2%}",
                    critique=critique,
                )
            )

        meta_prompt_text = _P.format(
            examples="\n".join(examples),
            instructions="\n".join(instructions),
            task=self.task,
            QUESTION_PLACEHOLDER=QUESTION_PLACEHOLDER,
        )
        return meta_prompt_text

    def parse(self, generation: str) -> Optional[TaskPrompt]:
        new_inst = extract_xml_tag(generation, "instruction")
        if not new_inst:
            return None
        if QUESTION_PLACEHOLDER not in new_inst:
            new_inst = f"{new_inst}\n\n{QUESTION_PLACEHOLDER}"
        return GsmTaskPrompt(new_inst)
