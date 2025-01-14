# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Optional, List, Tuple

from experiments.medmcqa.example import MedMcqaExample
from crispo.optimizer.opro_meta_prompt import OproMetaPrompt
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag
from experiments.medmcqa.task_prompt import QUESTION_PLACEHOLDER, MedMcqaTaskPrompt

_P = """
Your task is to generate the instruction <instruction>.

Below are some examples:
{examples}

Below are some previous instructions with their scores. The score ranges from 0 to 100.
{instructions}

Generate an instruction that is different from all the instructions above, and has a higher score than all the instructions above.
It should be concise, effective, and generally applicable to all examples above.

Write your final new instruction in <instruction> tags.
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
</rated_instruction>
""".strip()


@dataclass
class MedMcqaOproMetaPrompt(OproMetaPrompt):
    prompt: str = _P

    def fill(
        self,
        prompt_score_pairs: List[Tuple[TaskPrompt, float]],
        few_shot_examples: List[MedMcqaExample],
        **kwargs,
    ) -> str:
        examples = []
        if few_shot_examples:
            for i, example in enumerate(few_shot_examples):
                examples.append(_E.format(input=example.x, output=example.y.label))

        instructions = []
        for prompt, score in prompt_score_pairs:
            instructions.append(
                _I.format(
                    instruction=prompt,
                    score=score if isinstance(score, str) else f"{score:.2%}",
                )
            )

        meta_prompt_text = _P.format(
            examples="\n".join(examples),
            instructions="\n".join(instructions),
        )
        return meta_prompt_text

    def parse(self, generation: str) -> Optional[TaskPrompt]:
        new_inst = extract_xml_tag(generation, "instruction")
        if not new_inst:
            return None
        if QUESTION_PLACEHOLDER not in new_inst:
            new_inst = f"{new_inst}\n\n{QUESTION_PLACEHOLDER}"

        if "<answer>" not in new_inst:
            new_inst = (
                f"{new_inst}\n\nChose an option and write it in <answer> XML tags"
            )
        return MedMcqaTaskPrompt(new_inst)
