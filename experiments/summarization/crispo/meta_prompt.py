# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Optional, List, Tuple

from crispo.optimizer.crispo_meta_prompt import CriSPOMetaPrompt
from crispo.task.example import Example
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag
from experiments.summarization.constants import ARTICLE_PLACEHOLDER
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder


@dataclass
class SummarizationCriSPOMetaPrompt(CriSPOMetaPrompt):
    prompt: str = (
        f"""
Your task is to optimize the instruction for a summarization task, where a writer is given an input text to write its summary following your instruction.

Below are some examples:
{{examples}}

Below are some previous instructions with their scores and critiques.
{{instructions}}

Generate an instruction that is different from all the instructions above, and has a higher score than all the instructions above.
It should be concise, effective, and generally applicable to all examples above.

Draft your new instruction step by step:

1. Compare high-score instructions to low-score ones, identify what suggestions could have improved them. List them in <suggestion> tags.
2. Apply the suggestions and draft a new instruction aiming for a higher score.
3. Be creative and vary the wording, paraphrase, position of {ARTICLE_PLACEHOLDER}, phrase order, grammar, sentence order, which specific example summaries to give, etc.
4. Write your final new instruction in <instruction> tags.
""".strip()
    )

    def parse(self, generation: str) -> Optional[TaskPrompt]:
        new_inst = extract_xml_tag(generation, "instruction")
        if not new_inst:
            return None
        if ARTICLE_PLACEHOLDER not in new_inst:
            new_inst = f"Given the input:\n\n{ARTICLE_PLACEHOLDER}\n\n" + new_inst
        if "<summary>" not in new_inst:
            new_inst += " Write your summary within <summary> tags."
        return SummarizationTaskPromptWithPlaceholder(new_inst)


_E_MISTRAL = """
EXAMPLE {id}
INPUT:
{article}
TARGET_SUMMARY:
{summary}
""".strip()

_I_MISTRAL = """
INSTRUCTION:
{instruction}
SCORE:
{score}
CRITIQUE:
{critique}
""".strip()


class CnnMetaPromptForMistral(SummarizationCriSPOMetaPrompt):
    prompt: str = (
        f"""
Your task is to optimize the instruction for a summarization task, where a writer is given an input text to write its summary following your instruction.

Below are some examples:
{{examples}}

Below are some previous instructions with their scores and critiques.
{{instructions}}

Generate an instruction that is different from all the instructions above, and has a higher score than all the instructions above.
It should be concise, effective, and generally applicable to all examples above.

Draft your new instruction step by step:

1. Compare high-score instructions to low-score ones, identify what suggestions could have improved them. List them in <suggestion> tags.
2. Apply the suggestions and draft a new instruction aiming for a higher score.
3. Be creative and vary the wording, paraphrase, position of {ARTICLE_PLACEHOLDER}, phrase order, grammar, sentence order, which specific example summaries to give, etc.
4. Write your final new instruction in <instruction> tags. 
5. You must use {ARTICLE_PLACEHOLDER} only once in new instruction.  
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
                    _E_MISTRAL.format(id=(i + 1), article=example.x, summary=example.y)
                )

        instructions = []
        for (prompt, score), critique in zip(prompt_score_pairs, critiques):
            instructions.append(
                _I_MISTRAL.format(
                    instruction=prompt, score=f"{score:.2%}", critique=critique
                )
            )

        meta_prompt_text = self.prompt.format(
            examples="\n\n".join(examples),
            instructions="\n\n".join(instructions),
        )
        return meta_prompt_text
