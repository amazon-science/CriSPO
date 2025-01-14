# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Optional, List, Tuple

from crispo.optimizer.opro_meta_prompt import OproMetaPrompt
from crispo.task.example import Example
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag
from experiments.summarization.task_prompt import SummarizationTaskPromptNoPlaceholder


@dataclass
class SummarizationOproMetaPrompt(OproMetaPrompt):
    prompt: str = (
        "Your task is to generate the instruction <instruction> for a summarization task."
    )

    def fill(
        self,
        prompt_score_pairs: List[Tuple[TaskPrompt, float]],
        few_shot_examples: List[Example],
        **kwargs,
    ) -> str:
        meta_prompt = [self.prompt, "\n"]

        if few_shot_examples:
            meta_prompt.append("Below are some examples:\n")
            for i, example in enumerate(few_shot_examples):
                meta_prompt.append(
                    f"\n<example>\n<instruction>?</instruction>\n<input>\n{example.x}\n</input>\n"
                    f"<summary>\n{example.y}\n</summary>\n</example>\n"
                )

        meta_prompt.append(
            "Below are some previous instructions with their scores. The score ranges from 0 to 100.\n"
        )
        for prompt, score in prompt_score_pairs:
            if isinstance(score, float):
                score_text = round(score * 100)
            elif isinstance(score, int) or isinstance(score, str):
                score_text = score
            else:
                raise TypeError(f"Unsupported score type {type(score)}")

            meta_prompt.append(
                f"\n<rated_instruction>\n<instruction>{prompt.prompt}</instruction>\n"
                f"<score>{score_text}</score>\n</rated_instruction>\n"
            )

        meta_prompt.append(
            "\n\nGenerate an instruction that"
            " is different from all the instructions <instruction> above,"
            " and has a higher score than all the instructions <instruction> above."
            " The instruction should begin with <instruction> and end with </instruction>."
            " The instruction should be concise, effective,"
            " and generally applicable to all examples above."
        )
        meta_prompt_text = "".join(meta_prompt)
        return meta_prompt_text

    def parse(self, generation: str) -> Optional[TaskPrompt]:
        new_inst = extract_xml_tag(generation, "instruction")
        if not new_inst:
            return None
        return SummarizationTaskPromptNoPlaceholder(new_inst)


class CnnMetaPromptForMistral(OproMetaPrompt):
    prompt: str = "Your task is to generate the instruction for a summarization task."

    def fill(
        self,
        prompt_score_pairs: List[Tuple[TaskPrompt, float]],
        few_shot_examples: List[Example],
        **kwargs,
    ) -> str:
        meta_prompt = [self.prompt, "\n"]

        if few_shot_examples:
            meta_prompt.append("Below are some examples:\n")
            for i, example in enumerate(few_shot_examples):
                meta_prompt.append(
                    f"\nEXAMPLE {i + 1}\nINSTRUCTION:\nthe instruction you generated will be inserted here.\n"
                    f"INPUT:\n{example.x}\n"
                    f"TARGET SUMMARY:\n{example.y}\n"
                )

        meta_prompt.append(
            "\nBelow are some previous instructions with their scores. The score ranges from 0 to 100.\n"
        )
        for prompt, score in prompt_score_pairs:
            if isinstance(score, float):
                score_text = "%.2f" % (score * 100)
            elif isinstance(score, int) or isinstance(score, str):
                score_text = score
            else:
                raise TypeError(f"Unsupported score type {type(score)}")

            meta_prompt.append(
                f"\nINSTRUCTION:\n{prompt.prompt}\n" f"SCORE:\n{score_text}\n"
            )

        meta_prompt.append(
            "\n\nGenerate an instruction that"
            " is different from all the instructions above and has a higher score."
            " The instruction should begin with <instruction> and end with </instruction>."
            " The instruction should be concise, effective,"
            " and generally applicable to all examples above."
        )
        meta_prompt_text = "".join(meta_prompt)
        return meta_prompt_text

    def parse(self, generation: str) -> Optional[TaskPrompt]:
        new_inst = extract_xml_tag(generation, "instruction")
        if not new_inst:
            return None
        return SummarizationTaskPromptNoPlaceholder(new_inst)
