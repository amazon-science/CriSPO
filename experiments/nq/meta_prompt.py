# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Optional, List, Tuple

from experiments.nq.example import encode_context
from experiments.nq.task_prompt import RAGTaskPrompt
from crispo.optimizer.opro_meta_prompt import OproMetaPrompt
from crispo.task.example import Example
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag


@dataclass
class RAGMetaPrompt(OproMetaPrompt):
    prompt: str = (
        "Your task is to generate the instruction <instruction> for a question answering task."
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
                    f"\n<example>\n<instruction>?</instruction>\n<question>\n{example.x[0]}\n</question>\n"
                    f"{encode_context(example.x, 20)}\n"
                    f"<answer>\n{example.get_y_str()}\n</answer>\n</example>\n"
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
        return RAGTaskPrompt(new_inst)
