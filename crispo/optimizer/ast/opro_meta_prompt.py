# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import abc
from dataclasses import dataclass
from typing import List, Tuple

from crispo.optimizer.opro_meta_prompt import OproMetaPrompt
from crispo.task.example import Example
from crispo.task.prompt import TaskPrompt


@dataclass
class OproASTMetaPrompt(OproMetaPrompt, abc.ABC):
    prompt: str = (
        "Your task is to append postscript instruction to the main instruction for a summarization task."
    )
    main_prompt: str = ""

    def fill(
        self,
        prompt_score_pairs: List[Tuple[TaskPrompt, float]],
        few_shot_examples: List[Example],
        **kwargs,
    ) -> str:
        meta_prompt = [
            self.prompt,
            "\n",
            f"The main instruction is:\n\n<instruction>{self.main_prompt}</instruction>\n\n",
        ]

        if few_shot_examples:
            meta_prompt.append("Below are some examples:\n")
            for i, example in enumerate(few_shot_examples):
                meta_prompt.append(
                    f"\n<example>\n<instruction>?</instruction>\n<input>\n{example.x}\n</input>\n"
                    f"<output>\n{example.y}\n</output>\n</example>\n"
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
                f"\n<rating>\n<postscript>{prompt.prompt}</postscript>\n"
                f"<score>{score_text}</score>\n</rating>\n"
            )

        meta_prompt.append(
            "\n\nGenerate a short postscript instruction that"
            " is different from all the postscript instructions <postscript> above,"
            " and has a higher score than all the postscript instructions <postscript> above."
            " The postscript instruction should begin with <postscript> and end with </postscript>."
            " The postscript instruction should be concise, effective,"
            " and generally applicable to all examples above."
        )
        meta_prompt_text = "".join(meta_prompt)
        return meta_prompt_text
