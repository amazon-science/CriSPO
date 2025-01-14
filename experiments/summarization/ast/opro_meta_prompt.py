# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# -*- coding:utf-8 -*-
from dataclasses import dataclass
from typing import Optional, List, Tuple

from crispo.optimizer.ast.opro_meta_prompt import OproASTMetaPrompt
from crispo.task.example import Example
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag
from experiments.summarization.suffix.suffix_prompt import SummarizationTaskPromptSuffix


@dataclass
class SuffixMetaPrompt(OproASTMetaPrompt):
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
                    f"\n<example>\n<instruction>?</instruction>\n<document>\n{example.x}\n</document>\n"
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

    def parse(self, generation: str) -> Optional[TaskPrompt]:
        suffix = extract_xml_tag(generation, "postscript")
        if not suffix:
            return None
        return SummarizationTaskPromptSuffix(
            main_prompt=self.main_prompt, prompt=suffix
        )
