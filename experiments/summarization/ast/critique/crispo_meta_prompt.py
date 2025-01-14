# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# -*- coding:utf-8 -*-
from dataclasses import dataclass
from typing import Optional

from crispo.optimizer.ast.crispo_meta_prompt import CriSPOASTMetaPrompt
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag
from experiments.summarization.suffix.suffix_prompt import SummarizationTaskPromptSuffix


@dataclass
class SuffixCritiqueMetaPrompt(CriSPOASTMetaPrompt):
    prompt: str = (
        """
Your task is to append postscript instruction to the main instruction for a summarization task, where a writer is given an input text to write its summary following your instruction.

The main instruction is:

<instruction>{main_prompt}</instruction>

Below are some examples:
{examples}

Below are some previous postscripts with their scores and critiques.
{postscripts}

Generate a short postscript that is different from all the postscripts above, and has a higher score than all the postscripts above.
It should be concise, effective, and generally applicable to all examples above.

Draft your new postscript step by step:

1. Compare high-score postscripts to low-score ones, identify what suggestions could have improved them. List them in <suggestion> tags.
2. Apply the suggestions and draft a new postscript aiming for a higher score.
3. Be creative and vary the wording, paraphrase, phrase order, grammar, sentence order, etc.
4. Write your final new short postscript in <postscript> tags.
""".strip()
    )

    def parse(self, generation: str) -> Optional[TaskPrompt]:
        new_inst = extract_xml_tag(generation, "postscript")
        if not new_inst:
            return None
        return SummarizationTaskPromptSuffix(
            main_prompt=self.main_prompt, prompt=new_inst
        )
