# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Optional


from experiments.gsm.meta_prompt_critique import GsmCritiqueMetaPrompt
from crispo.task.prompt import TaskPrompt
from experiments.medmcqa.meta_prompt_opro import MedMcqaOproMetaPrompt


@dataclass
class MedMcqaCritiqueMetaPrompt(GsmCritiqueMetaPrompt):
    task = "medical entrance exam multiple-choice questions"

    def parse(self, generation: str) -> Optional[TaskPrompt]:
        # noinspection PyTypeChecker
        return MedMcqaOproMetaPrompt.parse(self, generation)
