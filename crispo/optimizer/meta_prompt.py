# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import abc
from typing import List, Tuple

from crispo.task.example import Example
from crispo.task.prompt import TaskPrompt


class MetaPrompt(abc.ABC):
    @abc.abstractmethod
    def fill(
        self,
        prompt_score_pairs: List[Tuple[TaskPrompt, float]],
        few_shot_examples: List[Example],
        **kwargs
    ) -> str:
        pass

    @abc.abstractmethod
    def parse(self, generation: str) -> TaskPrompt:
        pass
