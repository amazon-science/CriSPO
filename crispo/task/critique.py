# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import abc
from typing import Union, Any, List
from crispo.task.example import Example


class CritiquePrompt(abc.ABC):
    @abc.abstractmethod
    def fill(
        self,
        prompt,
        predictions: List[Union[str, Any]],
        few_shot_examples: List[Example],
    ) -> str:
        pass

    @abc.abstractmethod
    def parse(self, generation: str):
        pass
