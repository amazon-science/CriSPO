# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import abc
from dataclasses import dataclass
from typing import Any, Union


@dataclass(eq=True, frozen=True)
class TaskPrompt(abc.ABC):
    prompt: str

    @abc.abstractmethod
    def fill(self, x: Union[str, Any]) -> str:
        pass

    @abc.abstractmethod
    def parse(self, generation: str) -> Union[str, Any]:
        pass

    def __str__(self) -> str:
        return self.prompt

    def short_str(self, max_length=50):
        return str(self).replace("\n", " ")[:max_length]
