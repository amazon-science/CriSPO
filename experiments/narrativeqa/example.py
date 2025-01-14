# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import List

from crispo.task.example import Example


@dataclass
class Input:
    context: str
    question: str


@dataclass
class NarrativeQaExample(Example):
    x: Input
    y: List[str]
