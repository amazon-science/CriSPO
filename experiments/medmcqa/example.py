# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

from experiments.gsm.example import Output
from crispo.task.example import Example


@dataclass
class MedMcqaExample(Example):
    x: str
    y: Output
