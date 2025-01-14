# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

from crispo.task.example import Example


@dataclass
class Output:
    label: str
    reasoning: str


@dataclass
class GsmExample(Example):
    x: str
    y: Output
