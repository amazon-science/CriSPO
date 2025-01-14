# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import statistics


class FloatList(float):

    def __new__(cls, *scores: float):
        return float.__new__(cls, statistics.mean(scores))

    def __init__(self, *scores: float):
        self.scores = scores


class FloatDict(float):
    def __new__(cls, value: float = None, **scores: float):
        return float.__new__(
            cls, statistics.mean(scores.values()) if value is None else value
        )

    def __init__(self, value: float = None, **scores: float):
        self.scores = scores

    def __getitem__(self, key):
        return self.scores[key]
