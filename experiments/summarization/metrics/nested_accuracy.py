# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from crispo.metrics.accuracy import Accuracy
from experiments.gsm.example import Output


class NestedAccuracy(Accuracy):
    def score(self, pred: Output, gold: Output, x: str = None) -> float:
        return super().score(pred.label, gold.label, x)
