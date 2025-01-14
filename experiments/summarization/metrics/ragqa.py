# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Union, Any, List
import pandas as pd

from crispo.metrics.floats import FloatDict
from crispo.metrics.metric import Metric


class ExactMatch(Metric):
    def __init__(self):
        super().__init__()

    def score(self, pred: str, gold: List[str], x: Union[str, Any] = None) -> float:
        pred = pred.strip()
        for item in gold:
            if item.lower() == pred.lower():
                return 1.0

        return 0.0


from evaluate import load

squad_metric = load("squad")


class SquadMetric(Metric):

    def __init__(self):
        super().__init__()

    def score(self, pred: str, gold: dict, x: Union[str, Any] = None) -> FloatDict:
        predictions = [{"prediction_text": pred, "id": "56e10a3be3433e1400422b22"}]
        references = [{"answers": gold, "id": "56e10a3be3433e1400422b22"}]
        results = squad_metric.compute(predictions=predictions, references=references)
        return FloatDict(**results, value=results["f1"])

    def aggregate(self, scores: List[FloatDict]) -> FloatDict:
        results = pd.DataFrame([x.scores for x in scores]).mean().to_dict()
        return FloatDict(**results, value=results["f1"])
