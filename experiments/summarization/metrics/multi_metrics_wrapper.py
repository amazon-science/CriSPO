# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import statistics
from typing import Union, Any
import pandas as pd
from crispo.metrics.metric import Metric


class MultiMetricsWrapper(Metric):
    def __init__(self, compute_avg=True, **metrics: Metric) -> None:
        self.compute_avg = compute_avg
        self.metrics = metrics

    def score(
        self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None
    ) -> dict:
        scores = dict(
            (name, metric.score(pred, gold, x=x))
            for name, metric in self.metrics.items()
        )
        if self.compute_avg:
            scores["avg_score"] = statistics.mean(scores.values())
        return scores

    def get_description(self, result: dict) -> str:
        return "\n".join(f"{metric}: {score:.2f}" for metric, score in result.items())

    def aggregate(self, scores) -> dict:
        return pd.DataFrame(scores).mean().to_dict()
