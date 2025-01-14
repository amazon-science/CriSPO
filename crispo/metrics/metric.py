# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import abc
import statistics
from collections import defaultdict
from typing import Union, Any, Optional, List

from crispo.metrics.floats import FloatDict


class Metric(abc.ABC):

    @abc.abstractmethod
    def score(
        self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None
    ) -> float:
        pass

    def aggregate(self, scores):
        return statistics.mean(scores)

    def key(self, score: float):
        return score


class MetricDict(Metric):
    def __init__(self, primary: Optional[str] = None, **metrics: Metric):
        super().__init__()
        self.primary = primary
        self.metrics = metrics

    def score(
        self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None
    ) -> float:
        scores = dict(
            (name, metric.score(pred, gold, x)) for name, metric in self.metrics.items()
        )
        return FloatDict(**scores)

    def aggregate(self, scores: List[FloatDict]):
        scores_per_metric = defaultdict(list)
        for each in scores:
            for name, score in each.scores.items():
                scores_per_metric[name].append(score)
        aggregated_scores = dict(
            (name, metric.aggregate(scores_per_metric[name]))
            for name, metric in self.metrics.items()
        )
        return FloatDict(**aggregated_scores)

    def key(self, score: FloatDict):
        if self.primary:
            primary_score = score.scores[self.primary]
            if self.primary == "rank":
                primary_score = -primary_score
            return primary_score
        return score

    def __getitem__(self, item):
        return self.metrics[item]
