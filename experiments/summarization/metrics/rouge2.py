# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Union, Any
from rouge_score.rouge_scorer import RougeScorer
from crispo.metrics.metric import Metric


class Rouge2Fmeasure(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.scorer = RougeScorer(["rouge2"], use_stemmer=True)

    def score(
        self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None
    ) -> float:
        scores = self.scorer.score(gold, pred)
        return scores["rouge2"].fmeasure


def main():
    metric = Rouge2Fmeasure()
    score = metric.score("A B C D", "B C D")
    print(score)


if __name__ == "__main__":
    main()
