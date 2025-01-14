# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List

from rouge_score.rouge_scorer import RougeScorer

from crispo.metrics.metric import Metric


class MaxRougeLFmeasure(Metric):

    def __init__(self) -> None:
        super().__init__()
        self.scorer = RougeScorer(["rougeL"], use_stemmer=True)

    def score(self, pred: str, gold: List[str], x: str = None) -> float:
        # See: https://github.com/shmsw25/qa-hard-em/issues/18#issuecomment-676813048
        return max([self.scorer.score(g, pred)["rougeL"].fmeasure for g in gold])


def main():
    metric = MaxRougeLFmeasure()
    score = metric.score("Home", ["At home", "His house"])
    print(score)


if __name__ == "__main__":
    main()
