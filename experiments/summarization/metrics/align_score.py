# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from abc import ABC
from typing import Union, Any

import transformers
from alignscore import AlignScore

from crispo.metrics.metric import Metric


class AlignScoreScorer(Metric, ABC):
    def __init__(self, large=True) -> None:
        super().__init__()
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda:0"
        except:
            pass
        model_size = "large" if large else "base"
        self.scorer = AlignScore(
            model=f"roberta-{model_size}",
            batch_size=32,
            device=device,
            ckpt_path=transformers.utils.get_file_from_repo(
                "yzha/AlignScore", f"AlignScore-{model_size}.ckpt"
            ),
            evaluation_mode="nli_sp",
            verbose=False,
        )


class AlignScorePrecision(AlignScoreScorer):
    def score(
        self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None
    ) -> Union[float, dict]:
        return self.scorer.score(contexts=[gold], claims=[pred])[0]


class AlignScorePrecisionX(AlignScoreScorer):
    def score(
        self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None
    ) -> Union[float, dict]:
        return self.scorer.score(contexts=[x], claims=[pred])[0]


class AlignScoreRecall(AlignScoreScorer):
    def score(
        self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None
    ) -> Union[float, dict]:
        return self.scorer.score(contexts=[pred], claims=[gold])[0]


class AlignScoreF1(AlignScoreScorer):
    # noinspection PyTypeChecker
    def score(
        self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None
    ) -> Union[float, dict]:
        p = AlignScorePrecision.score(self, pred, gold)
        r = AlignScoreRecall.score(self, pred, gold)
        return 2 * p * r / (p + r)


def main():
    precision = AlignScorePrecision()
    p = precision.score(
        pred="Over the last century, global temperatures have increased by approximately one degree Celsius.",
        gold="Earth has warmed one degree in past 100 years."
        "Greenhouse gases are causing temperatures to rise."
        "Planets often in periods of warming or cooling.",
    )
    print(f"Precision: {p}")
    recall = AlignScoreRecall()
    r = recall.score(
        pred="Over the last century, global temperatures have increased by approximately one degree Celsius.",
        gold="Earth has warmed one degree in past 100 years."
        "Greenhouse gases are causing temperatures to rise."
        "Planets often in periods of warming or cooling.",
    )
    print(f"Recall: {r}")
    f1 = AlignScoreF1()
    f = f1.score(
        pred="Over the last century, global temperatures have increased by approximately one degree Celsius.",
        gold="Earth has warmed one degree in past 100 years."
        "Greenhouse gases are causing temperatures to rise."
        "Planets often in periods of warming or cooling.",
    )
    print(f"F1: {f}")


if __name__ == "__main__":
    main()
