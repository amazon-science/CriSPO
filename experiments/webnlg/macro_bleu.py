# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from sacrebleu import BLEU

from crispo.metrics.metric import Metric


class MacroBLEU(Metric):

    def __init__(self):
        super().__init__()
        self._scorer = BLEU(effective_order=True)

    def score(self, pred: str, gold: list[str], x: str = None) -> float:
        return self._scorer.sentence_score(pred, gold).score


def main():
    metric = MacroBLEU()
    score = metric.score(
        "It wasn't surprising.", ["It was not unexpected.", "No one was surprised."]
    )
    print(score)


if __name__ == "__main__":
    main()
