# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Union, Any

import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge_score.rouge_scorer import RougeScorer

from crispo.metrics.floats import FloatDict
from crispo.metrics.metric import Metric, MetricDict


class SummMultiMetrics(MetricDict):

    def __init__(
        self,
        primary: str | None = None,
        use_rouge="all",
        use_n_sent=True,
        use_n_word=True,
        description_style="simple",
        **metrics: Metric,
    ):
        super().__init__(primary, **metrics)
        self.use_rouge = use_rouge
        self.use_n_sent = use_n_sent
        self.use_n_word = use_n_word
        self.description_style = description_style

        if use_rouge:
            self.rouge_scorer = RougeScorer(
                ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
            )

    def compute_rouge(self, pred: Union[str, Any], gold: Union[str, Any]) -> dict:
        pred = "\n".join(sent_tokenize(pred))
        gold = "\n".join(sent_tokenize(gold))
        scores = self.rouge_scorer.score(prediction=pred, target=gold)
        ret = {}
        for k, v in scores.items():
            ret[k + "f"] = v.fmeasure * 100
            ret[k + "p"] = v.precision * 100
            ret[k + "r"] = v.recall * 100
        return ret

    def score(
        self, pred: Union[str, Any], gold: Union[str, Any], x: Union[str, Any] = None
    ) -> FloatDict:
        result = {}

        if self.use_rouge:
            result.update(self.compute_rouge(pred, gold))

        if self.use_n_sent:
            result["n_sent"] = len(sent_tokenize(pred))
            result["n_sent_diff"] = len(sent_tokenize(pred)) - len(sent_tokenize(gold))

        if self.use_n_word:
            result["n_word"] = len(word_tokenize(pred))
            result["n_word_diff"] = len(word_tokenize(pred)) - len(word_tokenize(gold))

        return FloatDict(value=result[self.primary] if self.primary else None, **result)

    def get_description(self, result) -> str:
        if self.description_style == "simple":
            ret = ""
            if self.use_rouge == "all":
                ret += f"ROUGE-1 FMeasure: {result['rouge1f']:.1f}\n"
                ret += f"ROUGE-1 Recall: {result['rouge1r']:.1f}\n"
                ret += f"ROUGE-1 Precision: {result['rouge1p']:.1f}\n"
            elif self.use_rouge == "recall":
                ret += f"ROUGE-1 Recall: {result['rouge1r']:.1f}\n"

            if self.use_n_sent:
                ret += f"Number of sentences: {result['n_sent']}\n"
                ret += f"Number of sentences difference: {result['n_sent_diff']}\n"

            if self.use_n_word:
                ret += f"Number of words: {result['n_word']}\n"
                ret += f"Number of words difference: {result['n_word_diff']}\n"
            return ret.strip()
        elif self.description_style == "detail":
            ret = ""
            if self.use_rouge == "all":
                ret += (
                    "ROUGE-1 score measures the word-level alignment between generated summary and "
                    "ground truth summary.\n"
                )
                ret += (
                    f"ROUGE-1 Recall is {result['rouge1r']:.1f}. It means the generated summary covers "
                    f"{result['rouge1r']:.1f}% words in the ground truth summary.\n"
                )
                ret += (
                    f"ROUGE-1 Precision: {result['rouge1p']:.1f}. It means {result['rouge1p']:.1f}% of words "
                    "in the generated summary appears in the ground truth summary.\n"
                )
                ret += (
                    f"ROUGE-1 FMeasure: {result['rouge1f']:.1f}. The FMeasure is the harmonic average of precision"
                    "and recall. Our primary goal is to maximize FMeasure.\n"
                )
                ret += (
                    "Since {metric} is lower, trying to increase {metric} would be more helpful in "
                    "maximizing FMeasure.\n"
                ).format(
                    metric=(
                        "Precision"
                        if result["rouge1p"] < result["rouge1r"]
                        else "Recall"
                    )
                )

            if self.use_n_sent:
                ret += "num. sentence difference: {abs_diff}\n".format(
                    abs_diff=round(abs(result["n_sent_diff"]))
                )
                ret += f"On average, the generated summary has {result['n_sent']} sentences.\n"
                ret += (
                    "The generated summary is {abs_diff} sentences {direction} than the ground truth summary.\n"
                ).format(
                    abs_diff=abs(result["n_sent_diff"]),
                    direction="fewer" if result["n_sent_diff"] < 0 else "more",
                )

            if self.use_n_word:
                ret += "num. word difference: {abs_diff}\n".format(
                    abs_diff=round(abs(result["n_word_diff"]))
                )
                ret += (
                    f"On average, the generated summary has {result['n_word']} words.\n"
                )
                ret += (
                    "The generated summary is {abs_diff} words {direction} than the ground truth summary.\n"
                ).format(
                    abs_diff=abs(result["n_word_diff"]),
                    direction="fewer" if result["n_word_diff"] < 0 else "more",
                )
            return ret.strip()
        else:
            raise ValueError(
                "Invalid description style: {}".format(self.description_style)
            )

    def aggregate(self, scores: list[FloatDict]) -> FloatDict:
        mean = pd.DataFrame([x.scores for x in scores]).mean().to_dict()
        return FloatDict(value=mean[self.primary] if self.primary else None, **mean)


if __name__ == "__main__":
    metric_detail = SummMultiMetrics(description_style="detail")
    score = metric_detail.score("hello world!", "This is a python hello world.")
    print("=" * 20)
    print(score)
    print("=" * 20)
    print("description style: detail")
    print(metric_detail.get_description(score))

    metric_simple = SummMultiMetrics(description_style="simple")
    score = metric_simple.score("hello world!", "This is a python hello world.")
    print("=" * 20)
    print(score)
    print("=" * 20)
    print("description style: simple")
    print(metric_simple.get_description(score))

# Output
# ====================
# {'rouge1f': 50.0, 'rouge1p': 100.0, 'rouge1r': 33.33333333333333, 'n_sent': 1, 'n_sent_diff': 0, 'n_word': 3, 'n_word_diff': 4}
# ====================
# description style: detail
# ROUGE-1 score measures the word-level alignment between generated summary and ground truth summary.
# ROUGE-1 Recall is 33.3. It means the generated summary covers 33.3% words in the ground truth summary.
# ROUGE-1 Precision: 100.0. It means 100.0% of words in the generated summary appears in the ground truth summary.
# ROUGE-1 FMeasure: 50.0. The FMeasure is the harmonic average of precisionand recall. Our primary goal is to maximize FMeasure.
# Since Recall is lower, trying to increase Recall would be more helpful in maximizing FMeasure.
# On average, the generated summary has 1 sentences.
# The generated summary is 0 sentences more than the ground truth summary.
# On average, the generated summary has 3 words.
# The generated summary is 4 words more than the ground truth summary.
# ====================
# {'rouge1f': 50.0, 'rouge1p': 100.0, 'rouge1r': 33.33333333333333, 'n_sent': 1, 'n_sent_diff': 0, 'n_word': 3, 'n_word_diff': 4}
# ====================
# description style: simple
# ROUGE-1 FMeasure: 50.0
# ROUGE-1 Recall: 33.3
# ROUGE-1 Precision: 100.0
# Number of sentences: 1
# Number of sentences difference: 0
# Number of words: 3
# Number of words difference: 4
