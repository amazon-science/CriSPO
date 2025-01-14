# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments import cdroot
from experiments.summarization.constants import ARTICLE_PLACEHOLDER
from experiments.summarization.crispo import (
    SummarizationCritiquePrompt,
)
from experiments.summarization.crispo import SummarizationCriSPOMetaPrompt
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder
from experiments.meeting_bank.dataset import (
    load_meetingbank_shuffled_quick_train_dev_test,
)
from experiments.summarization.metrics.rouge1 import Rouge1Fmeasure
from experiments.summarization.metrics.summ_multi_metrics import SummMultiMetrics
from crispo.trainer.trainer import Trainer
from crispo.llms.bedrock.claude2 import ClaudeInstant

cdroot()
# Dataset
train, dev, test = load_meetingbank_shuffled_quick_train_dev_test()

# Trainer
save_dir = "data/experiments/meetingbank/quick_critique1"
trainer = Trainer(save_dir)
task_llm = ClaudeInstant(temperature=0)
prompt_score_pairs, best_prompt_dev = trainer.fit(
    train,
    dev=dev,
    initial_task_prompts={
        SummarizationTaskPromptWithPlaceholder(
            f"Here is a news article:\n\n{ARTICLE_PLACEHOLDER}\n\nWrite a summary within <summary> tags for it."
        )
    },
    meta_prompt=SummarizationCriSPOMetaPrompt(),
    critique_prompt=SummarizationCritiquePrompt(),
    meta_llm=ClaudeInstant(temperature=1.0),
    task_llm=task_llm,
    metric=Rouge1Fmeasure(),
    num_search_steps=100,
    num_new_prompts_in_each_step=3,
    num_few_shot_examples_in_meta_prompt=2,
    few_shot_selection_criteria="random",
    dev_evaluation_per_n_steps=5,
)
if not best_prompt_dev:
    best_prompt_dev = list(prompt_score_pairs.keys())[0]

score = trainer.evaluate(
    best_prompt_dev,
    test,
    task_llm=task_llm,
    metric=SummMultiMetrics(primary="rouge1f"),
    desc=f"Evaluating final prompt on test",
    save_path=f"{save_dir}/test_best-prompt.csv",
)[0]

trainer.logger.info(
    f"`{best_prompt_dev}` score on test set ({len(test)} articles): `[red]{score:.4f}[/red]`"
)
