# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments import cdroot
from experiments.summarization.constants import ARTICLE_PLACEHOLDER
from experiments.summarization.crispo import SummarizationCritiquePrompt
from experiments.summarization.crispo import SummarizationCriSPOMetaPrompt
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder

# noinspection PyUnresolvedReferences
from experiments.meeting_bank.dataset import (
    load_meetingbank_shuffled_quick_train_dev_test,
    load_meetingbank_full_train_dev_test,
    load_meetingbank_debug_train_dev_test,
)
from experiments.summarization.metrics.rouge1 import Rouge1Fmeasure
from experiments.summarization.metrics.summ_multi_metrics import SummMultiMetrics
from crispo.trainer.trainer import Trainer
from experiments.summarization.utilities.dataset_util import truncated_dataset_loader
from crispo.llms import ClaudeSonnet
from crispo.llms import BedrockLlama

cdroot()

train, dev, test = truncated_dataset_loader(
    load_meetingbank_shuffled_quick_train_dev_test, max_input=2250, max_output=-1
)


# Trainer
save_dir = "data/experiments/meetingbank_llama/critiquey"
trainer = Trainer(save_dir=save_dir)
meta_llm = ClaudeSonnet(temperature=1.0)
task_llm = BedrockLlama(temperature=0.0)
metric = Rouge1Fmeasure()
prompt_score_pairs, best_prompt_dev = trainer.fit(
    train,
    dev=dev,
    initial_task_prompts={
        SummarizationTaskPromptWithPlaceholder(
            f"Here is a conversation:\n\n{ARTICLE_PLACEHOLDER}\n\nWrite a summary within <summary> tags for it."
        )
    },
    meta_llm=meta_llm,
    task_llm=task_llm,
    meta_prompt=SummarizationCriSPOMetaPrompt(),
    critique_prompt=SummarizationCritiquePrompt(),
    metric=metric,
    num_search_steps=100,
    num_new_prompts_in_each_step=3,
    num_few_shot_examples_in_meta_prompt=2,
    few_shot_selection_criteria="random",
    num_examples_in_critique=10,
)

trainer.evaluate(
    best_prompt_dev,
    test,
    task_llm=task_llm,
    metric=SummMultiMetrics(),
    desc=f"Evaluating final prompt on test",
    save_path=f"{save_dir}/test_best-prompt.csv",
)
