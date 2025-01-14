# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments import cdroot

# noinspection PyUnresolvedReferences
from experiments.cnn.dataset import (
    load_cnn_quick_train_dev_test,
    load_cnn_standard_train_dev_test,
    load_cnn_debug_train_dev_test,
    load_cnn_standard_train_dev_test_shuffled,
)
from experiments.summarization.meta_prompt import SummarizationOproMetaPrompt
from experiments.summarization.metrics.rouge1 import Rouge1Fmeasure
from experiments.summarization.metrics.summ_multi_metrics import SummMultiMetrics
from crispo.trainer.trainer import Trainer
from experiments.summarization.utilities.dataset_util import truncated_dataset_loader
from experiments.summarization.task_prompt import SummarizationTaskPromptNoPlaceholder
from crispo.llms import ClaudeSonnet
from crispo.llms import BedrockLlama

cdroot()
# Dataset
train, dev, test = truncated_dataset_loader(
    load_cnn_standard_train_dev_test_shuffled, max_input=2250, max_output=-1
)

# Trainer
save_dir = "data/experiments/cnn_llama/oproy"
trainer = Trainer(save_dir)
prompt_score_pairs, best_prompt_dev = trainer.fit(
    train,
    dev=dev,
    initial_task_prompts={
        SummarizationTaskPromptNoPlaceholder(
            "Write a summary within <summary> tags for the following text:"
        )
    },
    meta_prompt=SummarizationOproMetaPrompt(),
    meta_llm=ClaudeSonnet(temperature=1.0),
    task_llm=BedrockLlama(temperature=0.0),
    metric=Rouge1Fmeasure(),
    num_search_steps=100,
    num_new_prompts_in_each_step=3,
    num_few_shot_examples_in_meta_prompt=2,
    few_shot_selection_criteria="random",
    dev_evaluation_per_n_steps=5,
)

trainer.evaluate(
    best_prompt_dev,
    test,
    task_llm=BedrockLlama(temperature=0.0),
    metric=SummMultiMetrics(primary="rouge1f"),
    desc=f"Evaluating final prompt on test",
    save_path=f"{save_dir}/test_best-prompt.csv",
)
