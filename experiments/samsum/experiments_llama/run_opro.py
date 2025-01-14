# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments.summarization.metrics.rouge1 import Rouge1Fmeasure
from experiments.summarization.metrics.summ_multi_metrics import SummMultiMetrics
from crispo.trainer.trainer import Trainer
from experiments import cdroot
from experiments.summarization.meta_prompt import CnnMetaPromptForMistral
from experiments.summarization.task_prompt import SummarizationTaskPromptNoPlaceholder
from experiments.samsum.dataset import load_samsum_shuffled_quick_train_dev_test
from crispo.llms import BedrockLlama

cdroot()

train, dev, test = load_samsum_shuffled_quick_train_dev_test()

# Trainer
save_dir = "data/experiments/samsum_llama/opro3"
trainer = Trainer(save_dir)
prompt_score_pairs, best_prompt_dev = trainer.fit(
    train,
    dev=dev,
    initial_task_prompts={
        SummarizationTaskPromptNoPlaceholder(
            "Write a summary within <summary> tags for the following text:"
        )
    },
    meta_prompt=CnnMetaPromptForMistral(),
    meta_llm=BedrockLlama(temperature=1.0),
    task_llm=BedrockLlama(temperature=0.0),
    metric=Rouge1Fmeasure(),
    num_search_steps=100,
    num_new_prompts_in_each_step=3,
    num_few_shot_examples_in_meta_prompt=2,
    num_task_prompts_in_meta_prompt=1,
    few_shot_selection_criteria="random",
    dev_evaluation_per_n_steps=5,
)

trainer.evaluate(
    best_prompt_dev,
    test,
    task_llm=BedrockLlama(temperature=0.0),
    metric=SummMultiMetrics(),
    desc=f"Evaluating final prompt on test",
    save_path=f"{save_dir}/test_best-prompt.csv",
)
