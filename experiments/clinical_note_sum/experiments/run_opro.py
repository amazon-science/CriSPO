# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments.summarization.metrics.rouge1 import Rouge1Fmeasure
from experiments.summarization.metrics.summ_multi_metrics import SummMultiMetrics
from crispo.trainer.trainer import Trainer
from experiments import cdroot

# noinspection PyUnresolvedReferences
from experiments.clinical_note_sum.dataset import (
    load_clinical_note_sum_standard_train_dev_test,
)
from experiments.summarization.meta_prompt import SummarizationOproMetaPrompt
from experiments.summarization.task_prompt import SummarizationTaskPromptNoPlaceholder
from crispo.llms.bedrock.claude2 import ClaudeInstant

cdroot()

train, dev, test = load_clinical_note_sum_standard_train_dev_test()
data_name = "clinical_note_sum"

# Trainer
save_dir = f"data/experiments/{data_name}/shuffled_quick_opro"
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
    meta_llm=ClaudeInstant(temperature=1.0),
    task_llm=ClaudeInstant(temperature=0.0),
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
    task_llm=ClaudeInstant(temperature=0.0),
    metric=SummMultiMetrics(primary="rouge1f"),
    desc=f"Evaluating final prompt on test",
    save_path=f"{save_dir}/test_best-prompt.csv",
)
