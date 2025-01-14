# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments import cdroot
from experiments.clinical_note_sum.dataset import (
    load_clinical_note_sum_standard_train_dev_test,
)
from experiments.summarization.constants import ARTICLE_PLACEHOLDER
from experiments.summarization.crispo import SummarizationCritiquePromptForMistral
from experiments.summarization.crispo import CnnMetaPromptForMistral

# noinspection PyUnresolvedReferences
from experiments.cnn.dataset import (
    load_cnn_quick_train_dev_test,
    load_cnn_standard_train_dev_test,
    load_cnn_debug_train_dev_test,
    load_cnn_standard_train_dev_test_shuffled,
)
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder
from experiments.summarization.metrics.rouge1 import Rouge1Fmeasure
from experiments.summarization.metrics.summ_multi_metrics import SummMultiMetrics
from crispo.trainer.trainer import Trainer
from crispo.llms.bedrock.mistral import BedrockMistral

cdroot()
# Dataset
train, dev, test = load_clinical_note_sum_standard_train_dev_test()
# train, dev, test = load_meetingbank_full_train_dev_test()
# train, dev, test = load_meetingbank_debug_train_dev_test()
data_name = "clinical"

# Trainer
save_dir = "data/experiments/clinical_mistral/critique3-run3"
trainer = Trainer(save_dir=save_dir)
meta_llm = BedrockMistral(temperature=1.0)
task_llm = BedrockMistral(temperature=0.0)
metric = Rouge1Fmeasure()
prompt_score_pairs, best_prompt_dev = trainer.fit(
    train,
    dev=dev,
    initial_task_prompts={
        SummarizationTaskPromptWithPlaceholder(
            f"Here is a news article:\n\n{ARTICLE_PLACEHOLDER}\n\nWrite a summary within <summary> tags for it."
        )
    },
    meta_llm=meta_llm,
    task_llm=task_llm,
    meta_prompt=CnnMetaPromptForMistral(),
    critique_prompt=SummarizationCritiquePromptForMistral(),
    metric=metric,
    num_search_steps=100,
    num_new_prompts_in_each_step=3,
    num_few_shot_examples_in_meta_prompt=2,
    num_task_prompts_in_meta_prompt=1,
    few_shot_selection_criteria="random",
)

trainer.evaluate(
    best_prompt_dev,
    test,
    task_llm=task_llm,
    metric=SummMultiMetrics(primary="rouge1f"),
    desc=f"Evaluating final prompt on test",
    save_path=f"{save_dir}/test_best-prompt.csv",
)
