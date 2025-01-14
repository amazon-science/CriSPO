# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments import cdroot

# noinspection PyUnresolvedReferences
from experiments.meeting_bank.dataset import (
    load_meetingbank_shuffled_quick_train_dev_test,
    load_meetingbank_full_train_dev_test,
    load_meetingbank_debug_train_dev_test,
)
from experiments.nq.constants import CONTEXT_PLACEHOLDER, QUESTION_PLACEHOLDER
from experiments.nq.critique.critique_prompt import RAGCritiquePrompt
from experiments.nq.critique.meta_prompt import RAGMetaPrompt
from experiments.nq.dataset import load_natural_questions_quick_train_dev_test
from experiments.nq.task_prompt import RAGTaskPromptUniversalTemplate
from crispo.llms.bedrock.claude2 import ClaudeInstant
from experiments.summarization.metrics.ragqa import ExactMatch
from crispo.trainer.trainer import Trainer

cdroot()
# Dataset

train, dev, test = load_natural_questions_quick_train_dev_test()
# train, dev, test = load_natural_questions_debug_train_dev_test()
# Trainer
save_dir = "data/experiments/nq/critique"
trainer = Trainer(save_dir)

# Trainer

meta_llm = ClaudeInstant(temperature=1.0)
task_llm = ClaudeInstant(temperature=0.0)
metric = ExactMatch()

prompt_str = f"""Answer the question using the context provided.
{QUESTION_PLACEHOLDER}

{CONTEXT_PLACEHOLDER}

write your answer in <answer> tags."""

prompt_score_pairs, best_prompt_dev = trainer.fit(
    train,
    dev=dev,
    initial_task_prompts={RAGTaskPromptUniversalTemplate(prompt_str)},
    meta_llm=meta_llm,
    task_llm=task_llm,
    meta_prompt=RAGMetaPrompt(),
    critique_prompt=RAGCritiquePrompt(),
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
    metric=metric,
    desc=f"Evaluating final prompt on test",
    save_path=f"{save_dir}/test_best-prompt.csv",
)
