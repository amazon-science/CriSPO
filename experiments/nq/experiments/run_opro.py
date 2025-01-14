# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments import cdroot
from experiments.nq.dataset import load_natural_questions_quick_train_dev_test
from experiments.nq.meta_prompt import RAGMetaPrompt
from experiments.nq.task_prompt import RAGTaskPrompt
from crispo.llms.bedrock.claude2 import ClaudeInstant
from experiments.summarization.metrics.ragqa import ExactMatch
from crispo.trainer.trainer import Trainer

cdroot()
# Dataset

train, dev, test = load_natural_questions_quick_train_dev_test()

# Trainer
save_dir = "data/experiments/nq/opro"
trainer = Trainer(save_dir)

prompt_score_pairs, best_prompt_dev = trainer.fit(
    train,
    dev=dev,
    initial_task_prompts={
        RAGTaskPrompt(
            "Answer the question using the context provided. Write your answer in <answer> tags."
        )
    },
    meta_prompt=RAGMetaPrompt(),
    meta_llm=ClaudeInstant(temperature=1.0),
    task_llm=ClaudeInstant(temperature=0.0),
    metric=ExactMatch(),
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
    metric=ExactMatch(),
    desc=f"Evaluating final prompt on test",
    save_path=f"{save_dir}/test_best-prompt.csv",
)
