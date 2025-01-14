# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments import cdroot
from experiments.nq.meta_prompt import RAGMetaPrompt
from experiments.squad.dataset import load_squad_shuffled_standard_train_dev_test
from experiments.nq.task_prompt import RAGTaskPrompt
from experiments.summarization.metrics.ragqa import SquadMetric
from crispo.trainer.trainer import Trainer
from crispo.llms.bedrock.claude2 import ClaudeInstant
from crispo.llms import ClaudeSonnet

cdroot()
# Dataset
import sys

model = sys.argv[1]

if model == "Claude3":

    meta_llm = ClaudeSonnet(temperature=1.0)
    task_llm = ClaudeSonnet(temperature=0)
elif model == "Claude":
    meta_llm = ClaudeInstant(temperature=1.0)
    task_llm = ClaudeInstant(temperature=0)
train, dev, test = load_squad_shuffled_standard_train_dev_test()

dataname = "squad"

# Trainer
save_dir = f"data/experiments/{dataname}/{model}_opro"
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
    metric=SquadMetric(),
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
    metric=SquadMetric(),
    primary_metric_name="f1",
    desc=f"Evaluating final prompt on test",
    save_path=f"{save_dir}/test_best-prompt.csv",
)
