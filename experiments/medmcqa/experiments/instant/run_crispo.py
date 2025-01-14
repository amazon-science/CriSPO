# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import statistics

from experiments import cdroot
from experiments.medmcqa.critique_prompt import MedMcqaCritiquePrompt
from experiments.medmcqa.dataset import load_medmcqa_standard_train_dev_test
from experiments.summarization.metrics.nested_accuracy import NestedAccuracy
from crispo.trainer.trainer import Trainer
from experiments.medmcqa.meta_prompt_critique import MedMcqaCritiqueMetaPrompt
from experiments.medmcqa.task_prompt import MedMcqaTaskPrompt
from crispo.llms.bedrock.claude2 import ClaudeInstant

cdroot()
# Dataset
train, dev, test = load_medmcqa_standard_train_dev_test()

# Trainer
scores = []
for run in range(3):
    save_dir = f"data/experiments/medmcqa/claude_instant/critique-{run}"
    trainer = Trainer(save_dir=save_dir)
    task_llm = ClaudeInstant(temperature=0.0)
    metric = NestedAccuracy()
    prompt_score_pairs, best_prompt_dev = trainer.fit(
        train,
        dev=dev,
        initial_task_prompts={MedMcqaTaskPrompt()},
        meta_prompt=MedMcqaCritiqueMetaPrompt(),
        meta_llm=ClaudeInstant(temperature=1.0),
        critique_prompt=MedMcqaCritiquePrompt(),
        metric=metric,
        num_search_steps=100,
        num_new_prompts_in_each_step=3,
        few_shot_selection_criteria="random",
        dev_evaluation_per_n_steps=1,
        dev_evaluation_threshold=0.5,
        critique_example_selection_criteria="lowest_score",
        max_score_of_example_in_critique=1.0,
    )

    score, _, _ = trainer.evaluate(
        best_prompt_dev,
        test,
        task_llm=task_llm,
        metric=metric,
        desc=f"Evaluating final prompt on test",
        save_path=f"{save_dir}/test_best-prompt.csv",
    )
    trainer.logger.info(
        f"Best prompt:\n\n[light_magenta]{best_prompt_dev}[/light_magenta]\n\n"
        f"Final score on test set ({len(test)} samples): `[red]{score}[/red]`"
    )
    scores.append(score)

for run, score in enumerate(scores):
    print(f"Run-{run} score:\t{score}")
print("Scores in a row: " + "\t".join(str(score) for score in scores))
print(f"Average score:\t{statistics.mean(scores)}")
