# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from crispo.utilities.io_util import load_json, save_json
from experiments import cdroot
from experiments.medmcqa.dataset import load_medmcqa_standard_train_dev_test
from experiments.medmcqa.experiments.manual import (
    _input_of_example,
    _output_of_example,
    _score_of,
)
from experiments.medmcqa.task_prompt import QUESTION_PLACEHOLDER, MedMcqaTaskPrompt
from experiments.narrativeqa.experiments.instant.fewshot.multi_turn.reeval import (
    re_evaluate_dir,
)
from experiments.summarization.metrics.nested_accuracy import NestedAccuracy
from crispo.llms.bedrock.claude2 import ClaudeInstant
from crispo.llms import ClaudeSonnet

cdroot()
scores = []
metric = NestedAccuracy()
best_dev_prompt = f"""
{QUESTION_PLACEHOLDER}

Chose an option and write it in <answer> XML tags.
""".strip()

train, dev, test = load_medmcqa_standard_train_dev_test()
fewshot_candidates = train

for llm in [
    ClaudeInstant(temperature=0.0),
    ClaudeSonnet(temperature=0.0),
]:
    if isinstance(llm, ClaudeSonnet):
        concurrency = 4
    else:
        concurrency = 16
    for num_shot in [5]:
        save_dir = f"data/experiments/medmcqa/{llm.__class__.__name__}/{num_shot}-shot"
        result_path = f"{save_dir}/test-{num_shot}-shot.json"
        try:
            score = load_json(result_path + "0")
            print(f"{save_dir}: {score}")
        except FileNotFoundError:
            score = re_evaluate_dir(
                save_dir,
                num_shot=num_shot,
                metric=metric,
                prompt_cls=MedMcqaTaskPrompt,
                fewshot_candidates=fewshot_candidates,
                test=test,
                input_of_example=_input_of_example,
                output_of_example=_output_of_example,
                score_of=_score_of,
                best_dev_prompt=best_dev_prompt,
                concurrency=concurrency,
                # example_template_func=example_template_func,
                use_prompt_in_fewshots=False,
            )
            save_json(score, result_path)
