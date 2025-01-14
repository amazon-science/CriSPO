# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments.summarization.metrics.align_score import AlignScorePrecisionX
from crispo.metrics.metric import MetricDict
from experiments.summarization.metrics.rouge1 import Rouge1Fmeasure
from experiments.summarization.suffix.critique.critique_prompt import (
    SuffixCritiquePrompt,
)
from experiments.summarization.suffix.critique.crispo_meta_prompt import (
    SuffixCritiqueMetaPrompt,
)
from experiments.summarization.suffix.suffix_prompt import SummarizationTaskPromptSuffix
from crispo.trainer.trainer import Trainer
from experiments import cdroot
from experiments.cnn.dataset import load_cnn_standard_train_dev_test_shuffled
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder
from crispo.llms.bedrock.claude2 import ClaudeInstant

_P = """
Below are two 100-word summary examples of the upcoming input text. Write your own 100-word summary within <summary> tags, focusing only on the thr
ee most important details, people or locations mentioned. Directly reflect the style and main topics of the examples provided without extra context:


<example>
- Organizers hope to use social media to inspire "Occupy Wall Street" protest on Saturday in New York's financial district
- Adbusters co-founder wants to emulate uprisings in Egypt, Iran by drawing thousands to protest financial fraud and lack of justice
</example>


<example>
- Hacktivist group Anonymous urged supporters to participate in planned sit-in against financial fraud in New York City
- Protest aims to emulate uprisings in Egypt and Iran by gathering thousands to call for justice and oppose Wall Street corruption
</example>


INSERT_INPUT_HERE
""".strip()

cdroot()
train, dev, test = load_cnn_standard_train_dev_test_shuffled()

# Trainer
save_dir = "data/experiments/cnn/cnn_suffix_tuning_critique_all_dev_steps"
trainer = Trainer(save_dir)
meta_prompt = SuffixCritiqueMetaPrompt(main_prompt=_P)
task_llm = ClaudeInstant(temperature=0.0)
metric = MetricDict(
    rouge1=Rouge1Fmeasure(), faithfulness=AlignScorePrecisionX(), primary="rank"
)
initial_prompt = SummarizationTaskPromptSuffix(
    main_prompt=meta_prompt.main_prompt,
    prompt="Every word of your summary must be faithful to the input text.",
)
main_score, _, _ = trainer.evaluate(
    SummarizationTaskPromptWithPlaceholder(initial_prompt.main_prompt),
    test,
    task_llm=task_llm,
    metric=metric,
    desc=f"Evaluating main prompt on test",
    save_path=f"{save_dir}/initial-prompt.csv",
)
trainer.logger.info(
    f"Main prompt score on test set ({len(test)} samples): `[red]{main_score}[/red]`"
)

seed_score, _, _ = trainer.evaluate(
    initial_prompt,
    test,
    task_llm=task_llm,
    metric=metric,
    desc=f"Evaluating seed prompt on test",
    save_path=f"{save_dir}/initial-prompt.csv",
)
trainer.logger.info(
    f"Seed suffix score on test set ({len(test)} samples): `[red]{main_score}[/red]`"
)

prompt_score_pairs, best_prompt_dev = trainer.fit(
    train,
    dev=dev,
    initial_task_prompts={initial_prompt},
    meta_prompt=meta_prompt,
    critique_prompt=SuffixCritiquePrompt(main_prompt=meta_prompt.main_prompt),
    meta_llm=ClaudeInstant(temperature=1.0),
    task_llm=task_llm,
    metric=metric,
    num_search_steps=100,
    num_new_prompts_in_each_step=3,
    num_few_shot_examples_in_meta_prompt=2,
    few_shot_selection_criteria="random",
    dev_evaluation_per_n_steps=1,
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
    f"Final score on test set ({len(test)} samples): `[red]{score}[/red]`"
)
