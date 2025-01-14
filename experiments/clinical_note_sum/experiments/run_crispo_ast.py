# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd

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
from experiments.clinical_note_sum.dataset import (
    load_clinical_note_sum_standard_train_dev_test,
)
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder
from crispo.llms.bedrock.claude2 import ClaudeInstant

_P = """
Please carefully review the attached clinical case. You will generate a clinical note summary precisely matching the structured SOAP format and completeness demonstrated in Examples A, B and C below.


The summary should contain sections for <History> (150-250 words with 6-10 sentences), <Subjective> (150-250 words with 6-10 sentences), <Objective> (250
-500 words including verbatim exam quotes), <Assessment> (150-250 words with 6-10 sentences), <Plan> (150-250 words with 6-10 sentences).


INSERT_INPUT_HERE


<example_A>
REFERENCE SUMMARY A
</example_A>


<example_B>
REFERENCE SUMMARY B
</example_B>


<example_C>
REFERENCE SUMMARY C
</example_C>


Specifically focus on including verbatim quotes, comprehensive exam findings/results, counseling points and defined terms like "HTN". Compare each section of your initial summary directly to the examples, emphasizing inclusion of all highlighted details. Then have a physician directly review your summary, applying feedback to precisely match structure, style and completeness of the examples. Feel free to ask clarifying questions and revise your summary iteratively until directly replicating Examples A through C. Write your summary within <summary> tags
""".strip()

cdroot()
train, dev, test = load_clinical_note_sum_standard_train_dev_test()

# Trainer
save_dir = "data/experiments/clinical_note_sum/suffix_tuning_critique_all_dev_steps"
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
    f"Seed suffix score on test set ({len(test)} samples): `[red]{seed_score}[/red]`"
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

main_score["method"] = "Main Prompt"
seed_score["method"] = "Seed Suffix"
score["method"] = "Suffix Tuning"

df = pd.DataFrame([main_score, seed_score, score])
trainer.logger.info(df.to_markdown(index=False))
