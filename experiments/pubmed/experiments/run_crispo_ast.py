# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from crispo.metrics.metric import MetricDict
from experiments import cdroot
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder
from experiments.pubmed.dataset import load_pubmed_quick_train_dev_test
from crispo.llms.bedrock.claude2 import ClaudeInstant
from experiments.summarization.metrics.align_score import AlignScorePrecisionX
from experiments.summarization.metrics.rouge1 import Rouge1Fmeasure
from experiments.summarization.suffix.critique.critique_prompt import (
    SuffixCritiquePrompt,
)
from experiments.summarization.suffix.critique.crispo_meta_prompt import (
    SuffixCritiqueMetaPrompt,
)
from experiments.summarization.suffix.suffix_prompt import SummarizationTaskPromptSuffix
from crispo.trainer.trainer import Trainer

_P = """
Refer to the following concise reference example summaries:


<reference1>
- Study objectives were to determine receptor expression levels. Scintigraphy found 68% uptake indicating high somatostatin levels. 68Ga-DOTATATE helped with initial staging and follow-ups by assessing receptor status.
</reference1>


Within the <summary> tags below, write a 100-150 word bullet summary for the INSERT_INPUT_HERE text representing:


1) Study aims and population characteristics
2) Key methodology including sample size and techniques used
3) Important quantitative result and its implications


Match the style and technical terminology of the example reference summary. Your concise yet detailed summary should precisely capture the methodology, significant findings and conclusions. Feel free to directly quote important statistics and results.


INSERT_INPUT_HERE
""".strip()

cdroot()
train, dev, test = load_pubmed_quick_train_dev_test()

# Trainer
save_dir = "data/experiments/samsum/pubmed_suffix_tuning_critique_short"
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
score, _, _ = trainer.evaluate(
    SummarizationTaskPromptWithPlaceholder(initial_prompt.main_prompt),
    test,
    task_llm=task_llm,
    metric=metric,
    desc=f"Evaluating initial prompt on test",
    save_path=f"{save_dir}/initial-prompt.csv",
)
trainer.logger.info(
    f"Initial score on test set ({len(test)} samples): `[red]{score}[/red]`"
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
    dev_evaluation_per_n_steps=5,
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
