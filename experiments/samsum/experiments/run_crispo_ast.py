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
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder
from experiments.samsum.dataset import (
    load_samsum_shuffled_quick_train_dev_test,
)
from crispo.llms.bedrock.claude2 import ClaudeInstant

_P = """
The text below contains a discussion expressing several key facts and events. Your concise 1-sentence summary should relate only the 2 most important pieces of information stated, without assumptions or extra context.


Positive reference examples include:
<summary>John requests IKEA forks from James.</summary>
<summary>The movie received mixed reviews focusing solely on its convoluted plot.</summary>


Before writing your summary within <summary> tags, compare any drafts directly to these references. Striving for brevity, relate the core discussion using as few words as the examples, focusing only on the most salient facts expressed. Refer back to both the text and references iteratively, revising your prediction by removing additional words until it precisely matches the exemplary concision. The aim is to accurately reflect the key discussion using minimal words, as in the high-quality references provided.

INSERT_INPUT_HERE
""".strip()

cdroot()
train, dev, test = load_samsum_shuffled_quick_train_dev_test()

# Trainer
save_dir = "data/experiments/samsum/alignscore_suffix_tuning_critique_no_ref"
trainer = Trainer(save_dir)
meta_prompt = SuffixCritiqueMetaPrompt(main_prompt=_P)
task_llm = ClaudeInstant(temperature=0.0)
metric = MetricDict(
    rouge1=Rouge1Fmeasure(), faithfulness=AlignScorePrecisionX(), primary="rank"
)
initial_prompt = SummarizationTaskPromptSuffix(
    main_prompt=meta_prompt.main_prompt,
    prompt="Every word of your summary must be faithful to the conversation.",
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
