# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from experiments import cdroot
from experiments.summarization.constants import ARTICLE_PLACEHOLDER
from experiments.summarization.crispo import (
    SummarizationCritiquePrompt,
)
from experiments.summarization.crispo import SummarizationCriSPOMetaPrompt
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder
from experiments.samsum.dataset import (
    load_samsum_shuffled_quick_train_dev_test,
)
from experiments.summarization.metrics.rouge1 import Rouge1Fmeasure
from experiments.summarization.metrics.summ_multi_metrics import SummMultiMetrics
from crispo.trainer.trainer import Trainer
from crispo.llms.bedrock.mistral import BedrockMistral

cdroot()
# Dataset
train, dev, test = load_samsum_shuffled_quick_train_dev_test()

# Trainer
save_dir = "data/experiments/samsum_mistral/critique"
trainer = Trainer(save_dir)
task_llm = BedrockMistral(temperature=0)
prompt_score_pairs, best_prompt_dev = trainer.fit(
    train,
    dev=dev,
    initial_task_prompts={
        SummarizationTaskPromptWithPlaceholder(
            f"Here is a document:\n\n{ARTICLE_PLACEHOLDER}\n\nWrite a summary within <summary> tags for it."
        )
    },
    meta_prompt=SummarizationCriSPOMetaPrompt(),
    critique_prompt=SummarizationCritiquePrompt(),
    meta_llm=BedrockMistral(temperature=1.0),
    task_llm=task_llm,
    metric=Rouge1Fmeasure(),
    num_search_steps=100,
    num_new_prompts_in_each_step=3,
    num_few_shot_examples_in_meta_prompt=2,
    few_shot_selection_criteria="random",
    dev_evaluation_per_n_steps=5,
)
if not best_prompt_dev:
    best_prompt_dev = list(prompt_score_pairs.keys())[0]

# best_prompt_dev = SummarizationTaskPromptUniversalTemplate('The text below contains a discussion expressing several key facts and events. Your concise 1-sentence summary should relate only the 2 most important pieces of information stated, without assumptions or extra context.Positive reference examples include:<summary>John requests IKEA forks from James.</summary><summary>The movie received mixed reviews focusing solely on its convoluted plot.</summary>Before writing your summary within <summary> tags, compare any drafts directly to these references. Striving for brevity, relate the core discussion using as few words as the examples, focusing only on the most salient facts expressed. Refer back to both the text and references iteratively, revising your prediction by removing additional words until it precisely matches the exemplary concision. The aim is to accurately reflect the key discussion using minimal words, as in the high-quality references provided.\n\nINSERT_INPUT_HERE')
overall_score, scores, predictions = trainer.evaluate(
    best_prompt_dev,
    dataset=test,
    task_llm=task_llm,
    metric=SummMultiMetrics(),
    desc=f"Evaluating final prompt on test",
    save_path=f"{save_dir}/test_best-prompt.csv",
)

trainer.logger.info(
    f"`{best_prompt_dev}` score on test set ({len(test)} articles): `[red]{overall_score:.4f}[/red]`"
)
print(overall_score)
