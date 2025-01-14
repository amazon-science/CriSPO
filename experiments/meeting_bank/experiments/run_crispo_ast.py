# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from crispo.metrics.metric import MetricDict
from experiments import cdroot
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder
from experiments.meeting_bank.dataset import (
    load_meetingbank_shuffled_quick_train_dev_test,
)
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
from crispo.llms.bedrock.claude2 import ClaudeInstant

_P = """
The INSERT_INPUT_HERE text discusses key discussions and decisions made during an important committee or board meeting addressing several proposals related to environmental or transportation regulations. Your task is to write a clear 2-3 sentence summary within <summary> tags precisely conveying the TWO most important resolutions reached, focusing on verbatim extraction of pertinent details from the text.


To guide your summary, consider the following reference examples demonstrating capture of core resolutions at different levels of technical terminology:


"At their regular July session held at City Hall, members of the Transportation Committee voted to approve precise Bill 118421 establishing new $5,000 annual monitoring requirements with the condition that quarterly compliance reports must be submitted."


"On June 15th, the Environmental Review Board convened at the Town Hall to consider proposed regulations SB9 and SB10's impacts on air quality standards. After over two hours of discussion on effects to strengthen emissions limits, they decided to delay enforcement until mid-August to further review public comments."


Your summary should mention the specific date and location of the meeting. It must directly quote or closely paraphrase up to two key resolutions and verbatim extract any critical names, numbers, bills or technical terms stated to precisely convey the core decisions discussed. Refer closely to these examples focusing on verbatim extraction of important specifics from the text. Write your prediction within <summary> tags.
""".strip()

cdroot()
train, dev, test = load_meetingbank_shuffled_quick_train_dev_test()

# Trainer
save_dir = "data/experiments/meeting_bank/meeting_bank_suffix_tuning_critique_short2"
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
