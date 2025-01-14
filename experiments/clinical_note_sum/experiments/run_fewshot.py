# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from tqdm import tqdm

from crispo.llms.bedrock.claude2 import ClaudeInstant
from experiments import cdroot
from experiments.clinical_note_sum.dataset import (
    load_clinical_note_sum_standard_train_dev_test,
)
from experiments.common.example_selector import ExampleSelector
from experiments.summarization.constants import (
    ARTICLE_PLACEHOLDER,
    EXAMPLES_PLACEHOLDER,
)
from experiments.summarization.metrics.rouge1 import Rouge1Fmeasure
from experiments.summarization.task_prompt import SummarizationTaskPromptWithPlaceholder

cdroot()
# Dataset
train, dev, test = load_clinical_note_sum_standard_train_dev_test()
metric = Rouge1Fmeasure()
task_llm = ClaudeInstant(temperature=0)

# Replace it with the prompt (manual, opro/crispo tuned) to be tested
_P = f"""
Please carefully review the attached clinical case. You will generate a clinical note summary precisely matching the structured SOAP format and completeness demonstrated in Examples A, B and C below.

<examples>
{EXAMPLES_PLACEHOLDER}
</examples>

{ARTICLE_PLACEHOLDER}

The summary should contain sections for <History> (150-250 words with 6-10 sentences), <Subjective> (150-250 words with 6-10 sentences), <Objective> (250
-500 words including verbatim exam quotes), <Assessment> (150-250 words with 6-10 sentences), <Plan> (150-250 words with 6-10 sentences).

Specifically focus on including verbatim quotes, comprehensive exam findings/results, counseling points and defined terms like "HTN". Compare each section of your initial summary directly to the examples, emphasizing inclusion of all highlighted details. Then have a physician directly review your summary, applying feedback to precisely match structure, style and completeness of the examples. Feel free to ask clarifying questions and revise your summary iteratively until directly replicating Examples A through C. Write your summary within <summary> tags
""".strip()

selector = ExampleSelector([(example.x, example.y) for example in train])

k = 3
prompts = []
for example in tqdm(test, desc="Selecting examples"):
    examples = selector.select_k_example(k, example.x)
    examples = "\n".join(
        [
            f"<example_input>\n{x}\n</example_input>\n\n<example_summary>\n{y}\n</example_summary>"
            for x, y in examples
        ]
    )
    task_prompt = SummarizationTaskPromptWithPlaceholder(
        _P.replace(EXAMPLES_PLACEHOLDER, examples)
    )
    prompts.append(task_prompt.fill(example.x))

predictions = [
    SummarizationTaskPromptWithPlaceholder("").parse(y)
    for y in task_llm.batch_generate(prompts)
]

scores = [metric.score(p, example.y) for p, example in zip(predictions, test)]
score = metric.aggregate(scores)

print(f"{k}-shot score: {score:.4f}")
