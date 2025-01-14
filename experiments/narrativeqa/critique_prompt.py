# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List

from experiments.narrativeqa.example import NarrativeQaExample
from crispo.task.critique import CritiquePrompt
from crispo.utilities.prompt_util import extract_xml_tag

_P = """
For a reading comprehension task, a reader answers a question by integrating information and reasoning across a context document following the instruction below:

<instruction>{instruction}</instruction>
<examples>
{examples}
</examples>

Write a general and helpful critique in <critique> XML tags to improve the instruction such that the predicted answers are as close to the reference answers as possible.

1. Come up with several dimensions to compare its predicted answers and reference answers, e.g., number of words, style, precision, recall, etc.
2. List the difference between predicted answers and references on each dimension.
3. Identify specific phrases in the instruction that could have gotten these predicted answers different with references on each dimension.
4. Suggest specific action items that are general to all examples and helpful to improve the instruction.

""".strip()

_ERROR = """
<example>
<context>
{context}
</context>
<question>
{question}
</question>

<predicted_answer>
{predicted_answer}
</predicted_answer>

<reference_answer_1>
{reference_answer_1}
</reference_answer_1>
<reference_answer_2>
{reference_answer_2}
</reference_answer_2>
</example>
""".strip()


class NarrativeQaCritiquePrompt(CritiquePrompt):

    def fill(
        self,
        prompt,
        predictions: List[str],
        few_shot_examples: List[NarrativeQaExample],
    ) -> str:
        errors = []
        for wrong, right in zip(predictions, few_shot_examples):
            wrong: str = wrong
            right: NarrativeQaExample = right
            if wrong not in right.y:
                errors.append(
                    _ERROR.format(
                        context=right.x.context,
                        question=right.x.question,
                        predicted_answer=wrong,
                        reference_answer_1=right.y[0],
                        reference_answer_2=right.y[1],
                    )
                )
        prompt = _P.format(instruction=prompt, examples="\n\n".join(errors))
        return prompt

    def parse(self, generation: str):
        found = extract_xml_tag(generation, "critique")
        if found:
            return found
        return generation
