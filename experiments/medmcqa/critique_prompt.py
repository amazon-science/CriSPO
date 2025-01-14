# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List

from experiments.gsm.example import GsmExample, Output
from crispo.task.critique import CritiquePrompt
from crispo.utilities.prompt_util import extract_xml_tag

_P = """
For a medical entrance exam multiple-choice question, a student is expected to to select the correct answer following an instruction.

<instruction>{instruction}</instruction>
<examples>
{examples}
</examples>

Write a general and helpful critique in <critique> XML tags to improve the instruction such that the solved answers are as accurate as possible.

1. Come up with several dimensions to compare the wrong reasoning traits and the correct reasoning, e.g., number of reasoning steps, logic, strategy, flow, etc.
2. List the difference of the wrong reasoning and the correct reasoning on each dimension.
3. Identify specific phrases in the instruction that could have gotten these reasoning different with references on each dimension.
4. Suggest specific action items that are general to all examples and helpful to improve the instruction.

""".strip()

_ERROR = """
<question>
{question}
</question>
<wrong_reasoning>
{wrong_reasoning}
</wrong_reasoning>
<correct_reasoning>
{correct_reasoning}
</correct_reasoning>
<wrong_answer>
{wrong_answer}
</wrong_answer>
<correct_answer>
{correct_answer}
</correct_answer>
""".strip()


class MedMcqaCritiquePrompt(CritiquePrompt):

    def fill(
        self, prompt, predictions: List[Output], few_shot_examples: List[GsmExample]
    ) -> str:
        errors = []
        for wrong, right in zip(predictions, few_shot_examples):
            wrong: Output = wrong
            right: GsmExample = right
            if wrong.label != right.y.label:
                errors.append(
                    _ERROR.format(
                        question=right.x,
                        wrong_reasoning=wrong.reasoning,
                        correct_reasoning=right.y.reasoning,
                        wrong_answer=wrong.label,
                        correct_answer=right.y.label,
                    )
                )
        prompt = _P.format(instruction=prompt, examples="\n\n".join(errors))
        return prompt

    def parse(self, generation: str):
        found = extract_xml_tag(generation, "critique")
        if found:
            return found
        return generation
