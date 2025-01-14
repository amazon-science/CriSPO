# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Union, Any, List

from crispo.task.critique import CritiquePrompt
from crispo.task.example import Example
from crispo.utilities.prompt_util import extract_xml_tag

_P = """
In a summarization task, a writer is given an input text to write a summary following an instruction.

<instruction>{instruction}</instruction>
<examples>
{examples}
</examples>

Write a general and helpful critique in <critique> XML tags to improve the instruction such that the predicted summaries are as close to references as possible.

1. Come up with several dimensions to compare its predicted summaries and reference summaries, e.g., number of words, number of sentences, style, precision, recall, etc.
2. List the difference predicted summaries and references on each dimension.
3. Identify specific phrases in the instruction that could have gotten these predicted summaries different with references on each dimension.
4. Suggest specific action items that are general to all examples and helpful to improve the instruction.

""".strip()

_E = """
<example>
<input>
{document}
</input>
<predicted_summary>
{predicted_summary}
</predicted_summary>
<reference_summary>
{reference_summary}
</reference_summary>
</example>
""".strip()


class SummarizationCritiquePrompt(CritiquePrompt):
    def fill(
        self,
        prompt,
        predictions: List[Union[str, Any]],
        few_shot_examples: List[Example],
    ) -> str:
        examples = "\n".join(
            _E.format(document=e.x, predicted_summary=p, reference_summary=e.y)
            for p, e in zip(predictions, few_shot_examples)
        )
        prompt = _P.format(instruction=prompt, examples=examples)
        return prompt

    def parse(self, generation: str):
        found = extract_xml_tag(generation, "critique")
        if found:
            return found
        return generation


_E_MISTRAL = """
EXAMPLE {id}
INPUT:
{document}
PREDICTED_SUMMARY:
{predicted_summary}
REFERENCE_SUMMARY:
{reference_summary}
""".strip()

_E_MISTRAL_NO_INPUT = """
EXAMPLE {id}
PREDICTED_SUMMARY:
{predicted_summary}
REFERENCE_SUMMARY:
{reference_summary}
""".strip()

_P_MISTRAL = """
In a summarization task, a writer is given an input text to write a summary following an instruction.

INSTRUCTION: 
{instruction}

Here are a few examples using the instruction. 
{examples}

Write a general and helpful critique to improve the instruction such that the predicted summaries are as close to references as possible.

1. Come up with several dimensions to compare its predicted summaries and reference summaries, e.g., number of words, number of sentences, style, precision, recall, etc.
2. List the difference predicted summaries and references on each dimension.
3. Identify specific phrases in the instruction that could have gotten these predicted summaries different with references on each dimension.
4. Suggest specific action items that are general to all examples and helpful to improve the instruction.

""".strip()


class SummarizationCritiquePromptForMistral(CritiquePrompt):
    def fill(
        self,
        prompt,
        predictions: List[Union[str, Any]],
        few_shot_examples: List[Example],
    ) -> str:
        examples = "\n\n".join(
            _E_MISTRAL.format(
                id=(i + 1), document=e.x, predicted_summary=p, reference_summary=e.y
            )
            for i, (p, e) in enumerate(zip(predictions, few_shot_examples))
        )
        prompt = _P_MISTRAL.format(instruction=prompt, examples=examples)
        return prompt

    def parse(self, generation: str):
        return generation


class SummarizationCritiquePromptForMistralNoInput(CritiquePrompt):
    def fill(
        self,
        prompt,
        predictions: List[Union[str, Any]],
        few_shot_examples: List[Example],
    ) -> str:
        examples = "\n\n".join(
            _E_MISTRAL_NO_INPUT.format(
                id=(i + 1), predicted_summary=p, reference_summary=e.y
            )
            for i, (p, e) in enumerate(zip(predictions, few_shot_examples))
        )
        prompt = _P_MISTRAL.format(instruction=prompt, examples=examples)
        return prompt

    def parse(self, generation: str):
        return generation
