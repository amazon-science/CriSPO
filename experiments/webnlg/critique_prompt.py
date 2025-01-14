# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List

from experiments.webnlg.example import WebNLGExample
from crispo.task.critique import CritiquePrompt
from crispo.utilities.prompt_util import extract_xml_tag

_P = """
For a data-to-text task, a writer converts a list of "subject | predicate | object" triples into natural language text following the instruction below:

<instruction>{instruction}</instruction>
<examples>
{examples}
</examples>

Write a general and helpful critique in <critique/> XML tags to improve the instruction such that the predicted text are as close to the reference text as possible.

1. Come up with several dimensions to compare its predicted text and reference text, e.g., number of words, style, precision, recall, etc.
2. List the difference between predicted text and references on each dimension.
3. Identify specific phrases in the instruction that could have gotten these predicted text different with references on each dimension.
4. Suggest specific action items that are general to all examples and helpful to improve the instruction.

""".strip()

_ERROR = """
<example>
<triples>
{triples}
</triples>

<predicted_text>
{predicted_text}
</predicted_text>

<reference_text>
{reference_text}
</reference_text>
</example>
""".strip()


class WebNLGCritiquePrompt(CritiquePrompt):

    def fill(
        self, prompt, predictions: List[str], few_shot_examples: List[WebNLGExample]
    ) -> str:
        errors = []
        for wrong, right in zip(predictions, few_shot_examples):
            wrong: str = wrong
            right: WebNLGExample = right
            errors.append(
                _ERROR.format(
                    triples="\n".join(right.x),
                    predicted_text=wrong,
                    reference_text="\n".join(
                        f"<reference_text_{j + 1}>{text}</reference_text_{j + 1}>"
                        for j, text in enumerate(right.y)
                    ),
                )
            )
        prompt = _P.format(instruction=prompt, examples="\n\n".join(errors))
        return prompt

    def parse(self, generation: str):
        found = extract_xml_tag(generation, "critique")
        if found:
            return found
        return generation
