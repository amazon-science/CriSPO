# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Union, Any, List

from crispo.task.critique import CritiquePrompt
from crispo.task.example import Example
from crispo.utilities.prompt_util import extract_xml_tag

_P = """
In a task, an AI is given an input text to write a text following an instruction and a postscript.

<instruction>{instruction}</instruction>
<postscript>{postscript}</postscript>
<examples>
{examples}
</examples>

Write a general and helpful critique in <critique> XML tags to improve the postscript such that the predicted text are as close to references as possible.

1. Compare predicted text and the input text regarding their faithfulness.
2. Identify specific phrases in the postscript that could have gotten these predicted text unfaithful.
3. Suggest specific action items that are general to all examples and helpful to improve the postscript.

""".strip()

_E = """
<example>
<input>
{document}
</input>
<predicted_text>
{predicted_text}
</predicted_text>
</example>
""".strip()


class SuffixCritiquePrompt(CritiquePrompt):
    def __init__(self, main_prompt: str) -> None:
        super().__init__()
        self.main_prompt = main_prompt

    def fill(
        self,
        prompt,
        predictions: List[Union[str, Any]],
        few_shot_examples: List[Example],
    ) -> str:
        examples = "\n".join(
            _E.format(document=e.x, predicted_text=p)
            for p, e in zip(predictions, few_shot_examples)
        )
        prompt = _P.format(
            instruction=self.main_prompt, postscript=prompt, examples=examples
        )
        return prompt

    def parse(self, generation: str):
        found = extract_xml_tag(generation, "critique")
        if found:
            return found
        return generation
