# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Union, Any, List

from crispo.task.critique import CritiquePrompt
from crispo.task.example import Example
from crispo.utilities.prompt_util import extract_xml_tag
from experiments.nq.example import encode_context

_P = """
In a question-answering task, question and context are provided and the answer needs to be generated.

<instruction>{instruction}</instruction>
<examples>
{examples}
</examples>

Write a general and helpful critique in <critique> XML tags to improve the instruction such that the generated answer are the same as gold answer.

1. Come up with several dimensions to compare its generated and gold answer, e.g., number of words, style, precision, recall, etc.
2. List the difference between generated and gold answer on each dimension.
3. Identify specific phrases in the instruction that could have gotten these generated answer different with gold one on each dimension.
4. Suggest specific action items that are general to all examples and helpful to improve the instruction.
""".strip()

_E = """
<example>
<question>
{question}
</question>
{context}
<generated_answer>
{generated_answer}
</generated_answer>
<gold_answer>
{gold_answer}
</gold_answer>
</example>
""".strip()


class RAGCritiquePrompt(CritiquePrompt):
    def fill(
        self,
        prompt,
        predictions: List[Union[str, Any]],
        few_shot_examples: List[Example],
    ) -> str:
        examples = "\n".join(
            _E.format(
                question=e.x[0],
                context=encode_context(e.x, 20),
                generated_answer=p,
                gold_answer=e.get_y_str(),
            )
            for p, e in zip(predictions, few_shot_examples)
        )
        prompt = _P.format(instruction=prompt, examples=examples)
        # print ('===critique prompt===')
        # print (prompt)
        # print ('================================')
        return prompt

    def parse(self, generation: str):
        found = extract_xml_tag(generation, "critique")
        if found:
            return found
        return generation
