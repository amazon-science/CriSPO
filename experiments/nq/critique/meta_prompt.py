# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Optional, List, Tuple

from experiments.nq.constants import CONTEXT_PLACEHOLDER, QUESTION_PLACEHOLDER
from experiments.nq.task_prompt import RAGTaskPromptUniversalTemplate
from experiments.nq import encode_context
from crispo.optimizer.opro_meta_prompt import OproMetaPrompt
from crispo.task.example import Example
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag

_P = """
Your task is to optimize the instruction for a question-answering task, where the question and context are provided.

Below are some examples:
{examples}

Below are some previous instructions with their scores and critiques.
{instructions}

Generate an instruction that is different from all the instructions above, and has a higher score than all the instructions above.
It should be concise, effective, and generally applicable to all examples above.

Draft your new instruction step by step:

1. Compare high-score instructions to low-score ones, identify what suggestions could have improved them. List them in <suggestion> tags.
2. Apply the suggestions and draft a new instruction aiming for a higher score.
3. Be creative and vary the wording, paraphrase, position of "{question_placeholder}", "{context_placeholder}", phrase order, grammar, sentence order, which specific examples to give, etc.
4. Write your final new instruction in <instruction> tags.
""".strip()

_E = """
<example>
<instruction>?</instruction>
<question>
{question}
</question>
{context}
<answer>
{answer}
</answer>
</example>
""".strip()

_I = """
<rated_instruction>
<instruction>{instruction}</instruction>
<score>{score}</score>
<critique>
{critique}
</critique>
</rated_instruction>
""".strip()


@dataclass
class RAGMetaPrompt(OproMetaPrompt):
    # noinspection PyMethodOverriding
    def fill(
        self,
        prompt_score_pairs: List[Tuple[TaskPrompt, float]],
        few_shot_examples: List[Example],
        critiques,
        **kwargs,
    ) -> str:
        examples = []
        if few_shot_examples:
            for i, example in enumerate(few_shot_examples):
                examples.append(
                    _E.format(
                        question=example.x[0],
                        context=encode_context(example.x, 20),
                        answer=example.get_y_str(),
                    )
                )

        instructions = []
        for (prompt, score), critique in zip(prompt_score_pairs, critiques):
            instructions.append(
                _I.format(
                    instruction=prompt,
                    score=score if isinstance(score, str) else f"{score:.2%}",
                    critique=critique,
                )
            )

        meta_prompt_text = _P.format(
            examples="\n".join(examples),
            instructions="\n".join(instructions),
            question_placeholder=QUESTION_PLACEHOLDER,
            context_placeholder=CONTEXT_PLACEHOLDER,
        )
        return meta_prompt_text

    def parse(self, generation: str) -> Optional[TaskPrompt]:
        new_inst = extract_xml_tag(generation, "instruction")
        if not new_inst:
            return None
        return RAGTaskPromptUniversalTemplate(new_inst)
