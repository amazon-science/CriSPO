# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Optional, List, Tuple
from experiments.webnlg.example import WebNLGExample
from experiments.webnlg.task_prompt import WebNLGTaskPrompt
from crispo.optimizer.opro_meta_prompt import OproMetaPrompt
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag

_P = """
Your task is to generate the instruction <instruction/> for a data-to-text Natural Language Generation task.

Below are some examples:
{examples}

Below are some previous instructions with their scores. The score ranges from 0 to 100.
{instructions}

Generate an instruction that is different from all the instructions above, and has a higher score than all the instructions above.
It should be concise, effective, and generally applicable to all examples above.

Write your final new instruction in <instruction/> tags.
""".strip()

_E = """
<example>
<instruction>?</instruction>
<triples>
{triples}
</triples>
<reference_text>
{reference_text}
</reference_text>
</example>
""".strip()

_I = """
<rated_instruction>
<instruction>{instruction}</instruction>
<score>{score}</score>
</rated_instruction>
""".strip()


@dataclass
class WebNLGOproMetaPrompt(OproMetaPrompt):
    prompt: str = _P

    def fill(
        self,
        prompt_score_pairs: List[Tuple[TaskPrompt, float]],
        few_shot_examples: List[WebNLGExample],
        **kwargs,
    ) -> str:
        examples = self.format_examples(few_shot_examples)

        instructions = []
        for prompt, score in prompt_score_pairs:
            instructions.append(
                _I.format(
                    instruction=prompt,
                    score=score if isinstance(score, str) else f"{score:.2f}%",
                )
            )

        meta_prompt_text = _P.format(
            examples="\n".join(examples),
            instructions="\n".join(instructions),
        )
        return meta_prompt_text

    @staticmethod
    def format_examples(few_shot_examples):
        examples = []
        if few_shot_examples:
            for i, example in enumerate(few_shot_examples):
                example: WebNLGExample = example
                examples.append(
                    _E.format(
                        triples="\n".join(example.x),
                        reference_text="\n".join(
                            f"<reference_text_{j + 1}>{text}</reference_text_{j + 1}>"
                            for j, text in enumerate(example.y)
                        ),
                    )
                )
        return examples

    def parse(self, generation: str) -> Optional[TaskPrompt]:
        new_inst = extract_xml_tag(generation, "instruction")
        if new_inst is None:
            return None
        if (
            "<text>" not in new_inst
            and "</text>" not in new_inst
            and "<text/>" not in new_inst
        ):
            new_inst += "\nWrite down your natural-language text in <text/>."
        return WebNLGTaskPrompt(new_inst)
