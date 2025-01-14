# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag

TRIPLES_PLACEHOLDER = "TRIPLES_PLACEHOLDER"

_P = f"""
You are provided with structured data in the form of RDF triples. Each triple contains a subject, predicate, and object, which together describe an attribute or relationship for an entity. Your task is to generate a coherent, natural-language sentence or short paragraph that conveys the information in the data accurately and fluently. Aim to integrate all details from the data in a way that reads smoothly to a human.

Triples:

{TRIPLES_PLACEHOLDER}

Write down your natural-language text in <text/>.
""".strip()


@dataclass(eq=True, frozen=True)
class WebNLGTaskPrompt(TaskPrompt):
    prompt: str = _P

    def fill(self, x: list[str]) -> str:
        return self.prompt.replace(TRIPLES_PLACEHOLDER, "\n".join(x))

    def parse(self, generation: str, x=None):
        text = extract_xml_tag(generation, "text") or generation
        return text

    def __str__(self) -> str:
        return self.prompt
