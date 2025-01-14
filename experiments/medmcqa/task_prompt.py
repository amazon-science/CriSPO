# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import Union, Any
from crispo.task.prompt import TaskPrompt
from crispo.utilities.prompt_util import extract_xml_tag
from experiments.gsm.example import Output

QUESTION_PLACEHOLDER = "QUESTION_PLACEHOLDER"

_P = f"""
Let\'s solve the multiple-choice problem. Chose an option and write it in <answer> XML tags.

{QUESTION_PLACEHOLDER}
"""


@dataclass(eq=True, frozen=True)
class MedMcqaTaskPrompt(TaskPrompt):
    prompt: str = _P

    def fill(self, x: Union[str, Any]) -> str:
        return self.prompt.replace(QUESTION_PLACEHOLDER, x)

    def parse(self, generation: str):
        return Output(label=self.parse_label(generation), reasoning=generation)

    @staticmethod
    def parse_label(generation):
        found = extract_xml_tag(generation, "answer") or generation
        found = extract_choice(found)
        return found

    def __str__(self) -> str:
        return self.prompt


def extract_choice(prediction: str):
    for template in [
        "{}.",
        "({}).",
        "{}).",
        "{})",
        "({})",
        "{}:",
        "Option {}",
        "option {}",
        " {} ",
        "{} ",
        "{}",
    ]:
        for choice in "ABCDEFGH":
            filled_choice = template.format(choice)
            if filled_choice in prediction:
                return choice


def main():
    generation = """
    "Okay, here is my step-by-step work for this word problem:

Problem-solving strategy: 
1) Define variables for the number of lines each character has
2) Write expressions relating the variables based on information given
3) Isolate the expression for the unknown variable 
4) Solve for the unknown

Known quantities:
- Third character (C3) has 2 lines  
- Second character (C2) has some number of lines
- First character (C1) has 8 more lines than C2

Unknown: Number of lines for first character (C1)

Steps:
1) Let C3 = 2 lines 
2) C2 = 3 * C3 + 6 = 3 * 2 + 6 = 12 lines  
3) C1 = C2 + 8 = 12 + 8 = 20 lines

The number of lines the first character has is 20.

I defined variables for each character and wrote expressions directly relating them based on the information given in the problem. I isolated the variable for the unknown and solved it step-by-step without introducing any extraneous variables. Please let me know if you see any errors or flaws in my logic/reasoning for this word problem."
    """
    print(MedMcqaTaskPrompt.parse_label(generation))


if __name__ == "__main__":
    main()
