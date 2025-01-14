# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import List, Tuple

from crispo.task.example import Example


def encode_context(x: Tuple[str, List[str]], k=-1):
    context_str = ""
    context_selected = x[1]
    if k > 0:
        context_selected = context_selected[:k]

    for idx, context in enumerate(context_selected):
        context_str += f"Context {idx + 1}: {context}\n\n"
    context_str = f"<context>\n{context_str}</context>"
    return context_str


@dataclass
class RAGExample(Example):
    x: Tuple[str, List[str]]
    y: List[str]

    def get_y_str(self):
        if type(self.y) is dict:
            assert "text" in self.y
            return self.y["text"][0]
        else:
            return self.y[0]
