# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import re


def extract_xml_tag(generation: str, tag):
    begin = generation.rfind(f"<{tag}>")
    if begin == -1:
        return
    begin = begin + len(f"<{tag}>")
    end = generation.rfind(f"</{tag}>", begin)
    if end == -1:
        return
    value = generation[begin:end].strip()
    return value


def trim_space(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text
