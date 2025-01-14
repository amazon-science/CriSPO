# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass
from typing import List, Tuple

from elit_tokenizer import EnglishTokenizer


@dataclass
class Tokenization:
    sentences: List[str]
    tokens: List[List[str]]
    offsets: List[List[Tuple[int, int]]]


tokenizer = EnglishTokenizer()


def tokenize(text: str) -> Tokenization:
    raw_sents = tokenizer.decode(text, segment=2)
    sents = []
    tokens = []
    offsets = []
    for sentence in raw_sents:
        t = []
        o = []
        for token, offset in sentence:
            t.append(token)
            o.append(offset)
        sents.append(text[o[0][0] : o[-1][1]])
        tokens.append(t)
        offsets.append(o)
    return Tokenization(sents, tokens, offsets)


def main():
    text = "Emory NLP is a research lab in Atlanta, GA. It is founded by Jinho D. Choi in 2014. Dr. Choi is a professor at Emory University."
    print(tokenize(text))


if __name__ == "__main__":
    main()
