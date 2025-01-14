# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import traceback

from crispo.llms import LargeLanguageModel, TYPE_PROMPT
from crispo.llms.bedrock.wrapper import BedrockWrapper


class MistralWrapper(BedrockWrapper):

    def build_payload(
        self,
        prompt: TYPE_PROMPT,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        return {
            "prompt": f"<s>[INST] {prompt}\n[/INST]",
            "temperature": temperature,
            "top_p": top_p,
        }

    def parse_response(self, response: dict) -> str:
        return response["outputs"][0]["text"].lstrip()


class BedrockMistral(LargeLanguageModel):
    def __init__(self, max_new_tokens: int = None, temperature: float = None) -> None:
        super().__init__(max_new_tokens, temperature)
        self.inferencer = MistralWrapper("mistral.mistral-7b-instruct-v0:2")

    def generate(self, prompt: TYPE_PROMPT) -> str:
        if not prompt:
            return ""
        try:
            generation: str = self.inferencer.generate(prompt)
        except:
            traceback.print_exc()
            logging.warning("Mistral error.")
            return ""

        if not isinstance(generation, str):
            return ""
        generation = generation.lstrip()
        return generation


def main():
    import time

    inferencer = BedrockMistral()
    start = time.time()
    print(inferencer.generate("Who are you?"))
    print(f"1 call : {time.time() - start:.1f} seconds")

    start = time.time()
    print(list(inferencer.batch_generate(["Who are you?", "Good morning?"])))
    print(f"2 calls: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
