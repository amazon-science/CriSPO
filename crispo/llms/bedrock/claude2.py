# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Sequence

from crispo.llms.bedrock.wrapper import BedrockWrapper

from crispo.llms import LargeLanguageModel, TYPE_PROMPT


class ClaudeInstant(LargeLanguageModel):
    def __init__(
        self,
        model_name: str = "anthropic.claude-instant-v1",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 1,
        concurrency: int = 16,
        stop_sequences: Sequence[str] = (),
    ) -> None:
        super().__init__(
            max_new_tokens, temperature, top_p, top_k, concurrency, stop_sequences
        )
        self.client = BedrockWrapper(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        )

    def generate(self, prompt: TYPE_PROMPT) -> str:
        if isinstance(prompt, str):
            if not prompt.startswith("\n\nHuman: "):
                prompt = "\n\nHuman: " + prompt
            if not prompt.endswith("\n\nAssistant: "):
                prompt += "\n\nAssistant: "
        generation: str = self.client.generate(prompt)
        generation = generation.lstrip()
        return generation


def main():
    llm = ClaudeInstant()
    print(llm.generate("Who are you?"))


if __name__ == "__main__":
    main()
