# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
from typing import Union, List, Dict, Sequence

from crispo.llms import LargeLanguageModel, TYPE_PROMPT
from crispo.llms.bedrock.wrapper import BedrockWrapper


class BedrockClaude3(BedrockWrapper):
    def __init__(
        self,
        model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        max_retries: int = 5,
        throttling_retries: int = 99999,
        throttling_wait: int = 2,
        aws_profile: str = os.environ.get("AWS_PROFILE"),
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 1,
        max_pool_connections: int = 16,
        stop_sequences: Sequence[str] = (),
    ):
        super().__init__(
            model_name,
            max_retries,
            throttling_retries,
            throttling_wait,
            aws_profile,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            max_pool_connections,
            stop_sequences,
        )

    def build_payload(
        self,
        prompt: Union[str, List[Dict]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        return {
            "messages": (
                self.get_input_msg_claude3(prompt)
                if isinstance(prompt, str)
                else prompt
            ),
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop_sequences": self.stop_sequences,
            "anthropic_version": "bedrock-2023-05-31",
        }

    @staticmethod
    def get_input_msg_claude3(prompt: str):
        prompt = prompt.strip().lstrip("Human:").strip()
        assistant_part = ""
        if "Assistant" in prompt:
            assistant_part_i = prompt.find("Assistant:")
            assistant_part = prompt[assistant_part_i + len("Assistant:") :].strip()
            prompt = prompt[:assistant_part_i]
        input_msg = [{"role": "user", "content": prompt}]
        if assistant_part:
            input_msg.append({"role": "assistant", "content": assistant_part})
        return input_msg

    def parse_response(self, response: dict) -> str:
        content = response["content"]
        return content[0]["text"] if content else ""


class Claude3(LargeLanguageModel):
    def __init__(
        self,
        model_name: str,
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
        self.client = BedrockClaude3(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
        )

    def generate(self, prompt: TYPE_PROMPT) -> str:
        generation: str = self.client.generate(prompt)
        generation = generation.lstrip()
        return generation


class ClaudeSonnet(Claude3):

    def __init__(
        self,
        model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 1,
        concurrency: int = 8,
        stop_sequences: Sequence[str] = (),
    ) -> None:
        super().__init__(
            model_name,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            concurrency,
            stop_sequences,
        )


class ClaudeSonnet35(Claude3):

    def __init__(
        self,
        model_name: str = "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 1,
        concurrency: int = 8,
        stop_sequences: Sequence[str] = (),
    ) -> None:
        super().__init__(
            model_name,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            concurrency,
            stop_sequences,
        )


class ClaudeHaiku(Claude3):

    def __init__(
        self,
        model_name: str = "anthropic.claude-3-haiku-20240307-v1:0",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 1,
        concurrency: int = 16,
        stop_sequences: Sequence[str] = (),
    ) -> None:
        super().__init__(
            model_name,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            concurrency,
            stop_sequences,
        )


class ClaudeHaiku35(Claude3):

    def __init__(
        self,
        model_name: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 1,
        concurrency: int = 16,
        stop_sequences: Sequence[str] = (),
    ) -> None:
        super().__init__(
            model_name,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            concurrency,
            stop_sequences,
        )


def main():
    llm = ClaudeHaiku()
    print(llm.generate("Who are you?"))


if __name__ == "__main__":
    main()
