# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import abc
import asyncio
import contextvars
import functools
from typing import List, Sequence, Union, Dict

from tqdm.asyncio import tqdm_asyncio

TYPE_PROMPT = Union[str, List[Dict]]


class LargeLanguageModel(abc.ABC):
    def __init__(
        self,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 1,
        concurrency: int = 16,
        stop_sequences: Sequence[str] = (),
    ) -> None:
        """
        A blackbox Large Language Model.

        Args:
            max_new_tokens: The maximum number of tokens to sample.
            temperature: Sampling temperature.
            concurrency: The number of concurrent requests allowed per process (worker).
        """
        self.stop_sequences = stop_sequences
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._concurrency = concurrency
        self.lock = asyncio.Semaphore(concurrency)

    @property
    def concurrency(self):
        return self._concurrency

    @concurrency.setter
    def concurrency(self, concurrency):
        self._concurrency = concurrency
        self.lock = asyncio.Semaphore(concurrency)

    @abc.abstractmethod
    def generate(self, prompt: TYPE_PROMPT) -> str:
        """
        The actual implementation of the LLM, blocking API calls.

        Args:
            prompt: A prompt in Claude 2 format (https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-text-completion.html).

        Returns:
            The completion.
        """
        pass

    async def generate_async(self, prompt: TYPE_PROMPT) -> str:
        """
        Asynchronous API call to the LLM for a completion

        Args:
            prompt: A prompt in Claude 2 format (https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-text-completion.html).

        Returns:
            The completion future event.
        """
        async with self.lock:
            loop = asyncio.events.get_running_loop()
            ctx = contextvars.copy_context()
            func_call = functools.partial(ctx.run, self.generate, prompt)
            return await loop.run_in_executor(None, func_call)

    def batch_generate(
        self, prompts: List[TYPE_PROMPT], desc="Generating"
    ) -> List[str]:
        async def fire(_tasks):
            return await tqdm_asyncio.gather(*_tasks, desc=desc, disable=not desc)

        tasks = [self.generate_async(prompt) for prompt in prompts]

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(fire(tasks))
