# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import os
from time import sleep
from typing import Sequence

import boto3
import botocore
from botocore.config import Config
import botocore.errorfactory
import botocore.exceptions
from botocore.credentials import InstanceMetadataProvider
from botocore.utils import InstanceMetadataFetcher

from crispo.llms import TYPE_PROMPT


class BedrockWrapper:
    def __init__(
        self,
        model_name: str = "anthropic.claude-instant-v1",
        max_retries: int = 5,
        throttling_retries: int = 99999,
        throttling_wait: int = 2,
        aws_profile: str = os.environ.get("AWS_PROFILE"),
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 1,
        max_pool_connections: int = 16,
        stop_sequences: Sequence[str] = ("\n\nHuman:",),
    ):
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.throttling_wait = throttling_wait
        self.throttling_retries = throttling_retries
        self.model_name = model_name
        self.stop_sequences = list(stop_sequences)
        self.aws_profile = aws_profile
        self.max_retries = max_retries
        self.max_pool_connections = max_pool_connections
        self.bedrock = self.initialize_bedrock()

    def initialize_bedrock(self):
        if self.aws_profile:
            session = boto3.Session(profile_name=self.aws_profile)
        else:
            credentials_provider = InstanceMetadataProvider(
                iam_role_fetcher=InstanceMetadataFetcher(timeout=10, num_attempts=3)
            )
            boto3_credentials = credentials_provider.load().get_frozen_credentials()
            session = boto3.Session(
                aws_access_key_id=boto3_credentials.access_key,
                aws_secret_access_key=boto3_credentials.secret_key,
                aws_session_token=boto3_credentials.token,
            )

        config = Config(
            read_timeout=120,
            connect_timeout=120,
            max_pool_connections=self.max_pool_connections,
            retries={
                "max_attempts": self.max_retries,
            },
        )
        return session.client(
            "bedrock-runtime",
            "us-west-2",
            endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com/",
            config=config,
        )

    def build_payload(
        self,
        prompt: TYPE_PROMPT,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        return {
            "prompt": prompt,
            "max_tokens_to_sample": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop_sequences": self.stop_sequences,
            "anthropic_version": "bedrock-2023-05-31",
        }

    def generate(self, prompt: TYPE_PROMPT) -> str:
        body = self.build_payload(
            prompt, self.max_new_tokens, self.temperature, self.top_p, self.top_k
        )
        body = json.dumps(body)
        attempt = 0
        while attempt < self.throttling_retries:
            try:
                output = self.bedrock.invoke_model(
                    modelId=self.model_name,
                    body=body,
                    accept="application/json",
                    contentType="application/json",
                )

                response = json.loads(output["body"].read())
                result = self.parse_response(response)
                return result

            except (
                botocore.errorfactory.ClientError,
                botocore.exceptions.ClientError,
            ) as e:
                error_code = e.response["Error"]["Code"]
                print(f"{error_code}: {e}")
                if error_code == "ExpiredTokenException":
                    delay = 60
                    print(
                        f"Please update the credentials. Bedrock will retry in {delay} seconds."
                    )
                    sleep(delay)
                    self.bedrock = self.initialize_bedrock()
                    continue
                if error_code not in {
                    "ThrottlingException",
                    "ModelTimeoutException",
                    "ModelErrorException",
                    "ServiceUnavailableException",
                }:
                    raise e from None
                if attempt == self.throttling_retries - 1:
                    raise e from None
                sleep(self.throttling_wait * (attempt + 1))
            except (
                botocore.exceptions.ReadTimeoutError,
                botocore.exceptions.ConnectionClosedError,
            ) as e:
                print(e)
                sleep(3)
                continue
            attempt += 1

    def parse_response(self, response: dict) -> str:
        result = response["completion"]
        return result
