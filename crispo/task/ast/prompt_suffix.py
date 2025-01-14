# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import abc
from dataclasses import dataclass
from crispo.task.prompt import TaskPrompt


@dataclass(eq=True, frozen=True)
class TaskPromptSuffix(TaskPrompt, abc.ABC):
    main_prompt: str
