# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from crispo.task.example import Example


def truncate_text(text, max_length):
    if max_length == -1:
        return text
    return " ".join(text.split()[:max_length])


def truncated_dataset_loader(dataset_loader, max_input=-1, max_output=-1, **kwargs):
    splits = dataset_loader(**kwargs)
    truncated_split = []

    for data_split in splits:
        truncated_split.append(
            [
                Example(
                    x=truncate_text(example.x, max_input),
                    y=truncate_text(example.y, max_output),
                )
                for example in data_split
            ]
        )

    return tuple(truncated_split)
