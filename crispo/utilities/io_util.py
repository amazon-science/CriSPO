# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import os
import pickle
from typing import Union


def save_pickle(item, path):
    with open(path, "wb") as f:
        pickle.dump(item, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(
    item: Union[dict, list, str, int, float],
    path: str,
    ensure_ascii=False,
    cls=None,
    default=lambda o: repr(o),
    indent=2,
):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        json.dump(
            item,
            out,
            ensure_ascii=ensure_ascii,
            indent=indent,
            cls=cls,
            default=default,
        )


def load_json(path):
    with open(path, encoding="utf-8") as src:
        return json.load(src)
