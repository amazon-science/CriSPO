# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from os.path import abspath, join, dirname

from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))
with open(join(this_dir, "README.md"), encoding="utf-8") as file:
    long_description = file.read()

METRICS = [
    "rouge_score",
    "sacrebleu",
    "alignscore @ git+https://github.com/yuh-zha/AlignScore.git",
    "evaluate",
]

setup(
    name="CriSPO",
    version="0.0.0",
    description="CriSPO: Multi-Aspect Critique-Suggestion-guided Automatic Prompt Optimization for Text Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["docs", "tests*", "experiments"]),
    include_package_data=True,
    install_requires=[
        "pandas",
        "tabulate",
        "tqdm",
        "termcolor",
        "datasets",
        "matplotlib",
        "boto3",
    ],
    extras_require={"experiments": METRICS},
    tests_require=["pytest", "black"],
    python_requires=">=3.10",
)
