# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List
import matplotlib.pyplot as plt


def boxplot(data: List[List[float]], xlabel=None, ylabel=None, title=None):
    labels = list(range(len(data)))
    fig = plt.figure(figsize=(10, 8), dpi=80)
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.suptitle(title)
    return fig


def main():
    boxplot(
        data=[[0.5], [0.1, 0.3], [0.6, 0.8]], xlabel="step", ylabel="acc", title="GSM"
    )
    plt.show()


if __name__ == "__main__":
    main()
