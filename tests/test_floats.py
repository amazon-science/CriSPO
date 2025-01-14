# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from crispo.metrics.floats import FloatList, FloatDict


def test_float_list():
    floats = FloatList(2.0, 4.0)
    assert floats == 3.0
    assert floats.scores == (2.0, 4.0)


def test_float_dict():
    scores = dict(a=2.0, b=4.0)
    floats = FloatDict(**scores)
    assert floats == 3.0
    assert floats.scores == scores


def test_float_dict_override_value():
    scores = dict(a=2.0, b=4.0)
    floats = FloatDict(value=-1.0, **scores)
    assert floats == -1.0
    assert floats.scores == scores
