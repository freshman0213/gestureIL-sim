# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

"""Unit tests for the `gestureIL-sim` package version."""

# SRL
import gestureIL


def test_gestureIL_sim_version() -> None:
    """Test `gestureIL-sim` package version is set."""
    assert gestureIL.__version__ is not None
    assert gestureIL.__version__ != ""
