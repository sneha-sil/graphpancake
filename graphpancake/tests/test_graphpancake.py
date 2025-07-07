"""
Unit and regression test for the graphpancake package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import graphpancake


def test_graphpancake_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "graphpancake" in sys.modules
