"""
Test suite for narrativesLLM package.
"""

import pytest

def pytest_configure(config):
    """Configure pytest for narrativesLLM tests."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )