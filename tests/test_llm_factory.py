import os
from unittest.mock import patch

import pytest

from utils.llm_factory import get_model_id, use_bedrock


class TestUseBedrock:
    def test_false_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("USE_BEDROCK", None)
            assert use_bedrock() is False

    def test_true_when_set_lowercase(self):
        with patch.dict(os.environ, {"USE_BEDROCK": "true"}):
            assert use_bedrock() is True

    def test_true_when_set_uppercase(self):
        with patch.dict(os.environ, {"USE_BEDROCK": "TRUE"}):
            assert use_bedrock() is True

    def test_false_for_non_true_value(self):
        with patch.dict(os.environ, {"USE_BEDROCK": "yes"}):
            assert use_bedrock() is False

    def test_false_when_explicitly_false(self):
        with patch.dict(os.environ, {"USE_BEDROCK": "false"}):
            assert use_bedrock() is False


class TestGetModelId:
    def test_returns_anthropic_model_when_not_bedrock(self):
        with patch.dict(os.environ, {"USE_BEDROCK": "false", "ANTHROPIC_MODEL": "claude-sonnet-4-5"}):
            assert get_model_id() == "claude-sonnet-4-5"

    def test_returns_bedrock_model_when_bedrock(self):
        model = "us.anthropic.claude-sonnet-4-5-20250514-v1:0"
        with patch.dict(os.environ, {"USE_BEDROCK": "true", "BEDROCK_PRIMARY_MODEL": model}):
            assert get_model_id() == model

    def test_default_anthropic_model(self):
        with patch.dict(os.environ, {"USE_BEDROCK": "false"}):
            os.environ.pop("ANTHROPIC_MODEL", None)
            assert get_model_id() == "claude-sonnet-4-5"

    def test_default_bedrock_model_contains_claude(self):
        with patch.dict(os.environ, {"USE_BEDROCK": "true"}):
            os.environ.pop("BEDROCK_PRIMARY_MODEL", None)
            assert "claude" in get_model_id()

    def test_model_switches_with_bedrock_flag(self):
        with patch.dict(os.environ, {
            "USE_BEDROCK": "false",
            "ANTHROPIC_MODEL": "claude-sonnet-4-5",
            "BEDROCK_PRIMARY_MODEL": "us.anthropic.claude-sonnet-4-5-20250514-v1:0",
        }):
            direct_model = get_model_id()

        with patch.dict(os.environ, {
            "USE_BEDROCK": "true",
            "ANTHROPIC_MODEL": "claude-sonnet-4-5",
            "BEDROCK_PRIMARY_MODEL": "us.anthropic.claude-sonnet-4-5-20250514-v1:0",
        }):
            bedrock_model = get_model_id()

        assert direct_model != bedrock_model
