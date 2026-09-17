"""Tests for pytest-rerunfailures retry support."""

import pytest
from unittest import mock

from pytest_reportportal.service import PyTestService


class TestRetryDetection:
    """Tests for retry detection and state tracking."""

    def test_detect_retry_attempt_returns_execution_count(self):
        """Verify execution_count is returned from detect_retry_attempt."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        test_item = mock.MagicMock()
        test_item.execution_count = 1

        result = service._detect_retry_attempt(test_item)
        assert result == 1

    def test_detect_retry_attempt_second_execution(self):
        """Verify second execution_count is detected."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        test_item = mock.MagicMock()
        test_item.execution_count = 2

        result = service._detect_retry_attempt(test_item)
        assert result == 2

    def test_detect_retry_attempt_defaults_to_one(self):
        """Verify execution_count defaults to 1 if missing."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        test_item = mock.MagicMock(spec=[])  # No execution_count attribute

        result = service._detect_retry_attempt(test_item)
        assert result == 1

    def test_get_item_key_returns_object_id(self):
        """Verify item key is generated from object id."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        test_item = mock.MagicMock()
        key = service._get_item_key(test_item)

        assert key == str(id(test_item))

    def test_get_item_key_consistent(self):
        """Verify item key is consistent for same object."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        test_item = mock.MagicMock()
        key1 = service._get_item_key(test_item)
        key2 = service._get_item_key(test_item)

        assert key1 == key2

    def test_retry_state_tracks_execution_count(self):
        """Verify retry state tracks execution_count changes."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        test_item = mock.MagicMock()
        test_item.execution_count = 1

        key = service._get_item_key(test_item)
        service._retry_tracker[key] = {"last_reported_execution_count": 1}

        # Simulate retry
        test_item.execution_count = 2
        current_execution = service._detect_retry_attempt(test_item)
        last_reported = service._retry_tracker[key]["last_reported_execution_count"]

        assert current_execution > last_reported


class TestRetryMetadata:
    """Tests for retry metadata in payloads."""

    def test_retry_metadata_in_finish_payload(self):
        """Verify retry metadata included in finish payloads."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        leaf = {
            "name": "test_retry",
            "description": "Test description",
            "status": "PASSED",
            "item_id": "item-123",
            "retry": True,
            "retry_of": "parent-item-id"
        }

        payload = service._build_finish_step_rq(leaf)

        assert payload.get("retry") == True
        assert payload.get("retry_of") == "parent-item-id"

    def test_retry_metadata_defaults_to_false(self):
        """Verify retry metadata defaults to False."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        leaf = {
            "name": "test_normal",
            "description": "Test description",
            "status": "PASSED",
            "item_id": "item-456"
        }

        payload = service._build_finish_step_rq(leaf)

        assert payload.get("retry") == False
        assert payload.get("retry_of") is None

    def test_start_payload_includes_retry_fields(self):
        """Verify start payload includes retry fields."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        leaf = {
            "name": "test_item",
            "description": "Test description",
            "parent": mock.MagicMock(),
            "retry": True,
            "retry_of": "previous-item-id"
        }

        payload = service._build_start_step_rq(leaf)

        assert "retry" in payload
        assert payload.get("retry") == True
        assert payload.get("retry_of") == "previous-item-id"


class TestRetryStateTracking:
    """Tests for retry state management."""

    def test_retry_state_initialized(self):
        """Verify retry tracking dicts initialized."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        assert hasattr(service, '_retry_tracker')
        assert hasattr(service, '_active_leaves')
        assert isinstance(service._retry_tracker, dict)
        assert isinstance(service._active_leaves, dict)

    def test_cleanup_clears_state(self):
        """Verify cleanup properly clears retry state."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        # Populate state
        service._retry_tracker["test"] = {"data": "value"}
        service._active_leaves["test"] = {"leaf": "data"}

        # Cleanup
        service.cleanup_retry_state()

        assert len(service._retry_tracker) == 0
        assert len(service._active_leaves) == 0

    def test_active_leaves_tracks_current_attempt(self):
        """Verify active_leaves tracks the current retry attempt."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        test_item = mock.MagicMock()

        leaf_attempt_1 = {
            "name": "test_flaky",
            "item_id": "item-attempt-1",
            "execution_count": 1
        }

        key = service._get_item_key(test_item)
        service._active_leaves[key] = leaf_attempt_1

        assert service._active_leaves[key]["item_id"] == "item-attempt-1"

        # Simulate second attempt
        leaf_attempt_2 = {
            "name": "test_flaky",
            "item_id": "item-attempt-2",
            "execution_count": 2
        }
        service._active_leaves[key] = leaf_attempt_2

        assert service._active_leaves[key]["item_id"] == "item-attempt-2"

    def test_retry_tracker_tracks_last_reported(self):
        """Verify retry_tracker tracks last reported execution."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        test_item = mock.MagicMock()
        key = service._get_item_key(test_item)

        # First execution
        service._retry_tracker[key] = {
            "last_reported_execution_count": 1,
            "attempts": ["item-1"]
        }

        # Track second attempt
        service._retry_tracker[key]["last_reported_execution_count"] = 2
        service._retry_tracker[key]["attempts"].append("item-2")

        assert len(service._retry_tracker[key]["attempts"]) == 2
        assert service._retry_tracker[key]["last_reported_execution_count"] == 2
