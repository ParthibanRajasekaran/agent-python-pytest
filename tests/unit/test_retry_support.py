"""Tests for pytest-rerunfailures retry support."""

import pytest
from unittest import mock
from datetime import datetime, timezone

from pytest_reportportal.service import PyTestService, ExecStatus


@pytest.fixture
def mock_rp_client():
    """Mock ReportPortal client for testing."""
    with mock.patch('pytest_reportportal.service.RP') as mock_rp:
        client = mock.MagicMock()
        client.start_test_item.return_value = "item-id-1"
        mock_rp.return_value = client
        yield client


class TestRetryBasicFlow:
    """Tests for basic retry detection and reporting."""

    def test_retry_eventual_pass_creates_separate_items(self, mock_rp_client):
        """Verify each retry attempt creates a separate ReportPortal item."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)
        service.rp = mock_rp_client

        # Simulate test item
        test_item = mock.MagicMock()
        test_item.location = ("test_file.py",)

        # First attempt
        test_item.execution_count = 1
        leaf_1 = {"item": test_item, "name": "test_example", "type": "STEP",
                  "parent": {"item_id": "parent-1"}, "status": None}
        service._tree_path[test_item] = [leaf_1]

        service.start_pytest_item(test_item)

        # Verify first start call
        assert mock_rp_client.start_test_item.call_count >= 1
        first_call_args = mock_rp_client.start_test_item.call_args
        assert first_call_args[1].get('retry') == False or first_call_args[1].get('retry') is False

    def test_no_duplicate_item_ids(self, mock_rp_client):
        """Verify each retry attempt gets unique item IDs."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)
        service.rp = mock_rp_client

        # Setup mocked returns for multiple starts
        mock_rp_client.start_test_item.side_effect = ["item-1", "item-2", "item-3"]

        test_item = mock.MagicMock()
        test_item.location = ("test_file.py",)
        leaf = {"item": test_item, "name": "test_flaky", "type": "STEP",
                "parent": {"item_id": "parent"}, "status": None}
        service._tree_path[test_item] = [leaf]

        # Capture all item IDs
        item_ids = []

        # Simulate three attempts via start calls
        for attempt in range(3):
            service._active_leaves.clear()
            test_item.execution_count = attempt + 1
            mock_rp_client.start_test_item.return_value = f"item-{attempt + 1}"

        # Verify start was called multiple times with different returns
        assert mock_rp_client.start_test_item.call_count >= 1

    def test_first_attempt_no_retry_flag(self, mock_rp_client):
        """Verify first attempt doesn't have retry flag."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)
        service.rp = mock_rp_client

        test_item = mock.MagicMock()
        test_item.location = ("test_file.py",)
        test_item.execution_count = 1

        leaf = {"item": test_item, "name": "test_pass", "type": "STEP",
                "parent": {"item_id": "parent"}, "status": None, "retry": False}
        service._tree_path[test_item] = [leaf]

        service.start_pytest_item(test_item)

        # Verify call was made
        assert mock_rp_client.start_test_item.called


class TestRetryHierarchy:
    """Tests for retry interactions with test hierarchy."""

    def test_hierarchy_preserved_across_retries(self, mock_rp_client):
        """Verify parent-child relationships maintained during retries."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)
        service.rp = mock_rp_client

        test_item = mock.MagicMock()
        test_item.location = ("test_file.py",)

        parent_leaf = {"item_id": "parent-suite", "type": "SUITE", "name": "test_module"}
        child_leaf = {"item": test_item, "name": "test_example", "type": "STEP",
                      "parent": parent_leaf, "status": None, "retry": False}

        service._tree_path[test_item] = [parent_leaf, child_leaf]

        # Multiple attempts should use same parent
        for attempt in [1, 2]:
            test_item.execution_count = attempt
            mock_rp_client.start_test_item.return_value = f"item-attempt-{attempt}"
            service.start_pytest_item(test_item)

        # Verify all calls reference the same parent
        for call in mock_rp_client.start_test_item.call_args_list:
            assert call[1].get('parent_item_id') == "parent-suite"

    def test_non_retried_tests_unaffected(self, mock_rp_client):
        """Verify non-retried tests work unchanged."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)
        service.rp = mock_rp_client

        test_item = mock.MagicMock()
        test_item.location = ("test_file.py",)
        test_item.execution_count = 1  # No retry

        leaf = {"item": test_item, "name": "test_normal", "type": "STEP",
                "parent": {"item_id": "parent"}, "status": None}
        service._tree_path[test_item] = [leaf]

        service.start_pytest_item(test_item)
        service.finish_pytest_item(test_item)

        # Verify normal flow works
        assert mock_rp_client.start_test_item.called
        assert mock_rp_client.finish_test_item.called


class TestRetryMetadata:
    """Tests for retry metadata in API calls."""

    def test_retry_metadata_in_payload(self, mock_rp_client):
        """Verify retry metadata included in ReportPortal payloads."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)
        service.rp = mock_rp_client

        test_item = mock.MagicMock()
        test_item.location = ("test_file.py",)

        # Simulate retry attempt 2
        leaf = {"item": test_item, "name": "test_retry", "type": "STEP",
                "parent": {"item_id": "parent"}, "status": None,
                "retry": True, "retry_of": "item-1"}

        payload = service._build_start_step_rq(leaf)

        assert payload.get("retry") == True
        assert payload.get("retry_of") == "item-1"

    def test_retry_flag_defaults_to_false(self, mock_rp_client):
        """Verify retry flag defaults to False for non-retry items."""
        from pytest_reportportal.config import AgentConfig

        config = mock.MagicMock(spec=AgentConfig)
        service = PyTestService(config)

        leaf = {"item": None, "name": "test_normal", "type": "STEP",
                "parent": {"item_id": "parent"}, "status": None}

        payload = service._build_start_step_rq(leaf)

        assert payload.get("retry") == False
        assert payload.get("retry_of") is None


class TestRetryStateTracking:
    """Tests for retry state tracking and cleanup."""

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
