"""Tests for storage module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nerfprobe.storage import ResultStore, BaselineStore
from nerfprobe_core import ProbeResult, ModelTarget, ProbeType


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_result():
    target = ModelTarget(provider_id="openai", model_name="gpt-4o")
    return ProbeResult(
        probe_name="math_probe",
        probe_type=ProbeType.MATH,
        target=target,
        passed=True,
        score=1.0,
        latency_ms=100.0,
        raw_response="Answer",
    )


class TestResultStore:
    def test_append_creates_file(self, temp_data_dir, mock_result):
        store = ResultStore(path=temp_data_dir)
        store.append(mock_result)
        
        assert (temp_data_dir / "results.jsonl").exists()
        with open(temp_data_dir / "results.jsonl") as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["probe_name"] == "math_probe"
            assert data["score"] == 1.0

    def test_get_recent(self, temp_data_dir, mock_result):
        store = ResultStore(path=temp_data_dir)
        for i in range(10):
            mock_result.score = float(i)
            store.append(mock_result)
            
        recent = store.get_recent(limit=5)
        assert len(recent) == 5
        # Should be most recent first (last in file is most recent)
        assert recent[0]["score"] == 9.0
        assert recent[4]["score"] == 5.0

    def test_get_trends(self, temp_data_dir, mock_result):
        store = ResultStore(path=temp_data_dir)
        store.append(mock_result)
        
        trends = store.get_trends(model="gpt-4o")
        assert "math_probe" in trends
        assert len(trends["math_probe"]) == 1
        assert trends["math_probe"][0][1] == 1.0


class TestBaselineStore:
    def test_save_and_get_baseline(self, temp_data_dir, mock_result):
        store = BaselineStore(path=temp_data_dir)
        
        results = [mock_result, mock_result] # 1.0, 1.0
        store.save_baseline("gpt-4o", results)
        
        score = store.get_baseline_score("gpt-4o", "math_probe")
        assert score == 1.0
        
        baselines = store.get_model_baselines("gpt-4o")
        assert "math_probe" in baselines
        assert baselines["math_probe"]["samples"] == 2

    def test_updates_existing_baseline(self, temp_data_dir, mock_result):
        store = BaselineStore(path=temp_data_dir)
        
        store.save_baseline("gpt-4o", [mock_result])
        assert store.get_baseline_score("gpt-4o", "math_probe") == 1.0
        
        # New baseline with lower score
        mock_result.score = 0.5
        store.save_baseline("gpt-4o", [mock_result])
        assert store.get_baseline_score("gpt-4o", "math_probe") == 0.5
