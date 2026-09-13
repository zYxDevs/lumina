"""Tests for utils.memory — memory pressure watchdog."""
from __future__ import annotations

import os
import time
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# ModelTracker (pure logic, no native calls)
# ---------------------------------------------------------------------------

class TestModelTracker:
    """Unit tests for the LRU eviction tracker."""

    def _make_tracker(self, **kwargs):
        from utils.memory import ModelTracker
        defaults = dict(threshold=0.80, min_loaded=1, idle_ttl=0)
        defaults.update(kwargs)
        return ModelTracker(**defaults)

    def test_touch_and_snapshot(self):
        t = self._make_tracker()
        t.touch("detect")
        snap = t.snapshot()
        assert "detect" in snap
        # age should be near zero
        assert snap["detect"] < 1.0

    def test_eviction_skipped_when_env_set(self, monkeypatch):
        t = self._make_tracker()
        unload = MagicMock()
        t.register("detect", unload)
        t.touch("detect")

        monkeypatch.setenv("LUMINA_KEEP_MODELS", "1")
        with patch("utils.memory.get_memory_usage", return_value=(0.95, "vram")):
            evicted = t.evict_if_needed()

        assert evicted == 0
        unload.assert_not_called()

    def test_no_eviction_under_threshold(self):
        t = self._make_tracker()
        unload = MagicMock()
        t.register("detect", unload)
        t.touch("detect")

        with patch("utils.memory.get_memory_usage", return_value=(0.50, "vram")):
            evicted = t.evict_if_needed()

        assert evicted == 0
        unload.assert_not_called()

    def test_no_eviction_when_memory_unknown(self):
        t = self._make_tracker()
        unload = MagicMock()
        t.register("detect", unload)
        t.touch("detect")

        with patch("utils.memory.get_memory_usage", return_value=(None, "none")):
            evicted = t.evict_if_needed()

        assert evicted == 0
        unload.assert_not_called()

    def test_evicts_lru_over_threshold(self):
        t = self._make_tracker(min_loaded=1)
        u1 = MagicMock()
        u2 = MagicMock()
        t.register("detect", u1)
        t.register("ocr", u2)

        # detect is older (touched first, then time passes)
        t.touch("detect")
        time.sleep(0.01)
        t.touch("ocr")

        with patch("utils.memory.get_memory_usage", return_value=(0.90, "vram")):
            evicted = t.evict_if_needed()

        assert evicted >= 1
        u1.assert_called_once()
        u2.assert_not_called()

    def test_respects_min_loaded(self):
        t = self._make_tracker(min_loaded=2)
        u1 = MagicMock()
        u2 = MagicMock()
        t.register("detect", u1)
        t.register("ocr", u2)

        t.touch("detect")
        t.touch("ocr")

        with patch("utils.memory.get_memory_usage", return_value=(0.95, "vram")):
            evicted = t.evict_if_needed()

        # both registered, min_loaded=2 → nothing evicted
        assert evicted == 0

    def test_eviction_error_handled(self):
        t = self._make_tracker()
        u1 = MagicMock(side_effect=RuntimeError("unload failed"))
        u2 = MagicMock()
        t.register("detect", u1)
        t.register("ocr", u2)

        t.touch("detect")
        time.sleep(0.01)
        t.touch("ocr")

        with patch("utils.memory.get_memory_usage", return_value=(0.90, "vram")):
            evicted = t.evict_if_needed()

        # detect unload failed, but tracker didn't crash
        assert evicted >= 0

    def test_unregister_via_access_removal(self):
        """Evicted model removed from access dict."""
        t = self._make_tracker(min_loaded=0)
        u1 = MagicMock()
        t.register("detect", u1)
        t.touch("detect")

        with patch("utils.memory.get_memory_usage", return_value=(0.90, "vram")):
            t.evict_if_needed()

        snap = t.snapshot()
        assert "detect" not in snap


# ---------------------------------------------------------------------------
# get_memory_usage — native queries (may return None on CI)
# ---------------------------------------------------------------------------

class TestGetMemoryUsage:
    """Smoke tests for native memory queries."""

    def test_returns_tuple(self):
        from utils.memory import get_memory_usage
        ratio, source = get_memory_usage()
        assert source in ("vram", "ram", "none")
        if ratio is not None:
            assert 0.0 <= ratio <= 1.0

    def test_check_pressure(self):
        from utils.memory import check_pressure
        # No assertion on result — just ensure it doesn't crash
        result = check_pressure(threshold=0.99)
        assert isinstance(result, bool)

    def test_ram_query(self):
        from utils.memory import _query_ram
        if os.name == "nt":
            ratio = _query_ram()
            # Should return a float on Windows
            if ratio is not None:
                assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# NVML probe (will be None on machines without NVIDIA GPU)
# ---------------------------------------------------------------------------

class TestNvmlProbe:
    def test_nvml_init_does_not_crash(self):
        from utils.memory import _init_nvml
        result = _init_nvml()
        assert isinstance(result, bool)

    def test_nvml_query(self):
        from utils.memory import _query_nvml
        ratio = _query_nvml()
        if ratio is not None:
            assert 0.0 <= ratio <= 1.0
