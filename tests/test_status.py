"""Heartbeat write/read roundtrip."""

import json

from rlvr_vs_base.status import write_status


def test_write_status_roundtrip(tmp_path):
    write_status(tmp_path, {"phase": "generate", "samples_done": 5})
    data = json.loads((tmp_path / "status.json").read_text())
    assert data["phase"] == "generate"
    assert data["samples_done"] == 5
    assert "updated" in data
    assert "disk_free_gb" in data


def test_write_status_overwrites_atomically(tmp_path):
    write_status(tmp_path, {"phase": "a"})
    write_status(tmp_path, {"phase": "b"})
    data = json.loads((tmp_path / "status.json").read_text())
    assert data["phase"] == "b"
    assert not list(tmp_path.glob(".tmp-*")), "tmp file must be renamed away"
