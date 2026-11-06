import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

import fetch  # noqa: E402

GOOD = {"id": "2608.23469v1", "sha256": "a" * 64}


def _manifest(tmp_path, *papers):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"papers": list(papers)}))
    return p


def test_file_and_url_are_derived_from_the_id_not_the_manifest(tmp_path):
    tampered = dict(GOOD, file="../../../etc/cron.d/x", url="http://evil.example/payload")
    [p] = fetch.load_manifest(_manifest(tmp_path, tampered))
    assert p["file"] == "2608.23469v1.pdf" and p["url"] == "https://arxiv.org/pdf/2608.23469v1"


@pytest.mark.parametrize("entry", [
    dict(GOOD, id="../x"), dict(GOOD, id="2608.23469"), dict(GOOD, id="2608.23469v1/../../a"),
    dict(GOOD, sha256="zz" * 32), dict(GOOD, sha256="a" * 63), {"sha256": "a" * 64},
])
def test_malformed_entries_are_rejected(tmp_path, entry):
    with pytest.raises(ValueError, match="bad manifest entry"):
        fetch.load_manifest(_manifest(tmp_path, entry))


def test_committed_manifest_is_valid():
    assert len(fetch.load_manifest()) == 41
