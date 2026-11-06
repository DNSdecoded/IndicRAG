import sqlite3

import pytest

import baseline_study as bs


@pytest.mark.parametrize("q,expected", [
    ("RO4003C loss tangent", True),            # identifier, short
    ("n257 array gain", True),
    ("SPIBB downtilt", True),                  # acronym
    ('what does "slot filling factor" mean', True),   # quoted phrase
    ("How does reinforcement learning tune base station tilt?", False),
    ("Explain the 2 main contributions of the paper on surrogate models in detail please", False),  # long, number only
    ("antenna ka gain kaise badhaye", False),
])
def test_lexical_exact_proxy(q, expected):
    assert bs.lexical_exact_proxy(q) is expected


def test_auc():
    assert bs._auc([3, 4], [1, 2]) == 1.0
    assert bs._auc([1], [1]) == 0.5


def test_query_log_share_reads_a_copy(tmp_path):
    db = tmp_path / "s.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE query_log (query_id TEXT, question TEXT)")
    con.executemany("INSERT INTO query_log VALUES (?, ?)",
                    [("1", "RO4003C loss tangent"), ("2", "how do surrogate models reduce simulation cost?")])
    con.commit()
    con.close()
    res = bs.query_log_share(db)
    assert res["n"] == 2 and res["share"] == 0.5
