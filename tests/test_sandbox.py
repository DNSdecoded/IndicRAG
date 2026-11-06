"""Escape corpus and resource limits for the execute_python sandbox (P0-SEC-2).

Every escape tries to reach os/builtins/the real __import__ and, if it gets
there, prints a marker built at runtime ('ESC' + 'APED'), so the source line a
traceback echoes back never contains the marker itself. A payload counts as
contained when the marker is absent and the sandbox reported a block or error.
"""

import pytest

from agent import tool_executor as te
from agent.tool_executor import execute_python

MARK = "ESC" + "APED"

# (id, payload). Each line ends by printing the marker plus something only an
# escape could obtain.
ESCAPES = [
    ("public_module_alias", "import fractions\nprint('ESC'+'APED', fractions.sys.modules['os'].getpid())"),
    ("private_module_alias", "import collections\nprint('ESC'+'APED', collections._sys.modules['os'].getpid())"),
    ("nested_public_alias", "import re\nprint('ESC'+'APED', re.enum.sys.modules['os'].getcwd())"),
    ("from_import_alias", "from fractions import sys\nprint('ESC'+'APED', sys.modules['os'].getpid())"),
    ("from_import_submodule", "from re import enum\nprint('ESC'+'APED', enum.sys.modules['os'].getpid())"),
    ("statistics_alias", "import statistics\nprint('ESC'+'APED', statistics.sys.modules['os'].getpid())"),
    ("generator_frame_walk",
     "def g():\n    yield gen.gi_frame.f_back.f_back.f_globals\ngen = g()\nG = next(gen)\n"
     "print('ESC'+'APED', G['_real_import']('os').getpid())"),
    ("coroutine_frame", "async def f():\n    pass\nc = f()\nprint('ESC'+'APED', c.cr_frame.f_globals)"),
    ("uncalled_str_format", "f = str.format\ndef g():\n    yield 1\nprint(f('{0.gi_code.co_name}', g()), 'ESC'+'APED')"),
    ("format_in_map", "print(list(map(str.format, ['{0.gi_frame}'], [1])), 'ESC'+'APED')"),
    ("formatter_vformat",
     "import string\ndef g():\n    yield 1\nprint(string.Formatter().vformat('{0.gi_frame}', [g()], {}), 'ESC'+'APED')"),
    ("builtins_open_alias",
     "import fractions\nb = fractions.sys.modules['builtins']\no = b.open\nprint('ESC'+'APED', o('/etc/hostname').read())"),
    ("dunder_import_name", "print('ESC'+'APED', __import__('os').getpid())"),
    ("subclasses_walk", "print('ESC'+'APED', ().__class__.__base__.__subclasses__())"),
    ("getattr_dunder", "print('ESC'+'APED', getattr(1, '__cl' + 'ass__'))"),
    ("getattr_alias", "g = getattr\nprint('ESC'+'APED', g(1, 'real'))"),
    ("dunder_in_format_string", "print('{0.__class__}'.format(1), 'ESC'+'APED')"),
    ("import_os", "import os\nprint('ESC'+'APED', os.getpid())"),
    ("import_importlib", "import importlib\nprint('ESC'+'APED', importlib.import_module('os'))"),
    ("relative_import", "from . import os\nprint('ESC'+'APED', os.getpid())"),
    ("eval_call", "print('ESC'+'APED', eval('1+1'))"),
    ("exec_call", "exec(\"print('ESC'+'APED')\")"),
    ("open_call", "print('ESC'+'APED', open('/etc/hostname').read())"),
    ("globals_builtins", "print('ESC'+'APED', globals()['__builtins__'])"),
    ("traceback_frame",
     "try:\n    1/0\nexcept Exception as e:\n    t = e.__traceback__\n    print('ESC'+'APED', t.tb_frame.f_globals)"),
    ("function_globals", "import random\nprint('ESC'+'APED', random.Random.seed.__globals__['_os'])"),
    ("module_loader", "import math\nprint('ESC'+'APED', math.__loader__)"),
    ("print_globals", "print('ESC'+'APED', print.__globals__['_sys'])"),
    ("type_call", "print('ESC'+'APED', type(1))"),
    ("breakpoint_call", "breakpoint()\nprint('ESC'+'APED')"),
    ("json_submodule", "import json\nprint('ESC'+'APED', json.decoder.re.enum.sys)"),
]


def test_escape_corpus_is_large_enough():
    assert len(ESCAPES) >= 20
    assert len({name for name, _ in ESCAPES}) == len(ESCAPES)


@pytest.mark.parametrize("name,payload", ESCAPES, ids=[n for n, _ in ESCAPES])
def test_escape_is_contained(name, payload):
    text = execute_python(payload)["text"]
    assert MARK not in text, f"{name} escaped: {text[:300]}"
    assert text.startswith(("Blocked:", "Runtime error:")), text[:300]


LEGIT = [
    ("statistics", "import statistics\nprint(statistics.mean([1, 2, 3, 4]))", "2.5"),
    ("math_alias", "import math as m\nprint(round(m.sqrt(2), 4))", "1.4142"),
    ("decimal", "from decimal import Decimal\nprint(Decimal('0.1') + Decimal('0.2'))", "0.3"),
    ("fractions", "from fractions import Fraction\nprint(Fraction(1, 3) + Fraction(1, 6))", "1/2"),
    ("counter", "from collections import Counter\nprint(Counter('aab').most_common(1))", "[('a', 2)]"),
    ("namedtuple_api",
     "from collections import namedtuple\nP = namedtuple('P', 'x y')\nprint(P(1, 2)._replace(y=5)._asdict())",
     "{'x': 1, 'y': 5}"),
    ("datetime", "import datetime\nprint((datetime.date(2026, 3, 1) - datetime.date(2026, 2, 1)).days)", "28"),
    ("json", "import json\nprint(json.dumps({'a': [1, 2]}))", '{"a": [1, 2]}'),
    ("regex", "import re\nprint(re.findall(r'\\d+', 'f 28.12 GHz, 3 dB'))", "['28', '12', '3']"),
    ("itertools", "import itertools\nprint(len(list(itertools.combinations(range(5), 2))))", "10"),
    ("functools", "import functools\nprint(functools.reduce(lambda a, b: a * b, range(1, 6)))", "120"),
    ("random_seeded", "import random\nrandom.seed(7)\nx = random.random()\nrandom.seed(7)\nprint(x == random.random())",
     "True"),
    ("class_def",
     "class Antenna:\n    def __init__(self, f):\n        self.f = f\n    def wl(self):\n        return 3e8 / self.f\n"
     "print(round(Antenna(28e9).wl() * 1000, 3))", "10.714"),
    ("fstring", "g = 3.5\nprint(f'gain {g:.1f} dBi')", "gain 3.5 dBi"),
    ("try_except", "try:\n    1 / 0\nexcept ZeroDivisionError as e:\n    print('caught', e)", "caught division by zero"),
    ("defaultdict", "from collections import defaultdict\nd = defaultdict(int)\nd['k'] += 2\nprint(dict(d))",
     "{'k': 2}"),
]


@pytest.mark.parametrize("name,code,expected", LEGIT, ids=[n for n, _, _ in LEGIT])
def test_legitimate_code_runs(name, code, expected):
    assert execute_python(code)["text"] == expected


def test_wall_clock_timeout(monkeypatch):
    monkeypatch.setattr(te, "_SANDBOX_TIMEOUT", 2)
    text = execute_python("while True:\n    pass")["text"]
    assert "timed out" in text


def test_memory_limit():
    text = execute_python("a = 10**4\nb = [0] * (a * a * a)\nprint('allocated')")["text"]
    assert text.startswith("Runtime error:") and "MemoryError" in text
    assert "allocated" not in text.split("\n")[-1]


def test_stdout_is_bounded():
    text = execute_python("for i in range(10**5):\n    print('x' * 100)")["text"]
    assert "output limit" in text
    assert len(text) < te._MAX_SANDBOX_OUTPUT + 3000


def test_stderr_is_bounded():
    text = execute_python("x = 'A' * 10000\nraise ValueError(x * 1000)")["text"]
    assert text.startswith("Runtime error:")
    assert len(text) < 2500


def test_huge_constant_power_is_rejected_without_evaluating_it():
    import time
    start = time.monotonic()
    assert te._validate_ast("x = 9**99999999").startswith("Exponentiation result too large")
    assert time.monotonic() - start < 0.5   # evaluating it in the server took seconds


def test_float_science_constants_are_allowed():
    assert execute_python("c = 3e8\nf = 28.12e9\nprint(round(c / f * 1000, 3))")["text"] == "10.669"


def test_process_and_file_limits_are_set():
    # Behaviour is untestable without an escape (fork/write need os), so pin the wiring.
    for limit in ("RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NPROC"):
        assert limit in te._SANDBOX_WRAPPER
