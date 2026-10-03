"""The full measurement (``perf`` marker) runs only when selected: ``make perf`` (``-m perf``).

The default run (``pytest -m "not docker"``) does not name ``perf``, so those tests are skipped
there; the smoke variant in `test_perf.py` is unmarked and runs everywhere.
"""

import pytest

PERF = "perf"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    markexpr: str = config.getoption("markexpr") or ""
    if PERF in markexpr:
        return
    skip = pytest.mark.skip(reason="full perf measurement: run with `make perf` (-m perf)")
    for item in items:
        if item.get_closest_marker(PERF) is not None:
            item.add_marker(skip)
