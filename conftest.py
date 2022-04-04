from typing import List

import pytest


def pytest_load_initial_conftests(early_config: pytest.Config,
                                  parser: pytest.Parser,
                                  args: List[str]) -> None:
    # PizzaCutter Template can add here additional pytest args
    print('shit')
    raise ValueError()
