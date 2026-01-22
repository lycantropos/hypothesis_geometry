from typing import Any

from ground.hints import Empty
from hypothesis import given, strategies as st

from hypothesis_geometry.planar import empty_geometries
from tests.strategies import data_object_strategy
from tests.utils import context


def test_basic() -> None:
    result: st.SearchStrategy[Empty[Any]] = empty_geometries(context=context)

    assert isinstance(result, st.SearchStrategy)


@given(data_object_strategy)
def test_properties(data: st.DataObject) -> None:
    strategy: st.SearchStrategy[Empty[Any]] = empty_geometries(context=context)

    result = data.draw(strategy)

    assert isinstance(result, context.empty_cls)
