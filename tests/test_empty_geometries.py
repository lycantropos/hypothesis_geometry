from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import Strategy
from hypothesis_geometry.planar import empty_geometries
from tests import strategies
from tests.utils import is_empty


def test_basic() -> None:
    result = empty_geometries()

    assert isinstance(result, Strategy)


@given(strategies.data)
def test_properties(data: DataObject) -> None:
    strategy = empty_geometries()

    result = data.draw(strategy)

    assert is_empty(result)
