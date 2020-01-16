from typing import (Tuple,
                    Type)

from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from hypothesis_geometry.planar import points
from tests import strategies
from tests.utils import Limits


@given(strategies.coordinates_strategies)
def test_basic(coordinates: Strategy[Coordinate]) -> None:
    result = points(coordinates)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.coordinates_strategies_with_limits_and_types)
def test_properties(
        data: DataObject,
        coordinates_with_limits_and_type: Tuple[Tuple[Strategy[Coordinate],
                                                      Limits],
                                                Type[Coordinate]]) -> None:
    coordinates_with_limits, type_ = coordinates_with_limits_and_type
    coordinates, (min_value, max_value) = coordinates_with_limits

    strategy = points(coordinates)

    result = data.draw(strategy)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(coordinate, type_) for coordinate in result)
    assert all(min_value <= coordinate
               and (max_value is None or coordinate <= max_value)
               for coordinate in result)
