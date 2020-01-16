from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from hypothesis_geometry.planar import points
from tests import strategies
from tests.utils import (CoordinatesLimitsType,
                         point_has_coordinates_in_range,
                         point_has_valid_size)


@given(strategies.coordinates_strategies)
def test_basic(coordinates: Strategy[Coordinate]) -> None:
    result = points(coordinates)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.coordinates_strategies_with_limits_and_types)
def test_properties(data: DataObject,
                    coordinates_limits_type: CoordinatesLimitsType) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = points(coordinates)

    result = data.draw(strategy)

    assert isinstance(result, tuple)
    assert point_has_valid_size(result)
    assert all(isinstance(coordinate, type_) for coordinate in result)
    assert point_has_coordinates_in_range(result,
                                          min_value=min_value,
                                          max_value=max_value)
