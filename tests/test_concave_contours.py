from typing import Tuple

from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.core.contracts import (
    is_contour_non_convex,
    is_contour_strict,
    is_non_self_intersecting_contour)
from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from hypothesis_geometry.planar import concave_contours
from tests import strategies
from tests.utils import (CoordinatesLimitsType,
                         SizesPair,
                         contour_has_valid_size,
                         point_has_coordinates_in_range,
                         point_has_valid_size)


@given(strategies.coordinates_strategies_with_sizes_pairs)
def test_basic(coordinates_with_sizes_pair: Tuple[Strategy[Coordinate],
                                                  SizesPair]) -> None:
    coordinates, (min_size, max_size) = coordinates_with_sizes_pair

    result = concave_contours(coordinates,
                              min_size=min_size,
                              max_size=max_size)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.coordinates_strategies_limits_types_with_sizes_pairs)
def test_properties(
        data: DataObject,
        coordinates_limits_type_with_sizes_pair: Tuple[CoordinatesLimitsType,
                                                       SizesPair]) -> None:
    (coordinates_limits_type,
     (min_size, max_size)) = coordinates_limits_type_with_sizes_pair
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = concave_contours(coordinates,
                                min_size=min_size,
                                max_size=max_size)

    result = data.draw(strategy)

    assert isinstance(result, list)
    assert contour_has_valid_size(result,
                                  min_size=min_size,
                                  max_size=max_size)
    assert all(isinstance(vertex, tuple) for vertex in result)
    assert all(point_has_valid_size(vertex) for vertex in result)
    assert all(all(isinstance(coordinate, type_)
                   for coordinate in vertex)
               for vertex in result)
    assert all(point_has_coordinates_in_range(vertex,
                                              min_value=min_value,
                                              max_value=max_value)
               for vertex in result)
    assert is_contour_strict(result)
    assert is_contour_non_convex(result)
    assert is_non_self_intersecting_contour(result)
