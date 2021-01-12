from typing import Tuple

from ground.hints import Coordinate
from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import Strategy
from hypothesis_geometry.planar import (Size,
                                        triangular_contours)
from tests import strategies
from tests.utils import (CoordinatesLimitsType,
                         are_vertices_non_convex,
                         are_vertices_strict,
                         contour_has_coordinates_in_range,
                         contour_has_coordinates_types,
                         has_valid_size,
                         is_contour,
                         is_contour_counterclockwise,
                         is_contour_non_self_intersecting)


@given(strategies.coordinates_strategies)
def test_basic(coordinates: Strategy[Coordinate]) -> None:
    result = triangular_contours(coordinates)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.coordinates_strategy_with_limit_and_type_pairs)
def test_properties(data: DataObject,
                    coordinates_limits_type_pair: Tuple[CoordinatesLimitsType,
                                                        CoordinatesLimitsType]
                    ) -> None:
    (x_coordinates_limits_type,
     y_coordinates_limits_type) = coordinates_limits_type_pair
    ((x_coordinates, (min_x_value, max_x_value)),
     x_type) = x_coordinates_limits_type
    ((y_coordinates, (min_y_value, max_y_value)),
     y_type) = y_coordinates_limits_type

    strategy = triangular_contours(x_coordinates, y_coordinates)

    result = data.draw(strategy)

    assert is_contour(result)
    assert has_valid_size(result.vertices,
                          min_size=Size.TRIANGULAR_CONTOUR,
                          max_size=Size.TRIANGULAR_CONTOUR)
    assert contour_has_coordinates_types(result,
                                         x_type=x_type,
                                         y_type=y_type)
    assert contour_has_coordinates_in_range(result,
                                            min_x_value=min_x_value,
                                            max_x_value=max_x_value,
                                            min_y_value=min_y_value,
                                            max_y_value=max_y_value)
    assert are_vertices_strict(result.vertices)
    assert not are_vertices_non_convex(result.vertices)
    assert is_contour_non_self_intersecting(result)
    assert is_contour_counterclockwise(result)


@given(strategies.data,
       strategies.coordinates_strategies_with_limits_and_types)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: CoordinatesLimitsType
                          ) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = triangular_contours(coordinates)

    result = data.draw(strategy)

    assert is_contour(result)
    assert has_valid_size(result.vertices,
                          min_size=Size.TRIANGULAR_CONTOUR,
                          max_size=Size.TRIANGULAR_CONTOUR)
    assert contour_has_coordinates_types(result,
                                         x_type=type_,
                                         y_type=type_)
    assert contour_has_coordinates_in_range(result,
                                            min_x_value=min_value,
                                            max_x_value=max_value,
                                            min_y_value=min_value,
                                            max_y_value=max_value)
    assert are_vertices_strict(result.vertices)
    assert not are_vertices_non_convex(result.vertices)
    assert is_contour_non_self_intersecting(result)
    assert is_contour_counterclockwise(result)
