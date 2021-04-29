from typing import Tuple

from ground.hints import Scalar
from hypothesis import given
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import Strategy
from hypothesis_geometry.planar import rectangular_contours
from tests import strategies
from tests.utils import (ScalarsLimitsType,
                         are_vertices_non_convex,
                         contour_has_coordinates_in_range,
                         contour_has_coordinates_types,
                         contour_has_valid_sizes,
                         is_contour,
                         is_contour_counterclockwise,
                         is_contour_non_self_intersecting,
                         is_contour_strict)


@given(strategies.scalars_strategies)
def test_basic(scalars: Strategy[Scalar]) -> None:
    result = rectangular_contours(scalars)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.scalars_strategy_with_limit_and_type_pairs)
def test_properties(data: DataObject,
                    coordinates_limits_type_pair: Tuple[ScalarsLimitsType,
                                                        ScalarsLimitsType]
                    ) -> None:
    (x_coordinates_limits_type,
     y_coordinates_limits_type) = coordinates_limits_type_pair
    ((x_coordinates, (min_x_value, max_x_value)),
     x_type) = x_coordinates_limits_type
    ((y_coordinates, (min_y_value, max_y_value)),
     y_type) = y_coordinates_limits_type

    strategy = rectangular_contours(x_coordinates, y_coordinates)

    result = data.draw(strategy)

    assert is_contour(result)
    assert contour_has_valid_sizes(result,
                                   min_size=4,
                                   max_size=4)
    assert contour_has_coordinates_types(result,
                                         x_type=x_type,
                                         y_type=y_type)
    assert contour_has_coordinates_in_range(result,
                                            min_x_value=min_x_value,
                                            max_x_value=max_x_value,
                                            min_y_value=min_y_value,
                                            max_y_value=max_y_value)
    assert is_contour_strict(result)
    assert not are_vertices_non_convex(result.vertices)
    assert is_contour_non_self_intersecting(result)
    assert is_contour_counterclockwise(result)
    assert result.vertices[0] == min(result.vertices)


@given(strategies.data,
       strategies.scalars_strategies_with_limits_and_types)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: ScalarsLimitsType
                          ) -> None:
    (scalars, (min_value, max_value)), type_ = coordinates_limits_type

    strategy = rectangular_contours(scalars)

    result = data.draw(strategy)

    assert is_contour(result)
    assert contour_has_valid_sizes(result,
                                   min_size=4,
                                   max_size=4)
    assert contour_has_coordinates_types(result,
                                         x_type=type_,
                                         y_type=type_)
    assert contour_has_coordinates_in_range(result,
                                            min_x_value=min_value,
                                            max_x_value=max_value,
                                            min_y_value=min_value,
                                            max_y_value=max_value)
    assert is_contour_strict(result)
    assert not are_vertices_non_convex(result.vertices)
    assert is_contour_non_self_intersecting(result)
    assert is_contour_counterclockwise(result)
    assert result.vertices[0] == min(result.vertices)
