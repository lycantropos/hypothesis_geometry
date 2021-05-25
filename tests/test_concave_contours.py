from typing import Tuple

import pytest
from ground.hints import Scalar
from hypothesis import given
from hypothesis.errors import HypothesisWarning
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import Strategy
from hypothesis_geometry.planar import concave_contours
from tests import strategies
from tests.utils import (ScalarsLimitsType,
                         SizesPair,
                         are_vertices_non_convex,
                         are_vertices_strict,
                         contour_has_coordinates_in_range,
                         contour_has_coordinates_types,
                         contour_has_valid_sizes,
                         is_contour,
                         is_contour_counterclockwise,
                         is_contour_non_self_intersecting,
                         is_contour_strict)


@given(strategies.scalars_strategies,
       strategies.concave_contours_sizes_pairs)
def test_basic(coordinates: Strategy[Scalar],
               sizes_pair: SizesPair) -> None:
    min_size, max_size = sizes_pair

    result = concave_contours(coordinates,
                              min_size=min_size,
                              max_size=max_size)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.scalars_strategy_with_limit_and_type_pairs,
       strategies.concave_contours_sizes_pairs)
def test_properties(data: DataObject,
                    coordinates_limits_type_pair: Tuple[ScalarsLimitsType,
                                                        ScalarsLimitsType],
                    sizes_pair: SizesPair) -> None:
    (x_coordinates_limits_type,
     y_coordinates_limits_type) = coordinates_limits_type_pair
    ((x_coordinates, (min_x_value, max_x_value)),
     x_type) = x_coordinates_limits_type
    ((y_coordinates, (min_y_value, max_y_value)),
     y_type) = y_coordinates_limits_type
    min_size, max_size = sizes_pair

    strategy = concave_contours(x_coordinates, y_coordinates,
                                min_size=min_size,
                                max_size=max_size)

    result = data.draw(strategy)

    assert is_contour(result)
    assert contour_has_valid_sizes(result,
                                   min_size=min_size,
                                   max_size=max_size)
    assert contour_has_coordinates_types(result,
                                         x_type=x_type,
                                         y_type=y_type)
    assert contour_has_coordinates_in_range(result,
                                            min_x_value=min_x_value,
                                            max_x_value=max_x_value,
                                            min_y_value=min_y_value,
                                            max_y_value=max_y_value)
    assert is_contour_strict(result)
    assert are_vertices_non_convex(result.vertices)
    assert is_contour_non_self_intersecting(result)
    assert is_contour_counterclockwise(result)


@given(strategies.data,
       strategies.scalars_strategies_with_limits_and_types,
       strategies.concave_contours_sizes_pairs)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: ScalarsLimitsType,
                          sizes_pair: SizesPair) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair

    strategy = concave_contours(coordinates,
                                min_size=min_size,
                                max_size=max_size)

    result = data.draw(strategy)

    assert is_contour(result)
    assert contour_has_valid_sizes(result,
                                   min_size=min_size,
                                   max_size=max_size)
    assert contour_has_coordinates_types(result,
                                         x_type=type_,
                                         y_type=type_)
    assert contour_has_coordinates_in_range(result,
                                            min_x_value=min_value,
                                            max_x_value=max_value,
                                            min_y_value=min_value,
                                            max_y_value=max_value)
    assert are_vertices_strict(result.vertices)
    assert are_vertices_non_convex(result.vertices)
    assert is_contour_non_self_intersecting(result)
    assert is_contour_counterclockwise(result)


@given(strategies.scalars_strategies,
       strategies.invalid_concave_contours_sizes_pairs)
def test_invalid_sizes(coordinates: Strategy[Scalar],
                       invalid_sizes_pair: SizesPair) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        concave_contours(coordinates,
                         min_size=min_size,
                         max_size=max_size)


@given(strategies.scalars_strategies,
       strategies.non_valid_concave_contours_sizes_pairs)
def test_non_valid_sizes(coordinates: Strategy[Scalar],
                         non_valid_sizes_pair: SizesPair) -> None:
    min_size, max_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        concave_contours(coordinates,
                         min_size=min_size,
                         max_size=max_size)

    assert len(warnings) == 1
