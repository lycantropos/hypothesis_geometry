from typing import Tuple

import pytest
from ground.hints import Coordinate
from hypothesis import given
from hypothesis.errors import HypothesisWarning
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import Strategy
from hypothesis_geometry.planar import multicontours
from tests import strategies
from tests.utils import (CoordinatesLimitsType,
                         SizesPair,
                         contours_do_not_cross_or_overlap,
                         has_valid_size,
                         is_contour_counterclockwise,
                         is_contour_non_self_intersecting,
                         is_multicontour,
                         is_multicontour_strict,
                         multicontour_has_coordinates_in_range,
                         multicontour_has_coordinates_types,
                         multicontour_has_valid_sizes)


@given(strategies.coordinates_strategies,
       strategies.multicontours_sizes_pairs,
       strategies.convex_contours_sizes_pairs)
def test_basic(coordinates: Strategy[Coordinate],
               sizes_pair: SizesPair,
               contours_sizes_pair: SizesPair) -> None:
    min_size, max_size = sizes_pair
    min_contour_size, max_contour_size = contours_sizes_pair

    result = multicontours(coordinates,
                           min_size=min_size,
                           max_size=max_size,
                           min_contour_size=min_contour_size,
                           max_contour_size=max_contour_size)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.coordinates_strategy_with_limit_and_type_pairs,
       strategies.multicontours_sizes_pairs,
       strategies.concave_contours_sizes_pairs)
def test_properties(data: DataObject,
                    coordinates_limits_type_pair: Tuple[CoordinatesLimitsType,
                                                        CoordinatesLimitsType],
                    sizes_pair: SizesPair,
                    contours_sizes_pair: SizesPair) -> None:
    (x_coordinates_limits_type,
     y_coordinates_limits_type) = coordinates_limits_type_pair
    ((x_coordinates, (min_x_value, max_x_value)),
     x_type) = x_coordinates_limits_type
    ((y_coordinates, (min_y_value, max_y_value)),
     y_type) = y_coordinates_limits_type
    min_size, max_size = sizes_pair
    min_contour_size, max_contour_size = contours_sizes_pair

    strategy = multicontours(x_coordinates, y_coordinates,
                             min_size=min_size,
                             max_size=max_size,
                             min_contour_size=min_contour_size,
                             max_contour_size=max_contour_size)

    result = data.draw(strategy)

    assert is_multicontour(result)
    assert multicontour_has_valid_sizes(result,
                                        min_size=min_size,
                                        max_size=max_size,
                                        min_contour_size=min_contour_size,
                                        max_contour_size=max_contour_size)
    assert multicontour_has_coordinates_types(result,
                                              x_type=x_type,
                                              y_type=y_type)
    assert multicontour_has_coordinates_in_range(result,
                                                 min_x_value=min_x_value,
                                                 max_x_value=max_x_value,
                                                 min_y_value=min_y_value,
                                                 max_y_value=max_y_value)
    assert is_multicontour_strict(result)
    assert all(is_contour_non_self_intersecting(contour)
               for contour in result)
    assert contours_do_not_cross_or_overlap(result)
    assert all(is_contour_counterclockwise(contour)
               for contour in result)


@given(strategies.data,
       strategies.coordinates_strategies_with_limits_and_types,
       strategies.multicontours_sizes_pairs,
       strategies.convex_contours_sizes_pairs)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: CoordinatesLimitsType,
                          sizes_pair: SizesPair,
                          contours_sizes_pair: SizesPair) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair
    min_contour_size, max_contour_size = contours_sizes_pair

    strategy = multicontours(coordinates,
                             min_size=min_size,
                             max_size=max_size,
                             min_contour_size=min_contour_size,
                             max_contour_size=max_contour_size)

    result = data.draw(strategy)

    assert is_multicontour(result)
    assert has_valid_size(result,
                          min_size=min_size,
                          max_size=max_size)
    assert multicontour_has_valid_sizes(result,
                                        min_size=min_size,
                                        max_size=max_size,
                                        min_contour_size=min_contour_size,
                                        max_contour_size=max_contour_size)
    assert multicontour_has_coordinates_types(result,
                                              x_type=type_,
                                              y_type=type_)
    assert multicontour_has_coordinates_in_range(result,
                                                 min_x_value=min_value,
                                                 max_x_value=max_value,
                                                 min_y_value=min_value,
                                                 max_y_value=max_value)
    assert is_multicontour_strict(result)
    assert all(is_contour_non_self_intersecting(contour)
               for contour in result)
    assert contours_do_not_cross_or_overlap(result)
    assert all(is_contour_counterclockwise(contour)
               for contour in result)


@given(strategies.coordinates_strategies,
       strategies.invalid_multicontours_sizes_pairs)
def test_invalid_sizes(coordinates: Strategy[Coordinate],
                       invalid_sizes_pair: SizesPair) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        multicontours(coordinates,
                      min_size=min_size,
                      max_size=max_size)


@given(strategies.coordinates_strategies,
       strategies.invalid_convex_contours_sizes_pairs)
def test_invalid_contours_sizes(coordinates: Strategy[Coordinate],
                                invalid_contours_sizes_pair: SizesPair
                                ) -> None:
    min_contour_size, max_contour_size = invalid_contours_sizes_pair

    with pytest.raises(ValueError):
        multicontours(coordinates,
                      min_contour_size=min_contour_size,
                      max_contour_size=max_contour_size)


@given(strategies.coordinates_strategies,
       strategies.non_valid_convex_contours_sizes_pairs)
def test_non_valid_contours_sizes(coordinates: Strategy[Coordinate],
                                  non_valid_sizes_pair: SizesPair) -> None:
    min_size, max_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        multicontours(coordinates,
                      min_contour_size=min_size,
                      max_contour_size=max_size)

    assert len(warnings) == 1
