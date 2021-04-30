from typing import Tuple

import pytest
from ground.hints import Scalar
from hypothesis import given
from hypothesis.errors import HypothesisWarning
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import Strategy
from hypothesis_geometry.planar import polygons
from tests import strategies
from tests.utils import (ScalarsLimitsType,
                         SizesPair,
                         contours_do_not_cross_or_overlap,
                         is_contour_counterclockwise,
                         is_contour_non_self_intersecting,
                         is_polygon,
                         is_polygon_strict,
                         polygon_has_coordinates_in_range,
                         polygon_has_coordinates_types,
                         polygon_has_valid_sizes)


@given(strategies.scalars_strategies,
       strategies.concave_contours_sizes_pairs,
       strategies.polygon_holes_sizes_pairs,
       strategies.convex_contours_sizes_pairs)
def test_basic(scalars: Strategy[Scalar],
               sizes_pair: SizesPair,
               holes_sizes_pair: SizesPair,
               hole_sizes_pair: SizesPair) -> None:
    min_size, max_size = sizes_pair
    min_holes_size, max_holes_size = holes_sizes_pair
    min_hole_size, max_hole_size = hole_sizes_pair

    result = polygons(scalars,
                      min_size=min_size,
                      max_size=max_size,
                      min_holes_size=min_holes_size,
                      max_holes_size=max_holes_size,
                      min_hole_size=min_hole_size,
                      max_hole_size=max_hole_size)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.scalars_strategy_with_limit_and_type_pairs,
       strategies.concave_contours_sizes_pairs,
       strategies.polygon_holes_sizes_pairs,
       strategies.convex_contours_sizes_pairs)
def test_properties(data: DataObject,
                    coordinates_limits_type_pair: Tuple[ScalarsLimitsType,
                                                        ScalarsLimitsType],
                    sizes_pair: SizesPair,
                    holes_sizes_pair: SizesPair,
                    hole_sizes_pair: SizesPair) -> None:
    (x_coordinates_limits_type,
     y_coordinates_limits_type) = coordinates_limits_type_pair
    ((x_coordinates, (min_x_value, max_x_value)),
     x_type) = x_coordinates_limits_type
    ((y_coordinates, (min_y_value, max_y_value)),
     y_type) = y_coordinates_limits_type
    min_size, max_size = sizes_pair
    min_holes_size, max_holes_size = holes_sizes_pair
    min_hole_size, max_hole_size = hole_sizes_pair

    strategy = polygons(x_coordinates, y_coordinates,
                        min_size=min_size,
                        max_size=max_size,
                        min_holes_size=min_holes_size,
                        max_holes_size=max_holes_size,
                        min_hole_size=min_hole_size,
                        max_hole_size=max_hole_size)

    result = data.draw(strategy)

    assert is_polygon(result)
    assert polygon_has_valid_sizes(result,
                                   min_size=min_size,
                                   max_size=max_size,
                                   min_holes_size=min_holes_size,
                                   max_holes_size=max_holes_size,
                                   min_hole_size=min_hole_size,
                                   max_hole_size=max_hole_size)
    assert polygon_has_coordinates_types(result,
                                         x_type=x_type,
                                         y_type=y_type)
    assert polygon_has_coordinates_in_range(result,
                                            min_x_value=min_x_value,
                                            max_x_value=max_x_value,
                                            min_y_value=min_y_value,
                                            max_y_value=max_y_value)
    assert is_polygon_strict(result)
    assert is_contour_non_self_intersecting(result.border)
    assert all(is_contour_non_self_intersecting(hole) for hole in result.holes)
    assert contours_do_not_cross_or_overlap(result.holes)
    assert is_contour_counterclockwise(result.border)
    assert all(not is_contour_counterclockwise(hole) for hole in result.holes)


@given(strategies.data,
       strategies.scalars_strategies_with_limits_and_types,
       strategies.concave_contours_sizes_pairs,
       strategies.polygon_holes_sizes_pairs,
       strategies.convex_contours_sizes_pairs)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: ScalarsLimitsType,
                          sizes_pair: SizesPair,
                          holes_sizes_pair: SizesPair,
                          hole_sizes_pair: SizesPair) -> None:
    (scalars, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair
    min_holes_size, max_holes_size = holes_sizes_pair
    min_hole_size, max_hole_size = hole_sizes_pair

    strategy = polygons(scalars,
                        min_size=min_size,
                        max_size=max_size,
                        min_holes_size=min_holes_size,
                        max_holes_size=max_holes_size,
                        min_hole_size=min_hole_size,
                        max_hole_size=max_hole_size)

    result = data.draw(strategy)

    assert is_polygon(result)

    assert polygon_has_valid_sizes(result,
                                   min_size=min_size,
                                   max_size=max_size,
                                   min_holes_size=min_holes_size,
                                   max_holes_size=max_holes_size,
                                   min_hole_size=min_hole_size,
                                   max_hole_size=max_hole_size)
    assert polygon_has_coordinates_types(result,
                                         x_type=type_,
                                         y_type=type_)
    assert polygon_has_coordinates_in_range(result,
                                            min_x_value=min_value,
                                            max_x_value=max_value,
                                            min_y_value=min_value,
                                            max_y_value=max_value)
    assert is_polygon_strict(result)
    assert is_contour_non_self_intersecting(result.border)
    assert all(is_contour_non_self_intersecting(hole) for hole in result.holes)
    assert contours_do_not_cross_or_overlap(result.holes)
    assert is_contour_counterclockwise(result.border)
    assert all(not is_contour_counterclockwise(hole) for hole in result.holes)


@given(strategies.scalars_strategies,
       strategies.invalid_convex_contours_sizes_pairs)
def test_invalid_border_sizes(scalars: Strategy[Scalar],
                              invalid_sizes_pair: SizesPair) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        polygons(scalars,
                 min_size=min_size,
                 max_size=max_size)


@given(strategies.scalars_strategies,
       strategies.invalid_polygon_holes_sizes_pairs)
def test_invalid_holes_list_sizes(scalars: Strategy[Scalar],
                                  invalid_sizes_pair: SizesPair
                                  ) -> None:
    min_holes_size, max_holes_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        polygons(scalars,
                 min_holes_size=min_holes_size,
                 max_holes_size=max_holes_size)


@given(strategies.scalars_strategies,
       strategies.invalid_convex_contours_sizes_pairs)
def test_invalid_holes_sizes(scalars: Strategy[Scalar],
                             invalid_sizes_pair: SizesPair
                             ) -> None:
    min_hole_size, max_hole_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        polygons(scalars,
                 min_hole_size=min_hole_size,
                 max_hole_size=max_hole_size)


@given(strategies.scalars_strategies,
       strategies.non_valid_convex_contours_sizes_pairs)
def test_non_valid_border_sizes(scalars: Strategy[Scalar],
                                non_valid_sizes_pair: SizesPair) -> None:
    min_size, max_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        polygons(scalars,
                 min_size=min_size,
                 max_size=max_size)

    assert len(warnings) == 1


@given(strategies.scalars_strategies,
       strategies.non_valid_convex_contours_sizes_pairs)
def test_non_valid_holes_sizes(scalars: Strategy[Scalar],
                               non_valid_sizes_pair: SizesPair) -> None:
    min_size, max_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        polygons(scalars,
                 min_hole_size=min_size,
                 max_hole_size=max_size)

    assert len(warnings) == 1
