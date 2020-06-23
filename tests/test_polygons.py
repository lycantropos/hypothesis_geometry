from typing import Tuple

import pytest
from hypothesis import given
from hypothesis.errors import HypothesisWarning
from hypothesis.strategies import DataObject

from hypothesis_geometry.core.contracts import is_contour_strict
from hypothesis_geometry.hints import (Coordinate,
                                       Strategy)
from hypothesis_geometry.planar import polygons
from tests import strategies
from tests.utils import (CoordinatesLimitsType,
                         SizesPair,
                         contours_do_not_cross_or_overlap,
                         is_counterclockwise_contour,
                         is_multicontour_strict,
                         is_non_self_intersecting_contour,
                         is_polygon,
                         polygon_has_coordinates_in_range,
                         polygon_has_coordinates_types,
                         polygon_has_valid_sizes)


@given(strategies.coordinates_strategies,
       strategies.concave_contours_sizes_pairs,
       strategies.multicontours_sizes_pairs,
       strategies.convex_contours_sizes_pairs)
def test_basic(coordinates: Strategy[Coordinate],
               sizes_pair: SizesPair,
               holes_list_sizes_pair: SizesPair,
               holes_sizes_pair: SizesPair) -> None:
    min_size, max_size = sizes_pair
    min_holes_size, max_holes_size = holes_list_sizes_pair
    min_hole_size, max_hole_size = holes_sizes_pair

    result = polygons(coordinates,
                      min_size=min_size,
                      max_size=max_size,
                      min_holes_size=min_holes_size,
                      max_holes_size=max_holes_size,
                      min_hole_size=min_hole_size,
                      max_hole_size=max_hole_size)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.coordinates_strategy_with_limit_and_type_pairs,
       strategies.concave_contours_sizes_pairs,
       strategies.multicontours_sizes_pairs,
       strategies.convex_contours_sizes_pairs)
def test_properties(data: DataObject,
                    coordinates_limits_type_pair: Tuple[CoordinatesLimitsType,
                                                        CoordinatesLimitsType],
                    sizes_pair: SizesPair,
                    holes_list_sizes_pair: SizesPair,
                    holes_sizes_pair: SizesPair) -> None:
    (x_coordinates_limits_type,
     y_coordinates_limits_type) = coordinates_limits_type_pair
    ((x_coordinates, (min_x_value, max_x_value)),
     x_type) = x_coordinates_limits_type
    ((y_coordinates, (min_y_value, max_y_value)),
     y_type) = y_coordinates_limits_type
    min_size, max_size = sizes_pair
    min_holes_size, max_holes_size = holes_list_sizes_pair
    min_hole_size, max_hole_size = holes_sizes_pair

    strategy = polygons(x_coordinates, y_coordinates,
                        min_size=min_size,
                        max_size=max_size,
                        min_holes_size=min_holes_size,
                        max_holes_size=max_holes_size,
                        min_hole_size=min_hole_size,
                        max_hole_size=max_hole_size)

    result = data.draw(strategy)

    assert is_polygon(result)

    border, holes = result
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
    assert is_contour_strict(border)
    assert is_multicontour_strict(holes)
    assert is_non_self_intersecting_contour(border)
    assert all(is_non_self_intersecting_contour(hole)
               for hole in holes)
    assert contours_do_not_cross_or_overlap(holes)
    assert is_counterclockwise_contour(border)
    assert all(not is_counterclockwise_contour(hole)
               for hole in holes)


@given(strategies.data,
       strategies.coordinates_strategies_with_limits_and_types,
       strategies.concave_contours_sizes_pairs,
       strategies.multicontours_sizes_pairs,
       strategies.convex_contours_sizes_pairs)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: CoordinatesLimitsType,
                          sizes_pair: SizesPair,
                          holes_list_sizes_pair: SizesPair,
                          holes_sizes_pair: SizesPair) -> None:
    (coordinates, (min_value, max_value)), type_ = coordinates_limits_type
    min_size, max_size = sizes_pair
    min_holes_size, max_holes_size = holes_list_sizes_pair
    min_hole_size, max_hole_size = holes_sizes_pair

    strategy = polygons(coordinates,
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
    border, holes = result
    assert polygon_has_coordinates_types(result,
                                         x_type=type_,
                                         y_type=type_)
    assert polygon_has_coordinates_in_range(result,
                                            min_x_value=min_value,
                                            max_x_value=max_value,
                                            min_y_value=min_value,
                                            max_y_value=max_value)
    assert is_contour_strict(border)
    assert is_multicontour_strict(holes)
    assert is_non_self_intersecting_contour(border)
    assert all(is_non_self_intersecting_contour(hole)
               for hole in holes)
    assert contours_do_not_cross_or_overlap(holes)
    assert is_counterclockwise_contour(border)
    assert all(not is_counterclockwise_contour(hole)
               for hole in holes)


@given(strategies.coordinates_strategies,
       strategies.invalid_convex_contours_sizes_pairs)
def test_invalid_border_sizes(coordinates: Strategy[Coordinate],
                              invalid_sizes_pair: SizesPair) -> None:
    min_size, max_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        polygons(coordinates,
                 min_size=min_size,
                 max_size=max_size)


@given(strategies.coordinates_strategies,
       strategies.invalid_multicontours_sizes_pairs)
def test_invalid_holes_list_sizes(coordinates: Strategy[Coordinate],
                                  invalid_sizes_pair: SizesPair
                                  ) -> None:
    min_holes_size, max_holes_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        polygons(coordinates,
                 min_holes_size=min_holes_size,
                 max_holes_size=max_holes_size)


@given(strategies.coordinates_strategies,
       strategies.invalid_convex_contours_sizes_pairs)
def test_invalid_holes_sizes(coordinates: Strategy[Coordinate],
                             invalid_sizes_pair: SizesPair
                             ) -> None:
    min_hole_size, max_hole_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        polygons(coordinates,
                 min_hole_size=min_hole_size,
                 max_hole_size=max_hole_size)


@given(strategies.coordinates_strategies,
       strategies.non_valid_convex_contours_sizes_pairs)
def test_non_valid_border_sizes(coordinates: Strategy[Coordinate],
                                non_valid_sizes_pair: SizesPair) -> None:
    min_size, max_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        polygons(coordinates,
                 min_size=min_size,
                 max_size=max_size)

    assert len(warnings) == 1


@given(strategies.coordinates_strategies,
       strategies.non_valid_convex_contours_sizes_pairs)
def test_non_valid_holes_sizes(coordinates: Strategy[Coordinate],
                               non_valid_sizes_pair: SizesPair) -> None:
    min_size, max_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        polygons(coordinates,
                 min_hole_size=min_size,
                 max_hole_size=max_size)

    assert len(warnings) == 1
