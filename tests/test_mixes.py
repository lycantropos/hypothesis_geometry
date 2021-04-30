from typing import Tuple

import pytest
from ground.hints import Scalar
from hypothesis import given
from hypothesis.errors import HypothesisWarning
from hypothesis.strategies import DataObject

from hypothesis_geometry.hints import Strategy
from hypothesis_geometry.planar import mixes
from tests import strategies
from tests.utils import (ScalarsLimitsType,
                         SizesPair,
                         is_mix,
                         mix_has_coordinates_in_range,
                         mix_has_coordinates_types,
                         mix_has_valid_sizes,
                         mix_segments_do_not_cross_or_overlap)


@given(strategies.scalars_strategies,
       strategies.multipoints_sizes_pairs,
       strategies.multisegments_sizes_pairs,
       strategies.multipolygons_sizes_pairs,
       strategies.concave_contours_sizes_pairs,
       strategies.multicontours_sizes_pairs,
       strategies.convex_contours_sizes_pairs)
def test_basic(scalars: Strategy[Scalar],
               multipoints_sizes_pair: SizesPair,
               multisegments_sizes_pair: SizesPair,
               multipolygons_sizes_pair: SizesPair,
               multipolygons_border_sizes_pair: SizesPair,
               multipolygons_holes_list_sizes_pair: SizesPair,
               multipolygons_holes_sizes_pair: SizesPair) -> None:
    min_multipoint_size, max_multipoint_size = multipoints_sizes_pair
    min_multisegment_size, max_multisegment_size = multisegments_sizes_pair
    min_multipolygon_size, max_multipolygon_size = multipolygons_sizes_pair
    (min_multipolygon_border_size,
     max_multipolygon_border_size) = multipolygons_border_sizes_pair
    (min_multipolygon_holes_size,
     max_multipolygon_holes_size) = multipolygons_holes_list_sizes_pair
    (min_multipolygon_hole_size,
     max_multipolygon_hole_size) = multipolygons_holes_sizes_pair

    result = mixes(scalars, min_points_size=min_multipoint_size,
                   max_points_size=max_multipoint_size,
                   min_segments_size=min_multisegment_size,
                   max_segments_size=max_multisegment_size,
                   min_polygons_size=min_multipolygon_size,
                   max_polygons_size=max_multipolygon_size,
                   min_polygon_border_size=min_multipolygon_border_size,
                   max_polygon_border_size=max_multipolygon_border_size,
                   min_polygon_holes_size=min_multipolygon_holes_size,
                   max_polygon_holes_size=max_multipolygon_holes_size,
                   min_polygon_hole_size=min_multipolygon_hole_size,
                   max_polygon_hole_size=max_multipolygon_hole_size)

    assert isinstance(result, Strategy)


@given(strategies.data,
       strategies.scalars_strategy_with_limit_and_type_pairs,
       strategies.multipoints_sizes_pairs,
       strategies.multisegments_sizes_pairs,
       strategies.multipolygons_sizes_pairs,
       strategies.concave_contours_sizes_pairs,
       strategies.multicontours_sizes_pairs,
       strategies.convex_contours_sizes_pairs)
def test_properties(data: DataObject,
                    coordinates_limits_type_pair: Tuple[ScalarsLimitsType,
                                                        ScalarsLimitsType],
                    points_sizes_pair: SizesPair,
                    segments_sizes_pair: SizesPair,
                    polygons_sizes_pair: SizesPair,
                    polygon_border_sizes_pair: SizesPair,
                    polygon_holes_sizes_pair: SizesPair,
                    polygon_hole_sizes_pair: SizesPair) -> None:
    (x_coordinates_limits_type,
     y_coordinates_limits_type) = coordinates_limits_type_pair
    ((x_coordinates, (min_x_value, max_x_value)),
     x_type) = x_coordinates_limits_type
    ((y_coordinates, (min_y_value, max_y_value)),
     y_type) = y_coordinates_limits_type
    min_points_size, max_points_size = points_sizes_pair
    min_segments_size, max_segments_size = segments_sizes_pair
    min_polygons_size, max_polygons_size = polygons_sizes_pair
    (min_polygon_border_size,
     max_polygon_border_size) = polygon_border_sizes_pair
    min_polygon_holes_size, max_polygon_holes_size = polygon_holes_sizes_pair
    min_polygon_hole_size, max_polygon_hole_size = polygon_hole_sizes_pair

    strategy = mixes(x_coordinates, y_coordinates,
                     min_points_size=min_points_size,
                     max_points_size=max_points_size,
                     min_segments_size=min_segments_size,
                     max_segments_size=max_segments_size,
                     min_polygons_size=min_polygons_size,
                     max_polygons_size=max_polygons_size,
                     min_polygon_border_size=min_polygon_border_size,
                     max_polygon_border_size=max_polygon_border_size,
                     min_polygon_holes_size=min_polygon_holes_size,
                     max_polygon_holes_size=max_polygon_holes_size,
                     min_polygon_hole_size=min_polygon_hole_size,
                     max_polygon_hole_size=max_polygon_hole_size)

    result = data.draw(strategy)

    assert is_mix(result)
    assert mix_has_valid_sizes(result,
                               min_points_size=min_points_size,
                               max_points_size=max_points_size,
                               min_segments_size=min_segments_size,
                               max_segments_size=max_segments_size,
                               min_polygons_size=min_polygons_size,
                               max_polygons_size=max_polygons_size,
                               min_polygon_border_size=min_polygon_border_size,
                               max_polygon_border_size=max_polygon_border_size,
                               min_polygon_holes_size=min_polygon_holes_size,
                               max_polygon_holes_size=max_polygon_holes_size,
                               min_polygon_hole_size=min_polygon_hole_size,
                               max_polygon_hole_size=max_polygon_hole_size)
    assert mix_has_coordinates_types(result,
                                     x_type=x_type,
                                     y_type=y_type)
    assert mix_has_coordinates_in_range(result,
                                        min_x_value=min_x_value,
                                        max_x_value=max_x_value,
                                        min_y_value=min_y_value,
                                        max_y_value=max_y_value)
    assert mix_segments_do_not_cross_or_overlap(result)


@given(strategies.data,
       strategies.scalars_strategies_with_limits_and_types,
       strategies.multipoints_sizes_pairs,
       strategies.multisegments_sizes_pairs,
       strategies.multipolygons_sizes_pairs,
       strategies.concave_contours_sizes_pairs,
       strategies.multicontours_sizes_pairs,
       strategies.convex_contours_sizes_pairs)
def test_same_coordinates(data: DataObject,
                          coordinates_limits_type: ScalarsLimitsType,
                          multipoints_sizes_pair: SizesPair,
                          multisegments_sizes_pair: SizesPair,
                          multipolygons_sizes_pair: SizesPair,
                          multipolygons_border_sizes_pair: SizesPair,
                          polygon_holes_sizes_pair: SizesPair,
                          polygon_hole_sizes_pair: SizesPair) -> None:
    ((scalars, (min_multipolygon_value, max_multipolygon_value)),
     type_) = coordinates_limits_type
    min_multipoint_size, max_multipoint_size = multipoints_sizes_pair
    min_multisegment_size, max_multisegment_size = multisegments_sizes_pair
    min_multipolygon_size, max_multipolygon_size = multipolygons_sizes_pair
    (min_multipolygon_border_size,
     max_multipolygon_border_size) = multipolygons_border_sizes_pair
    min_polygon_holes_size, max_polygon_holes_size = polygon_holes_sizes_pair
    min_polygon_hole_size, max_polygon_hole_size = polygon_hole_sizes_pair

    strategy = mixes(scalars, min_points_size=min_multipoint_size,
                     max_points_size=max_multipoint_size,
                     min_segments_size=min_multisegment_size,
                     max_segments_size=max_multisegment_size,
                     min_polygons_size=min_multipolygon_size,
                     max_polygons_size=max_multipolygon_size,
                     min_polygon_border_size=min_multipolygon_border_size,
                     max_polygon_border_size=max_multipolygon_border_size,
                     min_polygon_holes_size=min_polygon_holes_size,
                     max_polygon_holes_size=max_polygon_holes_size,
                     min_polygon_hole_size=min_polygon_hole_size,
                     max_polygon_hole_size=max_polygon_hole_size)

    result = data.draw(strategy)

    assert is_mix(result)
    assert mix_has_valid_sizes(
            result,
            min_points_size=min_multipoint_size,
            max_points_size=max_multipoint_size,
            min_segments_size=min_multisegment_size,
            max_segments_size=max_multisegment_size,
            min_polygons_size=min_multipolygon_size,
            max_polygons_size=max_multipolygon_size,
            min_polygon_border_size=min_multipolygon_border_size,
            max_polygon_border_size=max_multipolygon_border_size,
            min_polygon_holes_size=min_polygon_holes_size,
            max_polygon_holes_size=max_polygon_holes_size,
            min_polygon_hole_size=min_polygon_hole_size,
            max_polygon_hole_size=max_polygon_hole_size)
    assert mix_has_coordinates_types(result,
                                     x_type=type_,
                                     y_type=type_)
    assert mix_has_coordinates_in_range(result,
                                        min_x_value=min_multipolygon_value,
                                        max_x_value=max_multipolygon_value,
                                        min_y_value=min_multipolygon_value,
                                        max_y_value=max_multipolygon_value)
    assert mix_segments_do_not_cross_or_overlap(result)


@given(strategies.scalars_strategies,
       strategies.invalid_mix_points_sizes_pairs)
def test_invalid_points_sizes(scalars: Strategy[Scalar],
                              invalid_sizes_pair: SizesPair) -> None:
    min_points_size, max_points_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        mixes(scalars,
              min_points_size=min_points_size,
              max_points_size=max_points_size)


@given(strategies.scalars_strategies,
       strategies.invalid_mix_polygons_sizes_pairs)
def test_invalid_polygons_sizes(scalars: Strategy[Scalar],
                                invalid_sizes_pair: SizesPair) -> None:
    min_polygons_size, max_polygons_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        mixes(scalars,
              min_polygons_size=min_polygons_size,
              max_polygons_size=max_polygons_size)


@given(strategies.scalars_strategies,
       strategies.invalid_mix_segments_sizes_pairs)
def test_invalid_segments_sizes(scalars: Strategy[Scalar],
                                invalid_sizes_pair: SizesPair) -> None:
    min_segments_size, max_segments_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        mixes(scalars,
              min_segments_size=min_segments_size,
              max_segments_size=max_segments_size)


@given(strategies.scalars_strategies,
       strategies.invalid_convex_contours_sizes_pairs)
def test_invalid_polygon_border_sizes(scalars: Strategy[Scalar],
                                      invalid_sizes_pair: SizesPair) -> None:
    min_polygon_border_size, max_polygon_border_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        mixes(scalars,
              min_polygon_border_size=min_polygon_border_size,
              max_polygon_border_size=max_polygon_border_size)


@given(strategies.scalars_strategies,
       strategies.invalid_polygon_holes_sizes_pairs)
def test_invalid_polygon_holes_list_sizes(scalars: Strategy[Scalar],
                                          invalid_sizes_pair: SizesPair
                                          ) -> None:
    min_polygon_holes_size, max_polygon_holes_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        mixes(scalars,
              min_polygon_holes_size=min_polygon_holes_size,
              max_polygon_holes_size=max_polygon_holes_size)


@given(strategies.scalars_strategies,
       strategies.invalid_convex_contours_sizes_pairs)
def test_invalid_polygon_holes_sizes(scalars: Strategy[Scalar],
                                     invalid_sizes_pair: SizesPair) -> None:
    min_polygon_hole_size, max_polygon_hole_size = invalid_sizes_pair

    with pytest.raises(ValueError):
        mixes(scalars,
              min_polygon_hole_size=min_polygon_hole_size,
              max_polygon_hole_size=max_polygon_hole_size)


@given(strategies.scalars_strategies,
       strategies.non_valid_convex_contours_sizes_pairs)
def test_non_valid_polygon_border_sizes(scalars: Strategy[Scalar],
                                        non_valid_sizes_pair: SizesPair
                                        ) -> None:
    min_polygon_border_size, max_polygon_border_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        mixes(scalars,
              min_polygon_border_size=min_polygon_border_size,
              max_polygon_border_size=max_polygon_border_size)

    assert len(warnings) == 1


@given(strategies.scalars_strategies,
       strategies.non_valid_convex_contours_sizes_pairs)
def test_non_valid_polygon_holes_sizes(scalars: Strategy[Scalar],
                                       non_valid_sizes_pair: SizesPair
                                       ) -> None:
    min_polygon_hole_size, max_polygon_hole_size = non_valid_sizes_pair

    with pytest.warns(HypothesisWarning) as warnings:
        mixes(scalars,
              min_polygon_hole_size=min_polygon_hole_size,
              max_polygon_hole_size=max_polygon_hole_size)

    assert len(warnings) == 1
