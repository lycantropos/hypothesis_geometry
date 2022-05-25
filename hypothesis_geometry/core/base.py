from functools import partial
from itertools import (chain,
                       cycle,
                       repeat)
from operator import add
from typing import (Callable,
                    List,
                    Optional,
                    Sequence,
                    Tuple,
                    Type)

from ground.base import (Context,
                         Orientation)
from ground.hints import (Box,
                          Empty,
                          Linear,
                          Maybe,
                          Mix,
                          Multipoint,
                          Multipolygon,
                          Multisegment,
                          Point,
                          Polygon,
                          Scalar,
                          Segment,
                          Shaped)
from hypothesis import strategies

from .constants import (MIN_CONTOUR_SIZE,
                        MinContourSize)
from .contracts import (are_segments_non_crossing_non_overlapping,
                        are_vertices_non_convex,
                        are_vertices_strict,
                        has_horizontal_lowermost_segment,
                        has_valid_size,
                        has_vertical_leftmost_segment,
                        multicontour_has_valid_sizes)
from .factories import (contour_vertices_to_edges,
                        to_convex_vertices_sequence,
                        to_max_convex_hull,
                        to_multicontour,
                        to_polygon,
                        to_star_contour_vertices,
                        to_vertices_sequence)
from .hints import (Chooser,
                    Domain,
                    Multicontour,
                    Orienteer,
                    Strategy)
from .utils import (cut,
                    pack,
                    pairwise,
                    sort_pair,
                    to_next_prime,
                    to_prior_prime)


def to_boxes(x_coordinates: Strategy[Scalar],
             y_coordinates: Optional[Strategy[Scalar]],
             *,
             context: Context) -> Strategy[Box[Scalar]]:
    return (strategies.builds(add,
                              strategies.lists(x_coordinates,
                                               min_size=2,
                                               max_size=2,
                                               unique=True)
                              .map(sort_pair),
                              strategies.lists(x_coordinates
                                               if y_coordinates is None
                                               else y_coordinates,
                                               min_size=2,
                                               max_size=2,
                                               unique=True)
                              .map(sort_pair))
            .map(pack(context.box_cls)))


def to_choosers() -> Strategy[Chooser]:
    return (strategies.randoms(use_true_random=True)
            .map(lambda random: random.choice))


def to_concave_vertices_sequences(x_coordinates: Strategy[Scalar],
                                  y_coordinates: Optional[Strategy[Scalar]],
                                  *,
                                  min_size: int,
                                  max_size: Optional[int],
                                  context: Context
                                  ) -> Strategy[Sequence[Point[Scalar]]]:
    return (_to_vertices_sequences(x_coordinates, y_coordinates,
                                   min_size=min_size,
                                   max_size=max_size,
                                   context=context)
            .filter(partial(are_vertices_non_convex,
                            orienteer=context.angle_orientation)))


def to_convex_vertices_sequences(x_coordinates: Strategy[Scalar],
                                 y_coordinates: Optional[Strategy[Scalar]],
                                 *,
                                 min_size: int,
                                 max_size: Optional[int],
                                 context: Context
                                 ) -> Strategy[Sequence[Point[Scalar]]]:
    if max_size is not None and max_size < MinContourSize.CONCAVE:
        return to_triangular_vertices_sequences(x_coordinates, y_coordinates,
                                                context=context)
    result = (strategies.builds(partial(to_convex_vertices_sequence,
                                        context=context),
                                to_points_in_general_position(
                                        x_coordinates,
                                        y_coordinates,
                                        min_size=min_size,
                                        max_size=max_size,
                                        context=context
                                ),
                                strategies.randoms(use_true_random=True))
              .filter(partial(has_valid_size,
                              min_size=min_size,
                              max_size=max_size)))
    result = (to_rectangular_vertices_sequences(x_coordinates, y_coordinates,
                                                context=context)
              | result
              if min_size <= 4
              else result)
    return (to_triangular_vertices_sequences(x_coordinates, y_coordinates,
                                             context=context)
            | result
            if min_size == 3
            else result)


def to_empty_geometries(context: Context) -> Strategy[Empty]:
    return strategies.builds(context.empty_cls)


def to_mixes(x_coordinates: Strategy[Scalar],
             y_coordinates: Optional[Strategy[Scalar]],
             *,
             min_points_size: int,
             max_points_size: Optional[int],
             min_segments_size: int,
             max_segments_size: Optional[int],
             min_polygons_size: int,
             max_polygons_size: Optional[int],
             min_polygon_border_size: int,
             max_polygon_border_size: Optional[int],
             min_polygon_holes_size: int,
             max_polygon_holes_size: Optional[int],
             min_polygon_hole_size: int,
             max_polygon_hole_size: Optional[int],
             context: Context) -> Strategy[Mix[Scalar]]:
    if y_coordinates is None:
        y_coordinates = x_coordinates
    min_polygon_points_count = to_next_prime(min_polygon_border_size
                                             + (min_polygon_holes_size
                                                * min_polygon_hole_size))
    min_polygons_points_count = min_polygons_size * min_polygon_points_count
    min_segments_points_count = 2 * min_segments_size
    min_points_count = (min_points_size + min_segments_points_count
                        + min_polygons_points_count)
    empty = context.empty
    mix_cls = context.mix_cls
    multipoint_cls = context.multipoint_cls
    multipolygon_cls = context.multipolygon_cls
    multisegment_cls = context.multisegment_cls

    @strategies.composite
    def xs_to_mix(draw: Callable[[Strategy[Domain]], Domain],
                  xs: List[Scalar]) -> Mix:
        (
            points_counts, segments_endpoints_counts, polygons_vertices_counts
        ) = to_points_counts(draw, len(xs))
        xs = sorted(xs)
        points, segments, polygons = [], [], []

        def draw_points(points_count: int) -> None:
            points.extend(draw(
                    to_unique_points_sequences(
                            strategies.sampled_from(xs[:points_count]),
                            y_coordinates,
                            min_size=points_count,
                            max_size=points_count,
                            context=context
                    )
            ))

        def draw_segments(points_count: int) -> None:
            size = points_count // 2
            segments.extend(draw(
                    to_non_crossing_non_overlapping_segments_sequences(
                            strategies.sampled_from(xs[:points_count]),
                            y_coordinates,
                            min_size=size,
                            max_size=size,
                            context=context
                    )
            ))

        def draw_polygon(points_count: int) -> None:
            polygon_x_coordinates = strategies.sampled_from(
                    xs[:to_prior_prime(points_count)]
            )
            polygons.append(draw(
                    to_polygons(polygon_x_coordinates,
                                y_coordinates,
                                min_size=min_polygon_border_size,
                                max_size=max_polygon_border_size,
                                min_holes_size=min_polygon_holes_size,
                                max_holes_size=max_polygon_holes_size,
                                min_hole_size=min_polygon_hole_size,
                                max_hole_size=max_polygon_hole_size,
                                context=context)
            ))

        drawers_with_points_counts = draw(strategies.permutations(
                tuple(chain(zip(repeat(draw_points), points_counts),
                            zip(repeat(draw_segments),
                                segments_endpoints_counts),
                            zip(repeat(draw_polygon),
                                polygons_vertices_counts)))
        ))
        to_contour_segments = context.contour_segments
        for index, (drawer, count) in enumerate(drawers_with_points_counts):
            drawer(count)
            can_touch_next_geometry = (
                    (index < len(drawers_with_points_counts) - 1
                     and (drawers_with_points_counts[index + 1][0]
                          is not draw_points))
                    and (drawer is draw_segments
                         and index < len(drawers_with_points_counts) - 1
                         and (drawers_with_points_counts[index + 1]
                              is not draw_points)
                         and not has_vertical_leftmost_segment(segments)
                         or drawer is draw_polygon
                         and index < len(drawers_with_points_counts) - 1
                         and (drawers_with_points_counts[index + 1]
                              is not draw_points)
                         and
                         not has_vertical_leftmost_segment(
                                 to_contour_segments(polygons[-1].border)
                         ))
            )
            xs = xs[count - can_touch_next_geometry:]
        return mix_cls(unpack_points(points), unpack_segments(segments),
                       unpack_polygons(polygons))

    @strategies.composite
    def ys_to_mix(draw: Callable[[Strategy[Domain]], Domain],
                  ys: List[Scalar]) -> Mix:
        (
            points_counts, segments_endpoints_counts, polygons_vertices_counts
        ) = to_points_counts(draw, len(ys))
        ys = sorted(ys)
        points, segments, polygons = [], [], []

        def draw_points(points_count: int) -> None:
            points.extend(draw(
                    to_unique_points_sequences(
                            x_coordinates,
                            strategies.sampled_from(ys[:points_count]),
                            min_size=points_count,
                            max_size=points_count,
                            context=context
                    )
            ))

        def draw_segments(points_count: int) -> None:
            size = points_count // 2
            segments.extend(draw(
                    to_non_crossing_non_overlapping_segments_sequences(
                            x_coordinates,
                            strategies.sampled_from(ys[:points_count]),
                            min_size=size,
                            max_size=size,
                            context=context
                    )
            ))

        def draw_polygon(points_count: int) -> None:
            polygons.append(draw(
                    to_polygons(x_coordinates,
                                strategies.sampled_from(ys[:points_count]),
                                min_size=min_polygon_border_size,
                                max_size=max_polygon_border_size,
                                min_holes_size=min_polygon_holes_size,
                                max_holes_size=max_polygon_holes_size,
                                min_hole_size=min_polygon_hole_size,
                                max_hole_size=max_polygon_hole_size,
                                context=context)
            ))

        drawers_with_points_counts = draw(
                strategies.permutations(
                        tuple(chain(zip(repeat(draw_points), points_counts),
                                    zip(repeat(draw_segments),
                                        segments_endpoints_counts),
                                    zip(repeat(draw_polygon),
                                        polygons_vertices_counts)))
                )
        )
        to_contour_segments = context.contour_segments
        for index, (drawer, count) in enumerate(drawers_with_points_counts):
            drawer(count)
            can_touch_next_geometry = (
                    (index < len(drawers_with_points_counts) - 1
                     and (drawers_with_points_counts[index + 1][0]
                          is not draw_points))
                    and (drawer is draw_segments
                         and index < len(drawers_with_points_counts) - 1
                         and (drawers_with_points_counts[index + 1]
                              is not draw_points)
                         and not has_horizontal_lowermost_segment(segments)
                         or drawer is draw_polygon
                         and index < len(drawers_with_points_counts) - 1
                         and (drawers_with_points_counts[index + 1]
                              is not draw_points)
                         and
                         not has_horizontal_lowermost_segment(
                                 to_contour_segments(polygons[-1].border)
                         ))
            )
            ys = ys[count - can_touch_next_geometry:]
        return mix_cls(unpack_points(points), unpack_segments(segments),
                       unpack_polygons(polygons))

    def unpack_points(points: Sequence[Point]) -> Maybe[Multipoint]:
        return multipoint_cls(points) if points else empty

    def unpack_segments(segments: Sequence[Segment]) -> Maybe[Linear]:
        return ((multisegment_cls(segments)
                 if len(segments) > 1
                 else segments[0])
                if segments
                else empty)

    def unpack_polygons(polygons: Sequence[Polygon]) -> Maybe[Shaped]:
        return ((multipolygon_cls(polygons)
                 if len(polygons) > 1
                 else polygons[0])
                if polygons
                else empty)

    def to_points_counts(draw: Callable[[Strategy[Domain]], Domain],
                         max_points_count: int
                         ) -> Tuple[List[int], List[int], List[int]]:
        max_polygons_points_count = (max_points_count - min_points_size
                                     - min_segments_points_count)
        polygons_size_upper_bound = (max_polygons_points_count
                                     // min_polygon_points_count)
        polygons_size = draw(strategies.integers(
                min_polygons_size,
                polygons_size_upper_bound
                if max_polygons_size is None
                else min(polygons_size_upper_bound, max_polygons_size)))
        max_polygon_points_count = (to_prior_prime(max_polygons_points_count
                                                   // polygons_size)
                                    if polygons_size
                                    else 0)
        polygons_points_counts = (
            [draw(polygons_points_counts)
             for polygons_points_counts in repeat(
                    strategies.integers(min_polygon_points_count,
                                        max_polygon_points_count),
                    polygons_size)]
            if polygons_size
            else [])
        polygons_points_count = sum(polygons_points_counts)
        segments_endpoints_count_upper_bound = (max_points_count
                                                - polygons_points_count
                                                - min_points_size)
        max_segments_points_count = (
            segments_endpoints_count_upper_bound
            if max_segments_size is None
            else min(segments_endpoints_count_upper_bound,
                     2 * max_segments_size))
        segments_points_count = draw(strategies.sampled_from(
                range(min_segments_points_count,
                      max_segments_points_count + 1, 2)))
        points_size_upper_bound = (max_points_count - segments_points_count
                                   - polygons_points_count)
        points_size = (points_size_upper_bound
                       if max_points_size is None
                       else min(points_size_upper_bound, max_points_size))
        segments_size = segments_points_count // 2
        return (partition(draw, points_size),
                [2 * size for size in partition(draw, segments_size)],
                polygons_points_counts)

    def partition(draw: Callable[[Strategy[Domain]], Domain],
                  value: int) -> List[int]:
        assert value >= 0, 'Value should be non-negative.'
        result = []
        while value:
            part = draw(strategies.integers(1, value))
            result.append(part)
            value -= part
        return result

    return ((strategies.lists(x_coordinates,
                              min_size=min_points_count,
                              unique=True)
             .flatmap(xs_to_mix))
            | (strategies.lists(y_coordinates,
                                min_size=min_points_count,
                                unique=True)
               .flatmap(ys_to_mix)))


def to_multicontours(x_coordinates: Strategy[Scalar],
                     y_coordinates: Optional[Strategy[Scalar]],
                     *,
                     min_size: int,
                     max_size: Optional[int],
                     min_contour_size: int,
                     max_contour_size: Optional[int],
                     context: Context) -> Strategy[Multicontour[Scalar]]:
    def points_to_multicontours(points: List[Point]) -> Strategy[Multicontour]:
        return strategies.builds(partial(to_multicontour, points,
                                         context=context),
                                 to_sizes(len(points)), to_choosers())

    def to_sizes(limit: int) -> Strategy[List[int]]:
        size_upper_bound = limit // min_contour_size
        return (strategies.integers(min_size,
                                    size_upper_bound
                                    if max_size is None
                                    else min(size_upper_bound, max_size))
                .flatmap(partial(_to_sizes,
                                 min_element_size=min_contour_size,
                                 limit=limit)))

    def _to_sizes(size: int,
                  min_element_size: int,
                  limit: int) -> Strategy[List[int]]:
        max_sizes = [min_element_size] * size
        indices = cycle(range(size))
        for _ in range(limit - size * min_element_size):
            max_sizes[next(indices)] += 1
        sizes_ranges = [range(min_element_size, max_element_size + 1)
                        for max_element_size in max_sizes]
        return (strategies.tuples(*[strategies.sampled_from(sizes_range)
                                    for sizes_range in sizes_ranges])
                .flatmap(strategies.permutations))

    min_points_count = min_size * min_contour_size
    max_points_count = (None
                        if max_size is None or max_contour_size is None
                        else max_size * max_contour_size)
    return (to_points_in_general_position(x_coordinates, y_coordinates,
                                          min_size=min_points_count,
                                          max_size=max_points_count,
                                          context=context)
            .flatmap(points_to_multicontours)
            .filter(partial(multicontour_has_valid_sizes,
                            min_size=min_size,
                            max_size=max_size,
                            min_contour_size=min_contour_size,
                            max_contour_size=max_contour_size)))


def to_multipoints(x_coordinates: Strategy[Scalar],
                   y_coordinates: Optional[Strategy[Scalar]],
                   *,
                   min_size: int,
                   max_size: Optional[int],
                   context: Context) -> Strategy[Multipoint[Scalar]]:
    return (to_unique_points_sequences(x_coordinates, y_coordinates,
                                       min_size=min_size,
                                       max_size=max_size,
                                       context=context)
            .map(context.multipoint_cls))


def to_multipolygons(x_coordinates: Strategy[Scalar],
                     y_coordinates: Optional[Strategy[Scalar]],
                     *,
                     min_size: int,
                     max_size: Optional[int],
                     min_border_size: int,
                     max_border_size: Optional[int],
                     min_holes_size: int,
                     max_holes_size: Optional[int],
                     min_hole_size: int,
                     max_hole_size: Optional[int],
                     context: Context) -> Strategy[Multipolygon]:
    if y_coordinates is None:
        y_coordinates = x_coordinates
    min_polygon_points_count = to_next_prime(min_border_size
                                             + min_holes_size * min_hole_size)

    @strategies.composite
    def xs_to_polygons(draw: Callable[[Strategy[Domain]], Domain],
                       xs: List[Scalar]) -> Sequence[Polygon]:
        size_upper_bound = len(xs) // min_polygon_points_count
        size = draw(strategies.integers(min_size,
                                        size_upper_bound
                                        if max_size is None
                                        else min(max_size, size_upper_bound)))
        xs = sorted(xs)
        to_contour_segments = context.contour_segments
        result = []
        start, coordinates_count = 0, len(xs)
        for index in range(size - 1):
            polygon_points_count = draw(
                    strategies.integers(min_polygon_points_count,
                                        (coordinates_count - start)
                                        // (size - index))
            )
            polygon_xs = xs[start:start + to_prior_prime(polygon_points_count)]
            polygon = draw(to_polygons(strategies.sampled_from(polygon_xs),
                                       y_coordinates,
                                       min_size=min_border_size,
                                       max_size=max_border_size,
                                       min_holes_size=min_holes_size,
                                       max_holes_size=max_holes_size,
                                       min_hole_size=min_hole_size,
                                       max_hole_size=max_hole_size,
                                       context=context))
            result.append(polygon)
            can_touch_next_polygon = not has_vertical_leftmost_segment(
                    to_contour_segments(polygon.border)
            )
            start += polygon_points_count - can_touch_next_polygon
        result.append(draw(to_polygons(strategies.sampled_from(xs[start:]),
                                       y_coordinates,
                                       min_size=min_border_size,
                                       max_size=max_border_size,
                                       min_holes_size=min_holes_size,
                                       max_holes_size=max_holes_size,
                                       min_hole_size=min_hole_size,
                                       max_hole_size=max_hole_size,
                                       context=context)))
        return result

    @strategies.composite
    def ys_to_polygons(draw: Callable[[Strategy[Domain]], Domain],
                       ys: List[Scalar]) -> Sequence[Polygon[Scalar]]:
        size_upper_bound = len(ys) // min_polygon_points_count
        size = draw(strategies.integers(min_size,
                                        size_upper_bound
                                        if max_size is None
                                        else min(max_size, size_upper_bound)))
        ys = sorted(ys)
        to_contour_segments = context.contour_segments
        result = []
        start, coordinates_count = 0, len(ys)
        for index in range(size - 1):
            polygon_points_count = draw(
                    strategies.integers(min_polygon_points_count,
                                        (coordinates_count - start)
                                        // (size - index))
            )
            polygon_ys = ys[start:start + to_prior_prime(polygon_points_count)]
            polygon = draw(to_polygons(x_coordinates,
                                       strategies.sampled_from(polygon_ys),
                                       min_size=min_border_size,
                                       max_size=max_border_size,
                                       min_holes_size=min_holes_size,
                                       max_holes_size=max_holes_size,
                                       min_hole_size=min_hole_size,
                                       max_hole_size=max_hole_size,
                                       context=context))
            result.append(polygon)
            can_touch_next_polygon = not has_horizontal_lowermost_segment(
                    to_contour_segments(polygon.border))
            start += polygon_points_count - can_touch_next_polygon
        result.append(draw(to_polygons(x_coordinates,
                                       strategies.sampled_from(ys[start:]),
                                       min_size=min_border_size,
                                       max_size=max_border_size,
                                       min_holes_size=min_holes_size,
                                       max_holes_size=max_holes_size,
                                       min_hole_size=min_hole_size,
                                       max_hole_size=max_hole_size,
                                       context=context)))
        return result

    min_points_count = min_size * min_polygon_points_count
    max_points_count = (None
                        if (max_size is None
                            or max_border_size is None
                            or max_holes_size is None
                            or max_hole_size is None)
                        else max(max_size * (max_border_size
                                             + max_hole_size * max_holes_size),
                                 min_points_count))
    polygons = ((strategies.lists(x_coordinates,
                                  min_size=min_points_count,
                                  max_size=max_points_count,
                                  unique=True)
                 .flatmap(xs_to_polygons))
                | (strategies.lists(y_coordinates,
                                    min_size=min_points_count,
                                    max_size=max_points_count,
                                    unique=True)
                   .flatmap(ys_to_polygons)))
    if not min_holes_size:
        def multicontour_to_polygons(multicontour: Multicontour[Scalar],
                                     polygon_cls: Type[Polygon]
                                     = context.polygon_cls
                                     ) -> Sequence[Polygon[Scalar]]:
            return [polygon_cls(contour, []) for contour in multicontour]

        polygons = ((to_multicontours(x_coordinates, y_coordinates,
                                      min_size=min_size,
                                      max_size=max_size,
                                      min_contour_size=min_border_size,
                                      max_contour_size=max_border_size,
                                      context=context)
                     .map(multicontour_to_polygons))
                    | polygons)
    return polygons.map(context.multipolygon_cls)


def to_multisegments(x_coordinates: Strategy[Scalar],
                     y_coordinates: Optional[Strategy[Scalar]],
                     *,
                     min_size: int,
                     max_size: Optional[int],
                     context: Context) -> Strategy[Multisegment[Scalar]]:
    return (to_non_crossing_non_overlapping_segments_sequences(
            x_coordinates,
            y_coordinates,
            min_size=min_size,
            max_size=max_size,
            context=context)
            .map(context.multisegment_cls))


def to_non_crossing_non_overlapping_segments_sequences(
        x_coordinates: Strategy[Scalar],
        y_coordinates: Optional[Strategy[Scalar]],
        *,
        min_size: int,
        max_size: Optional[int],
        context: Context
) -> Strategy[Sequence[Segment[Scalar]]]:
    if y_coordinates is None:
        y_coordinates = x_coordinates
    if max_size is not None and max_size < 2:
        return (to_segments(x_coordinates, y_coordinates,
                            context=context)
                .map(lambda segment: [segment])
                if max_size
                else strategies.builds(list))
    point_cls, segment_cls = context.point_cls, context.segment_cls

    def to_vertical_multisegment(x: Scalar,
                                 ys: List[Scalar]) -> Sequence[Segment]:
        return [segment_cls(point_cls(x, y), point_cls(x, next_y))
                for y, next_y in pairwise(sorted(ys))]

    def to_horizontal_multisegment(xs: List[Scalar],
                                   y: Scalar) -> Sequence[Segment]:
        return [segment_cls(point_cls(x, y), point_cls(next_x, y))
                for x, next_x in pairwise(sorted(xs))]

    next_min_size, next_max_size = (min_size + 1, (max_size
                                                   if max_size is None
                                                   else max_size + 1))
    result = (strategies.builds(to_vertical_multisegment, x_coordinates,
                                strategies.lists(y_coordinates,
                                                 min_size=next_min_size,
                                                 max_size=next_max_size,
                                                 unique=True))
              | strategies.builds(to_horizontal_multisegment,
                                  strategies.lists(x_coordinates,
                                                   min_size=next_min_size,
                                                   max_size=next_max_size,
                                                   unique=True),
                                  y_coordinates))
    if max_size is None or max_size >= MIN_CONTOUR_SIZE:
        result |= (to_vertices_sequences(x_coordinates, y_coordinates,
                                         min_size=max(min_size,
                                                      MIN_CONTOUR_SIZE),
                                         max_size=max_size,
                                         context=context)
                   .map(partial(contour_vertices_to_edges,
                                segment_cls=context.segment_cls))
                   .flatmap(partial(to_sub_lists,
                                    min_size=min_size)))
    return (result
            | (strategies.lists(to_segments(x_coordinates, y_coordinates,
                                            context=context),
                                min_size=min_size,
                                max_size=max_size)
               .filter(partial(are_segments_non_crossing_non_overlapping,
                               context=context))))


def to_points(x_coordinates: Strategy[Scalar],
              y_coordinates: Optional[Strategy[Scalar]],
              *,
              context: Context) -> Strategy[Point[Scalar]]:
    return strategies.builds(context.point_cls,
                             x_coordinates,
                             x_coordinates
                             if y_coordinates is None
                             else y_coordinates)


def to_points_in_general_position(x_coordinates: Strategy[Scalar],
                                  y_coordinates: Optional[Strategy[Scalar]],
                                  *,
                                  min_size: int,
                                  max_size: Optional[int],
                                  context: Context
                                  ) -> Strategy[Sequence[Point[Scalar]]]:
    grid_size_lower_bound = to_next_prime(min_size)
    grid_size_upper_bound = (max_size
                             if max_size is None
                             else max(max_size, grid_size_lower_bound))

    def coordinates_to_points(xs: Sequence[Scalar], ys: Sequence[Scalar]
                              ) -> Strategy[Sequence[Point[Scalar]]]:
        xs, ys = sorted(xs), sorted(ys)
        grid_size = to_prior_prime(min(len(xs), len(ys)))
        scales = (strategies.integers(0)
                  .flatmap(lambda scale
                           : strategies.integers(scale * grid_size + 1,
                                                 (scale + 1) * grid_size - 1)))
        return (strategies.builds(partial(x_indices_to_points, grid_size, xs,
                                          ys),
                                  scales)
                | strategies.builds(partial(y_indices_to_points, grid_size, xs,
                                            ys),
                                    scales))

    def x_indices_to_points(grid_size: int,
                            xs: Sequence[Scalar],
                            ys: Sequence[Scalar],
                            scale: int,
                            point_cls: Type[Point[Scalar]] = context.point_cls
                            ) -> Sequence[Point[Scalar]]:
        return [point_cls(xs[index],
                          ys[(scale * (index * index)) % grid_size])
                for index in range(grid_size)]

    def y_indices_to_points(grid_size: int,
                            xs: Sequence[Scalar],
                            ys: Sequence[Scalar],
                            scale: int,
                            point_cls: Type[Point[Scalar]] = context.point_cls
                            ) -> Sequence[Point[Scalar]]:
        return [point_cls(xs[(scale * (index * index)) % grid_size], ys[index])
                for index in range(grid_size)]

    return (strategies.tuples(strategies.lists(x_coordinates,
                                               unique=True,
                                               min_size=grid_size_lower_bound,
                                               max_size=grid_size_upper_bound),
                              strategies.lists(x_coordinates
                                               if y_coordinates is None
                                               else y_coordinates,
                                               unique=True,
                                               min_size=grid_size_lower_bound,
                                               max_size=grid_size_upper_bound))
            .flatmap(pack(coordinates_to_points)))


def to_polygons(x_coordinates: Strategy[Scalar],
                y_coordinates: Optional[Strategy[Scalar]],
                *,
                min_size: int,
                max_size: Optional[int],
                min_holes_size: int,
                max_holes_size: Optional[int],
                min_hole_size: int,
                max_hole_size: Optional[int],
                context: Context) -> Strategy[Polygon[Scalar]]:
    def points_to_polygons(points: Sequence[Point[Scalar]]
                           ) -> Strategy[Polygon[Scalar]]:
        max_border_points_count = len(points) - min_inner_points_count
        min_border_size = max(min_size,
                              len(context.points_convex_hull(points)))
        max_border_size = (max_border_points_count
                           if max_size is None
                           else min(max_size, max_border_points_count))
        return strategies.builds(partial(to_polygon, points,
                                         context=context),
                                 strategies.integers(min_border_size,
                                                     max_border_size),
                                 to_holes_sizes(points),
                                 to_choosers())

    def to_holes_sizes(points: Sequence[Point]) -> Strategy[List[int]]:
        max_inner_points_count = (
                len(points)
                - len(to_max_convex_hull(points, context.angle_orientation))
        )
        holes_size_scale = max_inner_points_count // min_hole_size
        points_max_hole_size = (holes_size_scale
                                if max_holes_size is None
                                else min(max_holes_size, holes_size_scale))
        return ((strategies.integers(min_holes_size, points_max_hole_size)
                 .flatmap(partial(_to_holes_sizes,
                                  min_hole_points_count=min_hole_size,
                                  max_hole_points_count
                                  =max_inner_points_count)))
                if max_inner_points_count >= min_hole_size
                else strategies.builds(list))

    def _to_holes_sizes(holes_size: int,
                        min_hole_points_count: int,
                        max_hole_points_count: int) -> Strategy[List[int]]:
        if not holes_size:
            return strategies.builds(list)
        max_holes_points_counts = [min_hole_points_count] * holes_size
        indices = cycle(range(holes_size))
        for _ in range(max_hole_points_count
                       - holes_size * min_hole_points_count):
            max_holes_points_counts[next(indices)] += 1
        sizes_ranges = [range(min_hole_points_count, max_hole_points_count + 1)
                        for max_hole_points_count in max_holes_points_counts]
        return (strategies.permutations([strategies.sampled_from(sizes_range)
                                         for sizes_range in sizes_ranges])
                .flatmap(pack(strategies.tuples))
                .map(list))

    def has_valid_sizes(polygon: Polygon) -> bool:
        return (has_valid_size(polygon.border.vertices,
                               min_size=min_size,
                               max_size=max_size)
                and multicontour_has_valid_sizes(
                        polygon.holes,
                        min_size=min_holes_size,
                        max_size=max_holes_size,
                        min_contour_size=min_hole_size,
                        max_contour_size=max_hole_size
                ))

    min_inner_points_count = min_hole_size * min_holes_size

    def has_valid_inner_points_count(points: Sequence[Point]) -> bool:
        return ((max_size is None
                 or len(context.points_convex_hull(points)) <= max_size)
                and (len(points)
                     - len(to_max_convex_hull(points,
                                              context.angle_orientation))
                     >= min_inner_points_count))

    min_points_count = min_size + min_inner_points_count
    max_points_count = (None
                        if (max_size is None or max_holes_size is None
                            or max_hole_size is None)
                        else max_size + max_hole_size * max_holes_size)
    return (to_points_in_general_position(x_coordinates, y_coordinates,
                                          min_size=min_points_count,
                                          max_size=max_points_count,
                                          context=context)
            .filter(has_valid_inner_points_count)
            .flatmap(points_to_polygons)
            .filter(has_valid_sizes))


def to_rectangular_vertices_sequences(x_coordinates: Strategy[Scalar],
                                      y_coordinates: Strategy[Scalar],
                                      *,
                                      context: Context
                                      ) -> Strategy[Sequence[Point[Scalar]]]:
    def to_vertices(box: Box, point_cls: Type[Point] = context.point_cls
                    ) -> Sequence[Point]:
        return [
            point_cls(box.min_x, box.min_y), point_cls(box.max_x, box.min_y),
            point_cls(box.max_x, box.max_y), point_cls(box.min_x, box.max_y),
        ]

    return (to_boxes(x_coordinates, y_coordinates,
                     context=context)
            .map(to_vertices))


def to_segments(x_coordinates: Strategy[Scalar],
                y_coordinates: Optional[Strategy[Scalar]],
                *,
                context: Context) -> Strategy[Segment[Scalar]]:
    def non_degenerate_endpoints(endpoints: Tuple[Point, Point]) -> bool:
        start, end = endpoints
        return start != end

    points = to_points(x_coordinates, y_coordinates,
                       context=context)
    return (strategies.tuples(points, points)
            .filter(non_degenerate_endpoints)
            .map(pack(context.segment_cls)))


def to_star_vertices_sequences(x_coordinates: Strategy[Scalar],
                               y_coordinates: Optional[Strategy[Scalar]],
                               *,
                               min_size: int,
                               max_size: Optional[int],
                               context: Context
                               ) -> Strategy[Sequence[Point[Scalar]]]:
    return (to_points_in_general_position(x_coordinates, y_coordinates,
                                          min_size=min_size,
                                          max_size=max_size,
                                          context=context)
            .map(partial(to_star_contour_vertices,
                         context=context))
            .filter(partial(has_valid_size,
                            min_size=min_size,
                            max_size=max_size)))


def to_sub_lists(values: Sequence[Domain],
                 *,
                 min_size: int) -> Strategy[List[Domain]]:
    return strategies.builds(cut,
                             strategies.permutations(values),
                             strategies.integers(min_size, len(values)))


def to_triangular_vertices_sequences(x_coordinates: Strategy[Scalar],
                                     y_coordinates: Optional[Strategy[Scalar]],
                                     *,
                                     context: Context
                                     ) -> Strategy[Sequence[Point[Scalar]]]:
    def to_counterclockwise_vertices(vertices_triplet
                                     : Tuple[Point, Point, Point],
                                     orienteer: Orienteer
                                     = context.angle_orientation
                                     ) -> Tuple[Point, Point, Point]:
        return (vertices_triplet
                if orienteer(*vertices_triplet) is Orientation.COUNTERCLOCKWISE
                else vertices_triplet[::-1])

    vertices = to_points(x_coordinates, y_coordinates,
                         context=context)
    return (strategies.tuples(vertices, vertices, vertices)
            .filter(partial(are_vertices_strict,
                            orienteer=context.angle_orientation))
            .map(to_counterclockwise_vertices)
            .map(list))


def to_unique_points_sequences(x_coordinates: Strategy[Scalar],
                               y_coordinates: Optional[Strategy[Scalar]],
                               *,
                               min_size: int,
                               max_size: Optional[int],
                               context: Context
                               ) -> Strategy[Sequence[Point[Scalar]]]:
    return strategies.lists(to_points(x_coordinates, y_coordinates,
                                      context=context),
                            unique=True,
                            min_size=min_size,
                            max_size=max_size)


def to_vertices_sequences(
        x_coordinates: Strategy[Scalar],
        y_coordinates: Optional[Strategy[Scalar]],
        *,
        min_size: int,
        max_size: Optional[int],
        context: Context
) -> Strategy[Sequence[Point[Scalar]]]:
    return (to_convex_vertices_sequences(x_coordinates, y_coordinates,
                                         min_size=min_size,
                                         max_size=max_size,
                                         context=context)
            | _to_vertices_sequences(x_coordinates, y_coordinates,
                                     min_size=min_size,
                                     max_size=max_size,
                                     context=context))


def _to_vertices_sequences(
        x_coordinates: Strategy[Scalar],
        y_coordinates: Optional[Strategy[Scalar]],
        *,
        min_size: int,
        max_size: Optional[int],
        context: Context
) -> Strategy[Sequence[Point[Scalar]]]:
    def to_points_with_sizes(points: Sequence[Point]
                             ) -> Strategy[Tuple[Sequence[Point], int]]:
        sizes = strategies.integers(min_size,
                                    len(points)
                                    if max_size is None
                                    else min(len(points), max_size))
        return strategies.tuples(strategies.just(points), sizes)

    return (to_star_vertices_sequences(x_coordinates, y_coordinates,
                                       min_size=min_size,
                                       max_size=max_size,
                                       context=context)
            | (to_points_in_general_position(x_coordinates, y_coordinates,
                                             min_size=min_size,
                                             max_size=max_size,
                                             context=context)
               .flatmap(to_points_with_sizes)
               .map(pack(partial(to_vertices_sequence,
                                 context=context)))
               .filter(partial(has_valid_size,
                               min_size=min_size,
                               max_size=max_size))))
