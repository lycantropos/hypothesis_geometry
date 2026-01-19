from collections.abc import Callable, Sequence
from functools import partial
from itertools import chain, cycle, groupby, repeat
from operator import add
from typing import Any, TypeAlias

from ground.context import Context
from ground.enums import Orientation
from ground.hints import (
    Box,
    Empty,
    Linear,
    Mix,
    Multipoint,
    Multipolygon,
    Multisegment,
    Point,
    Polygon,
    Segment,
    Shaped,
)
from hypothesis import strategies as st

from .constants import MIN_CONTOUR_SIZE, MinContourSize
from .contracts import (
    are_segments_non_crossing_non_overlapping,
    are_vertices_non_convex,
    are_vertices_strict,
    has_horizontal_lowermost_segment,
    has_valid_size,
    has_vertical_leftmost_segment,
    multicontour_has_valid_sizes,
)
from .factories import (
    contour_vertices_to_edges,
    to_convex_vertex_sequence,
    to_max_convex_hull,
    to_multicontour,
    to_polygon,
    to_star_contour_vertices,
    to_vertex_sequence,
)
from .hints import Chooser, Domain, Multicontour, Orienteer, ScalarT
from .utils import (
    cut,
    pack,
    pairwise,
    sort_pair,
    to_next_prime,
    to_prior_prime,
)


def to_boxes(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    context: Context[ScalarT],
) -> st.SearchStrategy[Box[ScalarT]]:
    return st.builds(
        add,
        st.lists(x_coordinates, min_size=2, max_size=2, unique=True).map(
            sort_pair
        ),
        st.lists(
            x_coordinates if y_coordinates is None else y_coordinates,
            min_size=2,
            max_size=2,
            unique=True,
        ).map(sort_pair),
    ).map(pack(context.box_cls))


def to_choosers() -> st.SearchStrategy[Chooser[Any]]:
    return st.randoms(use_true_random=True).map(lambda random: random.choice)


def to_concave_vertex_sequences(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    return _to_vertex_sequences(
        x_coordinates,
        y_coordinates,
        min_size=min_size,
        max_size=max_size,
        context=context,
    ).filter(
        partial(are_vertices_non_convex, orienteer=context.angle_orientation)
    )


def to_convex_vertex_sequences(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    if max_size is not None and max_size < MinContourSize.CONCAVE:
        return to_triangular_vertex_sequences(
            x_coordinates, y_coordinates, context=context
        )
    result = st.builds(
        partial(to_convex_vertex_sequence, context=context),
        to_points_in_general_position(
            x_coordinates,
            y_coordinates,
            min_size=min_size,
            max_size=max_size,
            context=context,
        ),
        st.randoms(use_true_random=True),
    ).filter(partial(has_valid_size, min_size=min_size, max_size=max_size))
    result = (
        to_rectangular_vertex_sequences(
            x_coordinates, y_coordinates, context=context
        )
        | result
        if min_size <= 4
        else result
    )
    return (
        to_triangular_vertex_sequences(
            x_coordinates, y_coordinates, context=context
        )
        | result
        if min_size == 3
        else result
    )


def to_empty_geometries(
    context: Context[ScalarT],
) -> st.SearchStrategy[Empty[ScalarT]]:
    return st.builds(context.empty_cls)


_Draw: TypeAlias = Callable[[st.SearchStrategy[Any]], Any]


def to_mixes(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_points_size: int,
    max_points_size: int | None,
    min_segments_size: int,
    max_segments_size: int | None,
    min_polygons_size: int,
    max_polygons_size: int | None,
    min_polygon_border_size: int,
    max_polygon_border_size: int | None,
    min_polygon_holes_size: int,
    max_polygon_holes_size: int | None,
    min_polygon_hole_size: int,
    max_polygon_hole_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Mix[ScalarT]]:
    if y_coordinates is None:
        y_coordinates = x_coordinates
    min_polygon_points_count = to_next_prime(
        min_polygon_border_size
        + (min_polygon_holes_size * min_polygon_hole_size)
    )
    min_polygons_points_count = min_polygons_size * min_polygon_points_count
    min_segments_points_count = 2 * min_segments_size
    min_points_count = (
        min_points_size + min_segments_points_count + min_polygons_points_count
    )
    empty = context.empty
    mix_cls = context.mix_cls
    multipoint_cls = context.multipoint_cls
    multipolygon_cls = context.multipolygon_cls
    multisegment_cls = context.multisegment_cls

    @st.composite
    def xs_to_mix(draw: _Draw, xs: list[ScalarT]) -> Mix[ScalarT]:
        (
            points_counts,
            segments_endpoints_counts,
            polygons_vertices_counts,
        ) = to_points_counts(draw, len(xs))
        xs = sorted(xs)
        points, segments, polygons = [], [], []

        def draw_points(points_count: int) -> None:
            points.extend(
                draw(
                    to_unique_points_sequences(
                        st.sampled_from(xs[:points_count]),
                        y_coordinates,
                        min_size=points_count,
                        max_size=points_count,
                        context=context,
                    )
                )
            )

        def draw_segments(points_count: int) -> None:
            size = points_count // 2
            segments.extend(
                draw(
                    to_non_crossing_non_overlapping_segments_sequences(
                        st.sampled_from(xs[:points_count]),
                        y_coordinates,
                        min_size=size,
                        max_size=size,
                        context=context,
                    )
                )
            )

        def draw_polygon(points_count: int) -> None:
            polygon_x_coordinates = st.sampled_from(
                xs[: to_prior_prime(points_count)]
            )
            polygons.append(
                draw(
                    to_polygons(
                        polygon_x_coordinates,
                        y_coordinates,
                        min_size=min_polygon_border_size,
                        max_size=max_polygon_border_size,
                        min_holes_size=min_polygon_holes_size,
                        max_holes_size=max_polygon_holes_size,
                        min_hole_size=min_polygon_hole_size,
                        max_hole_size=max_polygon_hole_size,
                        context=context,
                    )
                )
            )

        drawers_with_points_counts = draw(
            st.permutations(
                tuple(
                    chain(
                        zip(repeat(draw_points), points_counts),
                        zip(repeat(draw_segments), segments_endpoints_counts),
                        zip(repeat(draw_polygon), polygons_vertices_counts),
                    )
                )
            )
        )
        to_contour_segments = context.contour_segments
        for index, (drawer, count) in enumerate(drawers_with_points_counts):
            drawer(count)
            can_touch_next_geometry = (
                index < len(drawers_with_points_counts) - 1
                and (
                    drawers_with_points_counts[index + 1][0] is not draw_points
                )
            ) and (
                (
                    drawer is draw_segments
                    and index < len(drawers_with_points_counts) - 1
                    and (
                        drawers_with_points_counts[index + 1]
                        is not draw_points
                    )
                    and not has_vertical_leftmost_segment(segments)
                )
                or (
                    drawer is draw_polygon
                    and index < len(drawers_with_points_counts) - 1
                    and (
                        drawers_with_points_counts[index + 1]
                        is not draw_points
                    )
                    and not has_vertical_leftmost_segment(
                        to_contour_segments(polygons[-1].border)
                    )
                )
            )
            xs = xs[count - can_touch_next_geometry :]
        return mix_cls(
            unpack_points(points),
            unpack_segments(segments),
            unpack_polygons(polygons),
        )

    @st.composite
    def ys_to_mix(draw: _Draw, ys: list[ScalarT]) -> Mix[ScalarT]:
        (
            points_counts,
            segments_endpoints_counts,
            polygons_vertices_counts,
        ) = to_points_counts(draw, len(ys))
        ys = sorted(ys)
        points, segments, polygons = [], [], []

        def draw_points(points_count: int) -> None:
            points.extend(
                draw(
                    to_unique_points_sequences(
                        x_coordinates,
                        st.sampled_from(ys[:points_count]),
                        min_size=points_count,
                        max_size=points_count,
                        context=context,
                    )
                )
            )

        def draw_segments(points_count: int) -> None:
            size = points_count // 2
            segments.extend(
                draw(
                    to_non_crossing_non_overlapping_segments_sequences(
                        x_coordinates,
                        st.sampled_from(ys[:points_count]),
                        min_size=size,
                        max_size=size,
                        context=context,
                    )
                )
            )

        def draw_polygon(points_count: int) -> None:
            polygons.append(
                draw(
                    to_polygons(
                        x_coordinates,
                        st.sampled_from(ys[:points_count]),
                        min_size=min_polygon_border_size,
                        max_size=max_polygon_border_size,
                        min_holes_size=min_polygon_holes_size,
                        max_holes_size=max_polygon_holes_size,
                        min_hole_size=min_polygon_hole_size,
                        max_hole_size=max_polygon_hole_size,
                        context=context,
                    )
                )
            )

        drawers_with_points_counts = draw(
            st.permutations(
                tuple(
                    chain(
                        zip(repeat(draw_points), points_counts),
                        zip(repeat(draw_segments), segments_endpoints_counts),
                        zip(repeat(draw_polygon), polygons_vertices_counts),
                    )
                )
            )
        )
        to_contour_segments = context.contour_segments
        for index, (drawer, count) in enumerate(drawers_with_points_counts):
            drawer(count)
            can_touch_next_geometry = (
                index < len(drawers_with_points_counts) - 1
                and (
                    drawers_with_points_counts[index + 1][0] is not draw_points
                )
            ) and (
                (
                    drawer is draw_segments
                    and index < len(drawers_with_points_counts) - 1
                    and (
                        drawers_with_points_counts[index + 1]
                        is not draw_points
                    )
                    and not has_horizontal_lowermost_segment(segments)
                )
                or (
                    drawer is draw_polygon
                    and index < len(drawers_with_points_counts) - 1
                    and (
                        drawers_with_points_counts[index + 1]
                        is not draw_points
                    )
                    and not has_horizontal_lowermost_segment(
                        to_contour_segments(polygons[-1].border)
                    )
                )
            )
            ys = ys[count - can_touch_next_geometry :]
        return mix_cls(
            unpack_points(points),
            unpack_segments(segments),
            unpack_polygons(polygons),
        )

    def unpack_points(
        points: Sequence[Point[ScalarT]],
    ) -> Multipoint[ScalarT] | Empty[ScalarT]:
        return multipoint_cls(points) if points else empty

    def unpack_segments(
        segments: Sequence[Segment[ScalarT]],
    ) -> Linear[ScalarT] | Empty[ScalarT]:
        return (
            (multisegment_cls(segments) if len(segments) > 1 else segments[0])
            if segments
            else empty
        )

    def unpack_polygons(
        polygons: Sequence[Polygon[ScalarT]],
    ) -> Shaped[ScalarT] | Empty[ScalarT]:
        return (
            (multipolygon_cls(polygons) if len(polygons) > 1 else polygons[0])
            if polygons
            else empty
        )

    def to_points_counts(
        draw: _Draw, max_points_count: int
    ) -> tuple[list[int], list[int], list[int]]:
        max_polygons_points_count = (
            max_points_count - min_points_size - min_segments_points_count
        )
        polygons_size_upper_bound = (
            max_polygons_points_count // min_polygon_points_count
        )
        polygons_size = draw(
            st.integers(
                min_polygons_size,
                polygons_size_upper_bound
                if max_polygons_size is None
                else min(polygons_size_upper_bound, max_polygons_size),
            )
        )
        max_polygon_points_count = (
            to_prior_prime(max_polygons_points_count // polygons_size)
            if polygons_size
            else 0
        )
        polygons_points_counts = (
            [
                draw(polygons_points_counts)
                for polygons_points_counts in repeat(
                    st.integers(
                        min_polygon_points_count, max_polygon_points_count
                    ),
                    polygons_size,
                )
            ]
            if polygons_size
            else []
        )
        polygons_points_count = sum(polygons_points_counts)
        segments_endpoints_count_upper_bound = (
            max_points_count - polygons_points_count - min_points_size
        )
        max_segments_points_count = (
            segments_endpoints_count_upper_bound
            if max_segments_size is None
            else min(
                segments_endpoints_count_upper_bound, 2 * max_segments_size
            )
        )
        segments_points_count = draw(
            st.sampled_from(
                range(
                    min_segments_points_count, max_segments_points_count + 1, 2
                )
            )
        )
        points_size_upper_bound = (
            max_points_count - segments_points_count - polygons_points_count
        )
        points_size = (
            points_size_upper_bound
            if max_points_size is None
            else min(points_size_upper_bound, max_points_size)
        )
        segments_size = segments_points_count // 2
        return (
            partition(draw, points_size),
            [2 * size for size in partition(draw, segments_size)],
            polygons_points_counts,
        )

    def partition(draw: _Draw, value: int) -> list[int]:
        assert value >= 0, 'Value should be non-negative.'
        result = []
        while value:
            part = draw(st.integers(1, value))
            result.append(part)
            value -= part
        return result

    return (
        st.lists(
            x_coordinates, min_size=min_points_count, unique=True
        ).flatmap(xs_to_mix)
    ) | (
        st.lists(
            y_coordinates, min_size=min_points_count, unique=True
        ).flatmap(ys_to_mix)
    )


def to_multicontours(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    min_contour_size: int,
    max_contour_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Multicontour[ScalarT]]:
    def points_to_multicontours(
        points: Sequence[Point[ScalarT]], /
    ) -> st.SearchStrategy[Multicontour[ScalarT]]:
        return st.builds(
            partial(to_multicontour, points, context=context),
            to_sizes(len(points)),
            to_choosers(),
        )

    def to_sizes(limit: int) -> st.SearchStrategy[list[int]]:
        size_upper_bound = limit // min_contour_size
        return st.integers(
            min_size,
            size_upper_bound
            if max_size is None
            else min(size_upper_bound, max_size),
        ).flatmap(
            partial(_to_sizes, min_element_size=min_contour_size, limit=limit)
        )

    def _to_sizes(
        size: int, min_element_size: int, limit: int
    ) -> st.SearchStrategy[list[int]]:
        max_sizes = [min_element_size] * size
        indices = cycle(range(size))
        for _ in range(limit - size * min_element_size):
            max_sizes[next(indices)] += 1
        sizes_ranges = [
            range(min_element_size, max_element_size + 1)
            for max_element_size in max_sizes
        ]
        return st.tuples(
            *[st.sampled_from(sizes_range) for sizes_range in sizes_ranges]
        ).flatmap(st.permutations)

    min_points_count = min_size * min_contour_size
    max_points_count = (
        None
        if max_size is None or max_contour_size is None
        else max_size * max_contour_size
    )
    return (
        to_points_in_general_position(
            x_coordinates,
            y_coordinates,
            min_size=min_points_count,
            max_size=max_points_count,
            context=context,
        )
        .flatmap(points_to_multicontours)
        .filter(
            partial(
                multicontour_has_valid_sizes,
                min_size=min_size,
                max_size=max_size,
                min_contour_size=min_contour_size,
                max_contour_size=max_contour_size,
            )
        )
    )


def to_multipoints(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Multipoint[ScalarT]]:
    return to_unique_points_sequences(
        x_coordinates,
        y_coordinates,
        min_size=min_size,
        max_size=max_size,
        context=context,
    ).map(context.multipoint_cls)


def to_multipolygons(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    min_border_size: int,
    max_border_size: int | None,
    min_holes_size: int,
    max_holes_size: int | None,
    min_hole_size: int,
    max_hole_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Multipolygon[ScalarT]]:
    if y_coordinates is None:
        y_coordinates = x_coordinates
    min_polygon_points_count = to_next_prime(
        min_border_size + min_holes_size * min_hole_size
    )

    @st.composite
    def xs_to_polygons(
        draw: _Draw, xs: list[ScalarT]
    ) -> Sequence[Polygon[ScalarT]]:
        size_upper_bound = len(xs) // min_polygon_points_count
        size = draw(
            st.integers(
                min_size,
                size_upper_bound
                if max_size is None
                else min(max_size, size_upper_bound),
            )
        )
        xs = sorted(xs)
        to_contour_segments = context.contour_segments
        result = []
        start, coordinates_count = 0, len(xs)
        for index in range(size - 1):
            polygon_points_count = draw(
                st.integers(
                    min_polygon_points_count,
                    (coordinates_count - start) // (size - index),
                )
            )
            polygon_xs = xs[
                start : start + to_prior_prime(polygon_points_count)
            ]
            polygon = draw(
                to_polygons(
                    st.sampled_from(polygon_xs),
                    y_coordinates,
                    min_size=min_border_size,
                    max_size=max_border_size,
                    min_holes_size=min_holes_size,
                    max_holes_size=max_holes_size,
                    min_hole_size=min_hole_size,
                    max_hole_size=max_hole_size,
                    context=context,
                )
            )
            result.append(polygon)
            can_touch_next_polygon = not has_vertical_leftmost_segment(
                to_contour_segments(polygon.border)
            )
            start += polygon_points_count - can_touch_next_polygon
        result.append(
            draw(
                to_polygons(
                    st.sampled_from(xs[start:]),
                    y_coordinates,
                    min_size=min_border_size,
                    max_size=max_border_size,
                    min_holes_size=min_holes_size,
                    max_holes_size=max_holes_size,
                    min_hole_size=min_hole_size,
                    max_hole_size=max_hole_size,
                    context=context,
                )
            )
        )
        return result

    @st.composite
    def ys_to_polygons(
        draw: _Draw, ys: list[ScalarT]
    ) -> Sequence[Polygon[ScalarT]]:
        size_upper_bound = len(ys) // min_polygon_points_count
        size = draw(
            st.integers(
                min_size,
                size_upper_bound
                if max_size is None
                else min(max_size, size_upper_bound),
            )
        )
        ys = sorted(ys)
        to_contour_segments = context.contour_segments
        result = []
        start, coordinates_count = 0, len(ys)
        for index in range(size - 1):
            polygon_points_count = draw(
                st.integers(
                    min_polygon_points_count,
                    (coordinates_count - start) // (size - index),
                )
            )
            polygon_ys = ys[
                start : start + to_prior_prime(polygon_points_count)
            ]
            polygon = draw(
                to_polygons(
                    x_coordinates,
                    st.sampled_from(polygon_ys),
                    min_size=min_border_size,
                    max_size=max_border_size,
                    min_holes_size=min_holes_size,
                    max_holes_size=max_holes_size,
                    min_hole_size=min_hole_size,
                    max_hole_size=max_hole_size,
                    context=context,
                )
            )
            result.append(polygon)
            can_touch_next_polygon = not has_horizontal_lowermost_segment(
                to_contour_segments(polygon.border)
            )
            start += polygon_points_count - can_touch_next_polygon
        result.append(
            draw(
                to_polygons(
                    x_coordinates,
                    st.sampled_from(ys[start:]),
                    min_size=min_border_size,
                    max_size=max_border_size,
                    min_holes_size=min_holes_size,
                    max_holes_size=max_holes_size,
                    min_hole_size=min_hole_size,
                    max_hole_size=max_hole_size,
                    context=context,
                )
            )
        )
        return result

    min_points_count = min_size * min_polygon_points_count
    max_points_count = (
        None
        if (
            max_size is None
            or max_border_size is None
            or max_holes_size is None
            or max_hole_size is None
        )
        else max(
            max_size * (max_border_size + max_hole_size * max_holes_size),
            min_points_count,
        )
    )
    polygons = (
        st.lists(
            x_coordinates,
            min_size=min_points_count,
            max_size=max_points_count,
            unique=True,
        ).flatmap(xs_to_polygons)
    ) | (
        st.lists(
            y_coordinates,
            min_size=min_points_count,
            max_size=max_points_count,
            unique=True,
        ).flatmap(ys_to_polygons)
    )
    if not min_holes_size:

        def multicontour_to_polygons(
            multicontour: Multicontour[ScalarT],
            polygon_cls: type[Polygon[ScalarT]] = context.polygon_cls,
        ) -> Sequence[Polygon[ScalarT]]:
            return [polygon_cls(contour, []) for contour in multicontour]

        polygons = (
            to_multicontours(
                x_coordinates,
                y_coordinates,
                min_size=min_size,
                max_size=max_size,
                min_contour_size=min_border_size,
                max_contour_size=max_border_size,
                context=context,
            ).map(multicontour_to_polygons)
        ) | polygons
    return polygons.map(context.multipolygon_cls)


def to_multisegments(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Multisegment[ScalarT]]:
    return to_non_crossing_non_overlapping_segments_sequences(
        x_coordinates,
        y_coordinates,
        min_size=min_size,
        max_size=max_size,
        context=context,
    ).map(context.multisegment_cls)


def to_non_crossing_non_overlapping_segments_sequences(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Sequence[Segment[ScalarT]]]:
    if y_coordinates is None:
        y_coordinates = x_coordinates
    if max_size is not None and max_size < 2:
        return (
            to_segments(x_coordinates, y_coordinates, context=context).map(
                lambda segment: [segment]
            )
            if max_size
            else st.builds(list)
        )
    point_cls, segment_cls = context.point_cls, context.segment_cls

    def to_vertical_multisegment(
        x: ScalarT, ys: list[ScalarT]
    ) -> Sequence[Segment[ScalarT]]:
        return [
            segment_cls(point_cls(x, y), point_cls(x, next_y))
            for y, next_y in pairwise(sorted(ys))
        ]

    def to_horizontal_multisegment(
        xs: list[ScalarT], y: ScalarT
    ) -> Sequence[Segment[ScalarT]]:
        return [
            segment_cls(point_cls(x, y), point_cls(next_x, y))
            for x, next_x in pairwise(sorted(xs))
        ]

    next_min_size, next_max_size = (
        min_size + 1,
        (max_size if max_size is None else max_size + 1),
    )
    result = st.builds(
        to_vertical_multisegment,
        x_coordinates,
        st.lists(
            y_coordinates,
            min_size=next_min_size,
            max_size=next_max_size,
            unique=True,
        ),
    ) | st.builds(
        to_horizontal_multisegment,
        st.lists(
            x_coordinates,
            min_size=next_min_size,
            max_size=next_max_size,
            unique=True,
        ),
        y_coordinates,
    )
    if max_size is None or max_size >= MIN_CONTOUR_SIZE:
        result |= (
            to_vertex_sequences(
                x_coordinates,
                y_coordinates,
                min_size=max(min_size, MIN_CONTOUR_SIZE),
                max_size=max_size,
                context=context,
            )
            .map(
                partial(
                    contour_vertices_to_edges, segment_cls=context.segment_cls
                )
            )
            .flatmap(partial(to_sub_sequence_strategy, min_size=min_size))
        )
    return result | (
        st.lists(
            to_segments(x_coordinates, y_coordinates, context=context),
            min_size=min_size,
            max_size=max_size,
        ).filter(
            partial(are_segments_non_crossing_non_overlapping, context=context)
        )
    )


def to_points(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    context: Context[ScalarT],
) -> st.SearchStrategy[Point[ScalarT]]:
    return st.builds(
        context.point_cls,
        x_coordinates,
        x_coordinates if y_coordinates is None else y_coordinates,
    )


def to_points_in_general_position(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    grid_size_lower_bound = to_next_prime(min_size)
    grid_size_upper_bound = (
        max_size if max_size is None else max(max_size, grid_size_lower_bound)
    )

    def coordinates_to_points(
        xs: Sequence[ScalarT], ys: Sequence[ScalarT]
    ) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
        xs, ys = sorted(xs), sorted(ys)
        grid_size = to_prior_prime(min(len(xs), len(ys)))
        scales = st.integers(0).flatmap(
            lambda scale: st.integers(
                scale * grid_size + 1, (scale + 1) * grid_size - 1
            )
        )
        return st.builds(
            partial(x_indices_to_points, grid_size, xs, ys), scales
        ) | st.builds(partial(y_indices_to_points, grid_size, xs, ys), scales)

    def x_indices_to_points(
        grid_size: int,
        xs: Sequence[ScalarT],
        ys: Sequence[ScalarT],
        scale: int,
        point_cls: type[Point[ScalarT]] = context.point_cls,
    ) -> Sequence[Point[ScalarT]]:
        assert are_indices_pairs_sparse(
            range(grid_size),
            [
                (scale * (index * index)) % grid_size
                for index in range(grid_size)
            ],
        )
        return [
            point_cls(xs[index], ys[(scale * (index * index)) % grid_size])
            for index in range(grid_size)
        ]

    def y_indices_to_points(
        grid_size: int,
        xs: Sequence[ScalarT],
        ys: Sequence[ScalarT],
        scale: int,
        point_cls: type[Point[ScalarT]] = context.point_cls,
    ) -> Sequence[Point[ScalarT]]:
        assert are_indices_pairs_sparse(
            range(grid_size),
            [
                (scale * (index * index)) % grid_size
                for index in range(grid_size)
            ],
        )
        return [
            point_cls(xs[(scale * (index * index)) % grid_size], ys[index])
            for index in range(grid_size)
        ]

    def are_indices_pairs_sparse(
        left_indices: Sequence[int], right_indices: Sequence[int]
    ) -> bool:
        return (
            are_indices_sparse(left_indices)
            and are_indices_sparse(right_indices)
            and all(
                are_indices_sparse(sub_indices)
                for sub_indices in zip(
                    *[
                        (
                            first_index + second_index,
                            first_index - second_index,
                        )
                        for first_index, second_index in zip(
                            left_indices, right_indices, strict=True
                        )
                    ],
                    strict=True,
                )
            )
        )

    def are_indices_sparse(indices: Sequence[int]) -> bool:
        return all(
            len(list(group)) <= 2 for _, group in groupby(sorted(indices))
        )

    return st.tuples(
        st.lists(
            x_coordinates,
            unique=True,
            min_size=grid_size_lower_bound,
            max_size=grid_size_upper_bound,
        ),
        st.lists(
            x_coordinates if y_coordinates is None else y_coordinates,
            unique=True,
            min_size=grid_size_lower_bound,
            max_size=grid_size_upper_bound,
        ),
    ).flatmap(pack(coordinates_to_points))


def to_polygons(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    min_holes_size: int,
    max_holes_size: int | None,
    min_hole_size: int,
    max_hole_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Polygon[ScalarT]]:
    def points_to_polygons(
        points: Sequence[Point[ScalarT]],
    ) -> st.SearchStrategy[Polygon[ScalarT]]:
        max_border_points_count = len(points) - min_inner_points_count
        min_border_size = max(
            min_size, len(context.points_convex_hull(points))
        )
        max_border_size = (
            max_border_points_count
            if max_size is None
            else min(max_size, max_border_points_count)
        )
        return st.builds(
            partial(to_polygon, points, context=context),
            st.integers(min_border_size, max_border_size),
            to_hole_sizes(points),
            to_choosers(),
        )

    def to_hole_sizes(
        points: Sequence[Point[ScalarT]],
    ) -> st.SearchStrategy[Sequence[int]]:
        max_inner_points_count = len(points) - len(
            to_max_convex_hull(points, context.angle_orientation)
        )
        holes_size_scale = max_inner_points_count // min_hole_size
        points_max_hole_size = (
            holes_size_scale
            if max_holes_size is None
            else min(max_holes_size, holes_size_scale)
        )
        return (
            (
                st.integers(min_holes_size, points_max_hole_size).flatmap(
                    partial(
                        _to_hole_sizes,
                        min_hole_points_count=min_hole_size,
                        max_hole_points_count=max_inner_points_count,
                    )
                )
            )
            if max_inner_points_count >= min_hole_size
            else st.builds(list)
        )

    def _to_hole_sizes(
        holes_size: int, min_hole_points_count: int, max_hole_points_count: int
    ) -> st.SearchStrategy[Sequence[int]]:
        if not holes_size:
            return st.builds(list)
        max_holes_points_counts = [min_hole_points_count] * holes_size
        indices = cycle(range(holes_size))
        for _ in range(
            max_hole_points_count - holes_size * min_hole_points_count
        ):
            max_holes_points_counts[next(indices)] += 1
        sizes_ranges = [
            range(min_hole_points_count, max_hole_points_count + 1)
            for max_hole_points_count in max_holes_points_counts
        ]
        return (
            st.permutations(
                [st.sampled_from(sizes_range) for sizes_range in sizes_ranges]
            )
            .flatmap(pack(st.tuples))
            .map(list)
        )

    def has_valid_sizes(polygon: Polygon[ScalarT]) -> bool:
        return has_valid_size(
            polygon.border.vertices, min_size=min_size, max_size=max_size
        ) and multicontour_has_valid_sizes(
            polygon.holes,
            min_size=min_holes_size,
            max_size=max_holes_size,
            min_contour_size=min_hole_size,
            max_contour_size=max_hole_size,
        )

    min_inner_points_count = min_hole_size * min_holes_size

    def has_valid_inner_points_count(points: Sequence[Point[ScalarT]]) -> bool:
        return (
            max_size is None
            or len(context.points_convex_hull(points)) <= max_size
        ) and (
            len(points)
            - len(to_max_convex_hull(points, context.angle_orientation))
            >= min_inner_points_count
        )

    min_points_count = min_size + min_inner_points_count
    max_points_count = (
        None
        if (
            max_size is None or max_holes_size is None or max_hole_size is None
        )
        else max_size + max_hole_size * max_holes_size
    )
    return (
        to_points_in_general_position(
            x_coordinates,
            y_coordinates,
            min_size=min_points_count,
            max_size=max_points_count,
            context=context,
        )
        .filter(has_valid_inner_points_count)
        .flatmap(points_to_polygons)
        .filter(has_valid_sizes)
    )


def to_rectangular_vertex_sequences(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    context: Context[ScalarT],
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    def to_vertices(
        box: Box[ScalarT], point_cls: type[Point[ScalarT]] = context.point_cls
    ) -> Sequence[Point[ScalarT]]:
        return [
            point_cls(box.min_x, box.min_y),
            point_cls(box.max_x, box.min_y),
            point_cls(box.max_x, box.max_y),
            point_cls(box.min_x, box.max_y),
        ]

    return to_boxes(x_coordinates, y_coordinates, context=context).map(
        to_vertices
    )


def to_segments(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    context: Context[ScalarT],
) -> st.SearchStrategy[Segment[ScalarT]]:
    def non_degenerate_endpoints(
        endpoints: tuple[Point[ScalarT], Point[ScalarT]],
    ) -> bool:
        start, end = endpoints
        return start != end

    points = to_points(x_coordinates, y_coordinates, context=context)
    return (
        st.tuples(points, points)
        .filter(non_degenerate_endpoints)
        .map(pack(context.segment_cls))
    )


def to_star_vertex_sequences(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    return (
        to_points_in_general_position(
            x_coordinates,
            y_coordinates,
            min_size=min_size,
            max_size=max_size,
            context=context,
        )
        .map(partial(to_star_contour_vertices, context=context))
        .filter(partial(has_valid_size, min_size=min_size, max_size=max_size))
    )


def to_sub_sequence_strategy(
    values: Sequence[Domain], *, min_size: int
) -> st.SearchStrategy[Sequence[Domain]]:
    return st.builds(
        cut, st.permutations(values), st.integers(min_size, len(values))
    )


def to_triangular_vertex_sequences(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    context: Context[ScalarT],
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    def to_counterclockwise_vertices(
        vertices_triplet: tuple[
            Point[ScalarT], Point[ScalarT], Point[ScalarT]
        ],
        orienteer: Orienteer[ScalarT] = context.angle_orientation,
    ) -> tuple[Point[ScalarT], Point[ScalarT], Point[ScalarT]]:
        vertex, first_ray_point, second_ray_point = vertices_triplet
        return (
            vertices_triplet
            if (
                orienteer(vertex, first_ray_point, second_ray_point)
                is Orientation.COUNTERCLOCKWISE
            )
            else (second_ray_point, first_ray_point, vertex)
        )

    vertices = to_points(x_coordinates, y_coordinates, context=context)
    return (
        st.tuples(vertices, vertices, vertices)
        .filter(
            partial(are_vertices_strict, orienteer=context.angle_orientation)
        )
        .map(to_counterclockwise_vertices)
        .map(list)
    )


def to_unique_points_sequences(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    return st.lists(
        to_points(x_coordinates, y_coordinates, context=context),
        unique=True,
        min_size=min_size,
        max_size=max_size,
    )


def to_vertex_sequences(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    return to_convex_vertex_sequences(
        x_coordinates,
        y_coordinates,
        min_size=min_size,
        max_size=max_size,
        context=context,
    ) | _to_vertex_sequences(
        x_coordinates,
        y_coordinates,
        min_size=min_size,
        max_size=max_size,
        context=context,
    )


def _to_vertex_sequences(
    x_coordinates: st.SearchStrategy[ScalarT],
    y_coordinates: st.SearchStrategy[ScalarT] | None,
    *,
    min_size: int,
    max_size: int | None,
    context: Context[ScalarT],
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    def to_points_with_sizes(
        points: Sequence[Point[ScalarT]],
    ) -> st.SearchStrategy[tuple[Sequence[Point[ScalarT]], int]]:
        sizes = st.integers(
            min_size,
            len(points) if max_size is None else min(len(points), max_size),
        )
        return st.tuples(st.just(points), sizes)

    return to_star_vertex_sequences(
        x_coordinates,
        y_coordinates,
        min_size=min_size,
        max_size=max_size,
        context=context,
    ) | (
        to_points_in_general_position(
            x_coordinates,
            y_coordinates,
            min_size=min_size,
            max_size=max_size,
            context=context,
        )
        .flatmap(to_points_with_sizes)
        .map(pack(partial(to_vertex_sequence, context=context)))
        .filter(partial(has_valid_size, min_size=min_size, max_size=max_size))
    )
