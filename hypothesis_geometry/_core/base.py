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


def to_box_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
) -> st.SearchStrategy[Box[ScalarT]]:
    return st.builds(
        add,
        st.lists(
            x_coordinate_strategy, min_size=2, max_size=2, unique=True
        ).map(sort_pair),
        st.lists(
            (
                x_coordinate_strategy
                if y_coordinate_strategy is None
                else y_coordinate_strategy
            ),
            min_size=2,
            max_size=2,
            unique=True,
        ).map(sort_pair),
    ).map(pack(context.box_cls))


def to_chooser_strategy() -> st.SearchStrategy[Chooser[Any]]:
    return st.randoms(use_true_random=True).map(lambda random: random.choice)


def to_concave_vertex_sequence_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    return _to_vertex_sequences(
        x_coordinate_strategy,
        y_coordinate_strategy,
        min_size=min_size,
        max_size=max_size,
        context=context,
    ).filter(
        partial(are_vertices_non_convex, orienteer=context.angle_orientation)
    )


def to_convex_vertex_sequence_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    if max_size is not None and max_size < MinContourSize.CONCAVE:
        return to_triangular_vertex_sequence_strategy(
            x_coordinate_strategy, y_coordinate_strategy, context=context
        )
    result = st.builds(
        partial(to_convex_vertex_sequence, context=context),
        to_point_sequence_in_general_position_strategy(
            x_coordinate_strategy,
            y_coordinate_strategy,
            min_size=min_size,
            max_size=max_size,
            context=context,
        ),
        st.randoms(use_true_random=True),
    ).filter(partial(has_valid_size, min_size=min_size, max_size=max_size))
    result = (
        (
            to_rectangular_vertex_sequence_strategy(
                x_coordinate_strategy, y_coordinate_strategy, context=context
            )
            | result
        )
        if min_size <= 4
        else result
    )
    return (
        (
            to_triangular_vertex_sequence_strategy(
                x_coordinate_strategy, y_coordinate_strategy, context=context
            )
            | result
        )
        if min_size == 3
        else result
    )


def to_empty_geometry_strategy(
    *, context: Context[ScalarT]
) -> st.SearchStrategy[Empty[ScalarT]]:
    return st.builds(context.empty_cls)


_Draw: TypeAlias = Callable[[st.SearchStrategy[Any]], Any]


def to_mix_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_point_count: int,
    max_point_count: int | None,
    min_segment_count: int,
    max_segment_count: int | None,
    min_polygon_count: int,
    max_polygon_count: int | None,
    min_polygon_border_size: int,
    max_polygon_border_size: int | None,
    min_polygon_hole_count: int,
    max_polygon_hole_count: int | None,
    min_polygon_hole_size: int,
    max_polygon_hole_size: int | None,
) -> st.SearchStrategy[Mix[ScalarT]]:
    if y_coordinate_strategy is None:
        y_coordinate_strategy = x_coordinate_strategy
    min_polygon_point_count = to_next_prime(
        min_polygon_border_size
        + (min_polygon_hole_count * min_polygon_hole_size)
    )
    min_polygon_total_point_count = min_polygon_count * min_polygon_point_count
    min_segment_total_point_count = 2 * min_segment_count
    min_total_point_count = (
        min_point_count
        + min_segment_total_point_count
        + min_polygon_total_point_count
    )
    empty = context.empty
    mix_cls = context.mix_cls
    multipoint_cls = context.multipoint_cls
    multipolygon_cls = context.multipolygon_cls
    multisegment_cls = context.multisegment_cls

    @st.composite
    def xs_to_mix(draw: _Draw, xs: list[ScalarT], /) -> Mix[ScalarT]:
        (point_counts, segment_endpoint_counts, polygon_vertex_counts) = (
            to_point_counts(draw, len(xs))
        )
        xs = sorted(xs)
        points, segments, polygons = [], [], []

        def draw_points(point_count: int, /) -> None:
            points.extend(
                draw(
                    to_unique_point_sequence_strategy(
                        st.sampled_from(xs[:point_count]),
                        y_coordinate_strategy,
                        min_size=point_count,
                        max_size=point_count,
                        context=context,
                    )
                )
            )

        def draw_segments(point_count: int, /) -> None:
            size = point_count // 2
            segments.extend(
                draw(
                    to_non_crossing_non_overlapping_segment_sequence_strategy(
                        st.sampled_from(xs[:point_count]),
                        y_coordinate_strategy,
                        min_size=size,
                        max_size=size,
                        context=context,
                    )
                )
            )

        def draw_polygon(point_count: int, /) -> None:
            polygon_x_coordinate_strategy = st.sampled_from(
                xs[: to_prior_prime(point_count)]
            )
            polygons.append(
                draw(
                    to_polygon_strategy(
                        polygon_x_coordinate_strategy,
                        y_coordinate_strategy,
                        context=context,
                        min_size=min_polygon_border_size,
                        max_size=max_polygon_border_size,
                        min_hole_count=min_polygon_hole_count,
                        max_hole_count=max_polygon_hole_count,
                        min_hole_size=min_polygon_hole_size,
                        max_hole_size=max_polygon_hole_size,
                    )
                )
            )

        drawers_with_point_counts = draw(
            st.permutations(
                tuple(
                    chain(
                        zip(repeat(draw_points), point_counts),
                        zip(repeat(draw_segments), segment_endpoint_counts),
                        zip(repeat(draw_polygon), polygon_vertex_counts),
                    )
                )
            )
        )
        to_contour_segments = context.contour_segments
        for index, (drawer, count) in enumerate(drawers_with_point_counts):
            drawer(count)
            can_touch_next_geometry = (
                index < len(drawers_with_point_counts) - 1
                and (
                    drawers_with_point_counts[index + 1][0] is not draw_points
                )
            ) and (
                (
                    drawer is draw_segments
                    and index < len(drawers_with_point_counts) - 1
                    and (
                        drawers_with_point_counts[index + 1] is not draw_points
                    )
                    and not has_vertical_leftmost_segment(segments)
                )
                or (
                    drawer is draw_polygon
                    and index < len(drawers_with_point_counts) - 1
                    and (
                        drawers_with_point_counts[index + 1] is not draw_points
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
    def ys_to_mix(draw: _Draw, ys: list[ScalarT], /) -> Mix[ScalarT]:
        (point_counts, segment_endpoint_counts, polygon_vertex_counts) = (
            to_point_counts(draw, len(ys))
        )
        ys = sorted(ys)
        points, segments, polygons = [], [], []

        def draw_points(point_count: int, /) -> None:
            points.extend(
                draw(
                    to_unique_point_sequence_strategy(
                        x_coordinate_strategy,
                        st.sampled_from(ys[:point_count]),
                        min_size=point_count,
                        max_size=point_count,
                        context=context,
                    )
                )
            )

        def draw_segments(point_count: int, /) -> None:
            size = point_count // 2
            segments.extend(
                draw(
                    to_non_crossing_non_overlapping_segment_sequence_strategy(
                        x_coordinate_strategy,
                        st.sampled_from(ys[:point_count]),
                        min_size=size,
                        max_size=size,
                        context=context,
                    )
                )
            )

        def draw_polygon(point_count: int, /) -> None:
            polygons.append(
                draw(
                    to_polygon_strategy(
                        x_coordinate_strategy,
                        st.sampled_from(ys[:point_count]),
                        min_size=min_polygon_border_size,
                        max_size=max_polygon_border_size,
                        min_hole_count=min_polygon_hole_count,
                        max_hole_count=max_polygon_hole_count,
                        min_hole_size=min_polygon_hole_size,
                        max_hole_size=max_polygon_hole_size,
                        context=context,
                    )
                )
            )

        drawers_with_point_counts = draw(
            st.permutations(
                tuple(
                    chain(
                        zip(repeat(draw_points), point_counts),
                        zip(repeat(draw_segments), segment_endpoint_counts),
                        zip(repeat(draw_polygon), polygon_vertex_counts),
                    )
                )
            )
        )
        to_contour_segments = context.contour_segments
        for index, (drawer, count) in enumerate(drawers_with_point_counts):
            drawer(count)
            can_touch_next_geometry = (
                index < len(drawers_with_point_counts) - 1
                and (
                    drawers_with_point_counts[index + 1][0] is not draw_points
                )
            ) and (
                (
                    drawer is draw_segments
                    and index < len(drawers_with_point_counts) - 1
                    and (
                        drawers_with_point_counts[index + 1] is not draw_points
                    )
                    and not has_horizontal_lowermost_segment(segments)
                )
                or (
                    drawer is draw_polygon
                    and index < len(drawers_with_point_counts) - 1
                    and (
                        drawers_with_point_counts[index + 1] is not draw_points
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
        points: Sequence[Point[ScalarT]], /
    ) -> Multipoint[ScalarT] | Empty[ScalarT]:
        return multipoint_cls(points) if points else empty

    def unpack_segments(
        segments: Sequence[Segment[ScalarT]], /
    ) -> Linear[ScalarT] | Empty[ScalarT]:
        return (
            (multisegment_cls(segments) if len(segments) > 1 else segments[0])
            if segments
            else empty
        )

    def unpack_polygons(
        polygons: Sequence[Polygon[ScalarT]], /
    ) -> Shaped[ScalarT] | Empty[ScalarT]:
        return (
            (multipolygon_cls(polygons) if len(polygons) > 1 else polygons[0])
            if polygons
            else empty
        )

    def to_point_counts(
        draw: _Draw, max_total_point_count: int, /
    ) -> tuple[list[int], list[int], list[int]]:
        max_polygon_total_point_count = (
            max_total_point_count
            - min_point_count
            - min_segment_total_point_count
        )
        polygon_size_upper_bound = (
            max_polygon_total_point_count // min_polygon_point_count
        )
        polygon_count = draw(
            st.integers(
                min_polygon_count,
                (
                    polygon_size_upper_bound
                    if max_polygon_count is None
                    else min(polygon_size_upper_bound, max_polygon_count)
                ),
            )
        )
        max_polygon_point_count = (
            to_prior_prime(max_polygon_total_point_count // polygon_count)
            if polygon_count
            else 0
        )
        polygon_point_counts = (
            [
                draw(polygon_point_count_strategy)
                for polygon_point_count_strategy in repeat(
                    st.integers(
                        min_polygon_point_count, max_polygon_point_count
                    ),
                    polygon_count,
                )
            ]
            if polygon_count
            else []
        )
        total_polygon_point_count = sum(polygon_point_counts)
        segment_endpoint_count_upper_bound = (
            max_total_point_count - total_polygon_point_count - min_point_count
        )
        max_segment_point_count = (
            segment_endpoint_count_upper_bound
            if max_segment_count is None
            else min(segment_endpoint_count_upper_bound, 2 * max_segment_count)
        )
        segment_point_count = draw(
            st.sampled_from(
                range(
                    min_segment_total_point_count,
                    max_segment_point_count + 1,
                    2,
                )
            )
        )
        point_count_upper_bound = (
            max_total_point_count
            - segment_point_count
            - total_polygon_point_count
        )
        point_count = (
            point_count_upper_bound
            if max_point_count is None
            else min(point_count_upper_bound, max_point_count)
        )
        segment_count = segment_point_count // 2
        return (
            partition(draw, point_count),
            [2 * size for size in partition(draw, segment_count)],
            polygon_point_counts,
        )

    def partition(draw: _Draw, value: int, /) -> list[int]:
        assert value >= 0, 'Value should be non-negative.'
        result = []
        while value:
            part = draw(st.integers(1, value))
            result.append(part)
            value -= part
        return result

    return (
        st.lists(
            x_coordinate_strategy, min_size=min_total_point_count, unique=True
        ).flatmap(xs_to_mix)
    ) | (
        st.lists(
            y_coordinate_strategy, min_size=min_total_point_count, unique=True
        ).flatmap(ys_to_mix)
    )


def to_multicontour_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
    min_contour_size: int,
    max_contour_size: int | None,
) -> st.SearchStrategy[Multicontour[ScalarT]]:
    def point_sequence_to_multicontour_strategy(
        points: Sequence[Point[ScalarT]], /
    ) -> st.SearchStrategy[Multicontour[ScalarT]]:
        return st.builds(
            partial(to_multicontour, points, context=context),
            to_size_strategy(len(points)),
            to_chooser_strategy(),
        )

    def to_size_strategy(limit: int, /) -> st.SearchStrategy[list[int]]:
        size_upper_bound = limit // min_contour_size
        return st.integers(
            min_size,
            (
                size_upper_bound
                if max_size is None
                else min(size_upper_bound, max_size)
            ),
        ).flatmap(
            partial(_to_sizes, min_element_size=min_contour_size, limit=limit)
        )

    def _to_sizes(
        size: int, /, *, min_element_size: int, limit: int
    ) -> st.SearchStrategy[list[int]]:
        max_sizes = [min_element_size] * size
        indices = cycle(range(size))
        for _ in range(limit - size * min_element_size):
            max_sizes[next(indices)] += 1
        size_ranges = [
            range(min_element_size, max_element_size + 1)
            for max_element_size in max_sizes
        ]
        return st.tuples(
            *[st.sampled_from(size_range) for size_range in size_ranges]
        ).flatmap(st.permutations)

    min_total_point_count = min_size * min_contour_size
    max_total_point_count = (
        None
        if max_size is None or max_contour_size is None
        else max_size * max_contour_size
    )
    return (
        to_point_sequence_in_general_position_strategy(
            x_coordinate_strategy,
            y_coordinate_strategy,
            min_size=min_total_point_count,
            max_size=max_total_point_count,
            context=context,
        )
        .flatmap(point_sequence_to_multicontour_strategy)
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


def to_multipoint_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
) -> st.SearchStrategy[Multipoint[ScalarT]]:
    return to_unique_point_sequence_strategy(
        x_coordinate_strategy,
        y_coordinate_strategy,
        min_size=min_size,
        max_size=max_size,
        context=context,
    ).map(context.multipoint_cls)


def to_multipolygon_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
    min_border_size: int,
    max_border_size: int | None,
    min_hole_count: int,
    max_hole_count: int | None,
    min_hole_size: int,
    max_hole_size: int | None,
) -> st.SearchStrategy[Multipolygon[ScalarT]]:
    if y_coordinate_strategy is None:
        y_coordinate_strategy = x_coordinate_strategy
    min_polygon_point_count = to_next_prime(
        min_border_size + min_hole_count * min_hole_size
    )

    @st.composite
    def xs_to_polygons(
        draw: _Draw, xs: list[ScalarT], /
    ) -> Sequence[Polygon[ScalarT]]:
        size_upper_bound = len(xs) // min_polygon_point_count
        size = draw(
            st.integers(
                min_size,
                (
                    size_upper_bound
                    if max_size is None
                    else min(max_size, size_upper_bound)
                ),
            )
        )
        xs = sorted(xs)
        to_contour_segments = context.contour_segments
        result = []
        start, coordinate_count = 0, len(xs)
        for index in range(size - 1):
            polygon_point_count = draw(
                st.integers(
                    min_polygon_point_count,
                    (coordinate_count - start) // (size - index),
                )
            )
            polygon_xs = xs[
                start : start + to_prior_prime(polygon_point_count)
            ]
            polygon = draw(
                to_polygon_strategy(
                    st.sampled_from(polygon_xs),
                    y_coordinate_strategy,
                    min_size=min_border_size,
                    max_size=max_border_size,
                    min_hole_count=min_hole_count,
                    max_hole_count=max_hole_count,
                    min_hole_size=min_hole_size,
                    max_hole_size=max_hole_size,
                    context=context,
                )
            )
            result.append(polygon)
            can_touch_next_polygon = not has_vertical_leftmost_segment(
                to_contour_segments(polygon.border)
            )
            start += polygon_point_count - can_touch_next_polygon
        result.append(
            draw(
                to_polygon_strategy(
                    st.sampled_from(xs[start:]),
                    y_coordinate_strategy,
                    min_size=min_border_size,
                    max_size=max_border_size,
                    min_hole_count=min_hole_count,
                    max_hole_count=max_hole_count,
                    min_hole_size=min_hole_size,
                    max_hole_size=max_hole_size,
                    context=context,
                )
            )
        )
        return result

    @st.composite
    def ys_to_polygons(
        draw: _Draw, ys: list[ScalarT], /
    ) -> Sequence[Polygon[ScalarT]]:
        size_upper_bound = len(ys) // min_polygon_point_count
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
        start, coordinate_count = 0, len(ys)
        for index in range(size - 1):
            polygon_point_count = draw(
                st.integers(
                    min_polygon_point_count,
                    (coordinate_count - start) // (size - index),
                )
            )
            polygon_ys = ys[
                start : start + to_prior_prime(polygon_point_count)
            ]
            polygon = draw(
                to_polygon_strategy(
                    x_coordinate_strategy,
                    st.sampled_from(polygon_ys),
                    min_size=min_border_size,
                    max_size=max_border_size,
                    min_hole_count=min_hole_count,
                    max_hole_count=max_hole_count,
                    min_hole_size=min_hole_size,
                    max_hole_size=max_hole_size,
                    context=context,
                )
            )
            result.append(polygon)
            can_touch_next_polygon = not has_horizontal_lowermost_segment(
                to_contour_segments(polygon.border)
            )
            start += polygon_point_count - can_touch_next_polygon
        result.append(
            draw(
                to_polygon_strategy(
                    x_coordinate_strategy,
                    st.sampled_from(ys[start:]),
                    min_size=min_border_size,
                    max_size=max_border_size,
                    min_hole_count=min_hole_count,
                    max_hole_count=max_hole_count,
                    min_hole_size=min_hole_size,
                    max_hole_size=max_hole_size,
                    context=context,
                )
            )
        )
        return result

    min_total_point_count = min_size * min_polygon_point_count
    max_total_point_count = (
        None
        if (
            max_size is None
            or max_border_size is None
            or max_hole_count is None
            or max_hole_size is None
        )
        else max(
            max_size * (max_border_size + max_hole_size * max_hole_count),
            min_total_point_count,
        )
    )
    polygon_strategy = (
        st.lists(
            x_coordinate_strategy,
            min_size=min_total_point_count,
            max_size=max_total_point_count,
            unique=True,
        ).flatmap(xs_to_polygons)
    ) | (
        st.lists(
            y_coordinate_strategy,
            min_size=min_total_point_count,
            max_size=max_total_point_count,
            unique=True,
        ).flatmap(ys_to_polygons)
    )
    if min_hole_count == 0:

        def multicontour_to_polygons(
            multicontour: Multicontour[ScalarT],
            /,
            *,
            polygon_cls: type[Polygon[ScalarT]] = context.polygon_cls,
        ) -> Sequence[Polygon[ScalarT]]:
            return [polygon_cls(contour, []) for contour in multicontour]

        polygon_strategy = (
            to_multicontour_strategy(
                x_coordinate_strategy,
                y_coordinate_strategy,
                min_size=min_size,
                max_size=max_size,
                min_contour_size=min_border_size,
                max_contour_size=max_border_size,
                context=context,
            ).map(multicontour_to_polygons)
        ) | polygon_strategy
    return polygon_strategy.map(context.multipolygon_cls)


def to_multisegment_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
) -> st.SearchStrategy[Multisegment[ScalarT]]:
    return to_non_crossing_non_overlapping_segment_sequence_strategy(
        x_coordinate_strategy,
        y_coordinate_strategy,
        min_size=min_size,
        max_size=max_size,
        context=context,
    ).map(context.multisegment_cls)


def to_non_crossing_non_overlapping_segment_sequence_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
) -> st.SearchStrategy[Sequence[Segment[ScalarT]]]:
    if y_coordinate_strategy is None:
        y_coordinate_strategy = x_coordinate_strategy
    if max_size is not None and max_size < 2:
        return (
            to_segment_strategy(
                x_coordinate_strategy, y_coordinate_strategy, context=context
            ).map(lambda segment: [segment])
            if max_size
            else st.builds(list)
        )
    point_cls, segment_cls = context.point_cls, context.segment_cls

    def to_vertical_multisegment(
        x: ScalarT, ys: list[ScalarT], /
    ) -> Sequence[Segment[ScalarT]]:
        return [
            segment_cls(point_cls(x, y), point_cls(x, next_y))
            for y, next_y in pairwise(sorted(ys))
        ]

    def to_horizontal_multisegment(
        xs: list[ScalarT], y: ScalarT, /
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
        x_coordinate_strategy,
        st.lists(
            y_coordinate_strategy,
            min_size=next_min_size,
            max_size=next_max_size,
            unique=True,
        ),
    ) | st.builds(
        to_horizontal_multisegment,
        st.lists(
            x_coordinate_strategy,
            min_size=next_min_size,
            max_size=next_max_size,
            unique=True,
        ),
        y_coordinate_strategy,
    )
    if max_size is None or max_size >= MIN_CONTOUR_SIZE:
        result |= (
            to_vertex_sequence_strategy(
                x_coordinate_strategy,
                y_coordinate_strategy,
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
            to_segment_strategy(
                x_coordinate_strategy, y_coordinate_strategy, context=context
            ),
            min_size=min_size,
            max_size=max_size,
        ).filter(
            partial(are_segments_non_crossing_non_overlapping, context=context)
        )
    )


def to_point_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
) -> st.SearchStrategy[Point[ScalarT]]:
    return st.builds(
        context.point_cls,
        x_coordinate_strategy,
        (
            x_coordinate_strategy
            if y_coordinate_strategy is None
            else y_coordinate_strategy
        ),
    )


def to_point_sequence_in_general_position_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    grid_size_lower_bound = to_next_prime(min_size)
    grid_size_upper_bound = (
        max_size if max_size is None else max(max_size, grid_size_lower_bound)
    )

    def coordinate_sequence_pair_to_point_sequence_strategy(
        xs: Sequence[ScalarT], ys: Sequence[ScalarT], /
    ) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
        xs, ys = sorted(xs), sorted(ys)
        grid_size = to_prior_prime(min(len(xs), len(ys)))
        scale_strategy = st.integers(0).flatmap(
            lambda scale: st.integers(
                scale * grid_size + 1, (scale + 1) * grid_size - 1
            )
        )
        return st.builds(
            partial(x_indices_to_points, grid_size, xs, ys), scale_strategy
        ) | st.builds(
            partial(y_indices_to_points, grid_size, xs, ys), scale_strategy
        )

    def x_indices_to_points(
        grid_size: int,
        xs: Sequence[ScalarT],
        ys: Sequence[ScalarT],
        scale: int,
        /,
        *,
        point_cls: type[Point[ScalarT]] = context.point_cls,
    ) -> Sequence[Point[ScalarT]]:
        assert are_index_sequence_pair_sparse(
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
        /,
        *,
        point_cls: type[Point[ScalarT]] = context.point_cls,
    ) -> Sequence[Point[ScalarT]]:
        assert are_index_sequence_pair_sparse(
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

    def are_index_sequence_pair_sparse(
        left_indices: Sequence[int], right_indices: Sequence[int], /
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

    def are_indices_sparse(indices: Sequence[int], /) -> bool:
        return all(
            len(list(group)) <= 2 for _, group in groupby(sorted(indices))
        )

    return st.tuples(
        st.lists(
            x_coordinate_strategy,
            unique=True,
            min_size=grid_size_lower_bound,
            max_size=grid_size_upper_bound,
        ),
        st.lists(
            (
                x_coordinate_strategy
                if y_coordinate_strategy is None
                else y_coordinate_strategy
            ),
            unique=True,
            min_size=grid_size_lower_bound,
            max_size=grid_size_upper_bound,
        ),
    ).flatmap(pack(coordinate_sequence_pair_to_point_sequence_strategy))


def to_polygon_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
    min_hole_count: int,
    max_hole_count: int | None,
    min_hole_size: int,
    max_hole_size: int | None,
) -> st.SearchStrategy[Polygon[ScalarT]]:
    def point_sequence_to_polygon_strategy(
        points: Sequence[Point[ScalarT]], /
    ) -> st.SearchStrategy[Polygon[ScalarT]]:
        max_border_point_count = len(points) - min_inner_point_count
        min_border_size = max(
            min_size, len(context.points_convex_hull(points))
        )
        max_border_size = (
            max_border_point_count
            if max_size is None
            else min(max_size, max_border_point_count)
        )
        return st.builds(
            partial(to_polygon, points, context=context),
            st.integers(min_border_size, max_border_size),
            to_hole_sizes(points),
            to_chooser_strategy(),
        )

    def to_hole_sizes(
        points: Sequence[Point[ScalarT]], /
    ) -> st.SearchStrategy[Sequence[int]]:
        max_inner_point_count = len(points) - len(
            to_max_convex_hull(points, orienteer=context.angle_orientation)
        )
        hole_count_scale = max_inner_point_count // min_hole_size
        point_max_hole_size = (
            hole_count_scale
            if max_hole_count is None
            else min(max_hole_count, hole_count_scale)
        )
        return (
            (
                st.integers(min_hole_count, point_max_hole_size).flatmap(
                    partial(
                        _to_hole_sizes,
                        min_hole_point_count=min_hole_size,
                        max_hole_point_count=max_inner_point_count,
                    )
                )
            )
            if max_inner_point_count >= min_hole_size
            else st.builds(list)
        )

    def _to_hole_sizes(
        hole_count: int,
        /,
        *,
        min_hole_point_count: int,
        max_hole_point_count: int,
    ) -> st.SearchStrategy[Sequence[int]]:
        if hole_count == 0:
            return st.builds(list)
        max_hole_point_counts = [min_hole_point_count] * hole_count
        indices = cycle(range(hole_count))
        for _ in range(
            max_hole_point_count - hole_count * min_hole_point_count
        ):
            max_hole_point_counts[next(indices)] += 1
        size_ranges = [
            range(min_hole_point_count, max_hole_point_count + 1)
            for max_hole_point_count in max_hole_point_counts
        ]
        return (
            st.permutations(
                [st.sampled_from(size_range) for size_range in size_ranges]
            )
            .flatmap(pack(st.tuples))
            .map(list)
        )

    def has_valid_sizes(polygon: Polygon[ScalarT], /) -> bool:
        return has_valid_size(
            polygon.border.vertices, min_size=min_size, max_size=max_size
        ) and multicontour_has_valid_sizes(
            polygon.holes,
            min_size=min_hole_count,
            max_size=max_hole_count,
            min_contour_size=min_hole_size,
            max_contour_size=max_hole_size,
        )

    min_inner_point_count = min_hole_size * min_hole_count

    def has_valid_inner_point_count(
        points: Sequence[Point[ScalarT]], /
    ) -> bool:
        return (
            max_size is None
            or len(context.points_convex_hull(points)) <= max_size
        ) and (
            len(points)
            - len(
                to_max_convex_hull(points, orienteer=context.angle_orientation)
            )
            >= min_inner_point_count
        )

    min_point_count = min_size + min_inner_point_count
    max_point_count = (
        None
        if (
            max_size is None or max_hole_count is None or max_hole_size is None
        )
        else max_size + max_hole_size * max_hole_count
    )
    return (
        to_point_sequence_in_general_position_strategy(
            x_coordinate_strategy,
            y_coordinate_strategy,
            min_size=min_point_count,
            max_size=max_point_count,
            context=context,
        )
        .filter(has_valid_inner_point_count)
        .flatmap(point_sequence_to_polygon_strategy)
        .filter(has_valid_sizes)
    )


def to_rectangular_vertex_sequence_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    def to_vertices(
        box: Box[ScalarT],
        /,
        *,
        point_cls: type[Point[ScalarT]] = context.point_cls,
    ) -> Sequence[Point[ScalarT]]:
        return [
            point_cls(box.min_x, box.min_y),
            point_cls(box.max_x, box.min_y),
            point_cls(box.max_x, box.max_y),
            point_cls(box.min_x, box.max_y),
        ]

    return to_box_strategy(
        x_coordinate_strategy, y_coordinate_strategy, context=context
    ).map(to_vertices)


def to_segment_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
) -> st.SearchStrategy[Segment[ScalarT]]:
    def are_endpoints_non_degenerate(
        endpoints: tuple[Point[ScalarT], Point[ScalarT]], /
    ) -> bool:
        start, end = endpoints
        return start != end

    point_strategy = to_point_strategy(
        x_coordinate_strategy, y_coordinate_strategy, context=context
    )
    return (
        st.tuples(point_strategy, point_strategy)
        .filter(are_endpoints_non_degenerate)
        .map(pack(context.segment_cls))
    )


def to_star_vertex_sequence_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    return (
        to_point_sequence_in_general_position_strategy(
            x_coordinate_strategy,
            y_coordinate_strategy,
            min_size=min_size,
            max_size=max_size,
            context=context,
        )
        .map(partial(to_star_contour_vertices, context=context))
        .filter(partial(has_valid_size, min_size=min_size, max_size=max_size))
    )


def to_sub_sequence_strategy(
    values: Sequence[Domain], /, *, min_size: int
) -> st.SearchStrategy[Sequence[Domain]]:
    return st.builds(
        cut, st.permutations(values), st.integers(min_size, len(values))
    )


def to_triangular_vertex_sequence_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    def to_counterclockwise_vertices(
        vertex_triplet: tuple[Point[ScalarT], Point[ScalarT], Point[ScalarT]],
        /,
        *,
        orienteer: Orienteer[ScalarT] = context.angle_orientation,
    ) -> tuple[Point[ScalarT], Point[ScalarT], Point[ScalarT]]:
        vertex, first_ray_point, second_ray_point = vertex_triplet
        return (
            vertex_triplet
            if (
                orienteer(vertex, first_ray_point, second_ray_point)
                is Orientation.COUNTERCLOCKWISE
            )
            else (second_ray_point, first_ray_point, vertex)
        )

    vertex_strategy = to_point_strategy(
        x_coordinate_strategy, y_coordinate_strategy, context=context
    )
    return (
        st.tuples(vertex_strategy, vertex_strategy, vertex_strategy)
        .filter(
            partial(are_vertices_strict, orienteer=context.angle_orientation)
        )
        .map(to_counterclockwise_vertices)
        .map(list)
    )


def to_unique_point_sequence_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    return st.lists(
        to_point_strategy(
            x_coordinate_strategy, y_coordinate_strategy, context=context
        ),
        unique=True,
        min_size=min_size,
        max_size=max_size,
    )


def to_vertex_sequence_strategy(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    return to_convex_vertex_sequence_strategy(
        x_coordinate_strategy,
        y_coordinate_strategy,
        min_size=min_size,
        max_size=max_size,
        context=context,
    ) | _to_vertex_sequences(
        x_coordinate_strategy,
        y_coordinate_strategy,
        min_size=min_size,
        max_size=max_size,
        context=context,
    )


def _to_vertex_sequences(
    x_coordinate_strategy: st.SearchStrategy[ScalarT],
    y_coordinate_strategy: st.SearchStrategy[ScalarT] | None,
    /,
    *,
    context: Context[ScalarT],
    min_size: int,
    max_size: int | None,
) -> st.SearchStrategy[Sequence[Point[ScalarT]]]:
    def to_point_sequence_with_size_strategy(
        points: Sequence[Point[ScalarT]], /
    ) -> st.SearchStrategy[tuple[Sequence[Point[ScalarT]], int]]:
        size_strategy = st.integers(
            min_size,
            len(points) if max_size is None else min(len(points), max_size),
        )
        return st.tuples(st.just(points), size_strategy)

    return to_star_vertex_sequence_strategy(
        x_coordinate_strategy,
        y_coordinate_strategy,
        min_size=min_size,
        max_size=max_size,
        context=context,
    ) | (
        to_point_sequence_in_general_position_strategy(
            x_coordinate_strategy,
            y_coordinate_strategy,
            min_size=min_size,
            max_size=max_size,
            context=context,
        )
        .flatmap(to_point_sequence_with_size_strategy)
        .map(pack(partial(to_vertex_sequence, context=context)))
        .filter(partial(has_valid_size, min_size=min_size, max_size=max_size))
    )
