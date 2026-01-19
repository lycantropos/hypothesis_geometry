from collections.abc import Callable, Hashable, Iterable, Sequence
from functools import partial
from itertools import chain
from typing import Any, TypeAlias, TypeVar

from bentley_ottmann.planar import contour_self_intersects
from ground.context import get_context
from ground.enums import Location, Orientation
from ground.hints import (
    Box,
    Contour,
    Mix,
    Multipoint,
    Multipolygon,
    Multisegment,
    Point,
    Polygon,
    Segment,
)
from hypothesis import strategies as st

from hypothesis_geometry._core.contracts import (
    are_vertices_non_convex as _are_vertices_non_convex,
    are_vertices_strict as _are_vertices_strict,
    has_valid_size as _has_valid_size,
    multicontour_has_valid_sizes as _multicontour_has_valid_sizes,
)
from hypothesis_geometry._core.utils import (
    contours_do_not_cross_or_overlap as _contours_do_not_cross_or_overlap,
    flatten,
    segments_do_not_cross_or_overlap as _segments_do_not_cross_or_overlap,
)
from hypothesis_geometry.hints import Multicontour
from tests.hints import ScalarT

multicontour_has_valid_sizes = _multicontour_has_valid_sizes
has_valid_size = _has_valid_size
Domain = TypeVar('Domain')
Range = TypeVar('Range')
Key: TypeAlias = Callable[[Domain], Any]
Limits: TypeAlias = tuple[ScalarT, ScalarT | None]
ScalarStrategyLimitsWithType: TypeAlias = tuple[
    tuple[st.SearchStrategy[ScalarT], Limits[ScalarT]], type[ScalarT]
]
SizePair: TypeAlias = tuple[int, int | None]
context = get_context()


def identity(argument: Domain) -> Domain:
    return argument


def to_pairs(
    strategy: st.SearchStrategy[Domain],
) -> st.SearchStrategy[tuple[Domain, Domain]]:
    return st.tuples(strategy, strategy)


def pack(
    function: Callable[..., Range], /
) -> Callable[[Iterable[Any]], Range]:
    return partial(apply, function)


def apply(function: Callable[..., Range], args: Iterable[Domain]) -> Range:
    return function(*args)


def box_has_coordinates_in_range(
    box: Box[ScalarT],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return (
        is_coordinate_in_range(
            box.min_x, min_value=min_x_value, max_value=max_x_value
        )
        and is_coordinate_in_range(
            box.max_x, min_value=min_x_value, max_value=max_x_value
        )
        and is_coordinate_in_range(
            box.min_y, min_value=min_y_value, max_value=max_y_value
        )
        and is_coordinate_in_range(
            box.max_y, min_value=min_y_value, max_value=max_y_value
        )
    )


def contour_has_valid_sizes(
    contour: Contour[ScalarT], *, min_size: int, max_size: int | None
) -> bool:
    return has_valid_size(
        contour.vertices, min_size=min_size, max_size=max_size
    )


def mix_has_valid_sizes(
    mix: Mix[ScalarT],
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
) -> bool:
    return (
        has_valid_size(
            mix_to_points(mix),
            min_size=min_points_size,
            max_size=max_points_size,
        )
        and has_valid_size(
            mix_to_segments(mix),
            min_size=min_segments_size,
            max_size=max_segments_size,
        )
        and polygons_have_valid_sizes(
            mix_to_polygons(mix),
            min_size=min_polygons_size,
            max_size=max_polygons_size,
            min_border_size=min_polygon_border_size,
            max_border_size=max_polygon_border_size,
            min_holes_size=min_polygon_holes_size,
            max_holes_size=max_polygon_holes_size,
            min_hole_size=min_polygon_hole_size,
            max_hole_size=max_polygon_hole_size,
        )
    )


def multipolygon_has_valid_sizes(
    multipolygon: Multipolygon[ScalarT],
    *,
    min_size: int,
    max_size: int | None,
    min_border_size: int,
    max_border_size: int | None,
    min_holes_size: int,
    max_holes_size: int | None,
    min_hole_size: int,
    max_hole_size: int | None,
) -> bool:
    return polygons_have_valid_sizes(
        multipolygon.polygons,
        min_size=min_size,
        max_size=max_size,
        min_border_size=min_border_size,
        max_border_size=max_border_size,
        min_holes_size=min_holes_size,
        max_holes_size=max_holes_size,
        min_hole_size=min_hole_size,
        max_hole_size=max_hole_size,
    )


def polygon_has_valid_sizes(
    polygon: Polygon[ScalarT],
    *,
    min_size: int,
    max_size: int | None,
    min_holes_size: int,
    max_holes_size: int | None,
    min_hole_size: int,
    max_hole_size: int | None,
) -> bool:
    return contour_has_valid_sizes(
        polygon.border, min_size=min_size, max_size=max_size
    ) and multicontour_has_valid_sizes(
        polygon.holes,
        min_size=min_holes_size,
        max_size=max_holes_size,
        min_contour_size=min_hole_size,
        max_contour_size=max_hole_size,
    )


def polygons_have_valid_sizes(
    polygons: Sequence[Polygon[ScalarT]],
    *,
    min_size: int,
    max_size: int | None,
    min_border_size: int,
    max_border_size: int | None,
    min_holes_size: int,
    max_holes_size: int | None,
    min_hole_size: int,
    max_hole_size: int | None,
) -> bool:
    return has_valid_size(
        polygons, min_size=min_size, max_size=max_size
    ) and all(
        polygon_has_valid_sizes(
            polygon,
            min_size=min_border_size,
            max_size=max_border_size,
            min_holes_size=min_holes_size,
            max_holes_size=max_holes_size,
            min_hole_size=min_hole_size,
            max_hole_size=max_hole_size,
        )
        for polygon in polygons
    )


def contour_has_coordinates_in_range(
    contour: Contour[ScalarT],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return all(
        point_has_coordinates_in_range(
            vertex,
            min_x_value=min_x_value,
            max_x_value=max_x_value,
            min_y_value=min_y_value,
            max_y_value=max_y_value,
        )
        for vertex in contour.vertices
    )


def mix_has_coordinates_in_range(
    mix: Mix[ScalarT],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return (
        points_have_coordinates_in_range(
            mix_to_points(mix),
            min_x_value=min_x_value,
            max_x_value=max_x_value,
            min_y_value=min_y_value,
            max_y_value=max_y_value,
        )
        and segments_have_coordinates_in_range(
            mix_to_segments(mix),
            min_x_value=min_x_value,
            max_x_value=max_x_value,
            min_y_value=min_y_value,
            max_y_value=max_y_value,
        )
        and polygons_have_coordinates_in_range(
            mix_to_polygons(mix),
            min_x_value=min_x_value,
            max_x_value=max_x_value,
            min_y_value=min_y_value,
            max_y_value=max_y_value,
        )
    )


def mix_to_segments(mix: Mix[ScalarT]) -> Sequence[Segment[ScalarT]]:
    linear = mix.linear
    if isinstance(linear, context.segment_cls):
        return [linear]
    if isinstance(linear, context.multisegment_cls):
        return linear.segments
    if isinstance(linear, context.contour_cls):
        return context.contour_segments(linear)
    assert isinstance(linear, context.empty_cls), linear
    return []


def mix_to_polygons(mix: Mix[ScalarT], /) -> Sequence[Polygon[ScalarT]]:
    shaped = mix.shaped
    if isinstance(shaped, context.polygon_cls):
        return [shaped]
    if isinstance(shaped, context.multipolygon_cls):
        return shaped.polygons
    assert isinstance(shaped, context.empty_cls), shaped
    return []


def mix_to_points(mix: Mix[ScalarT]) -> Sequence[Point[ScalarT]]:
    discrete = mix.discrete
    if isinstance(discrete, context.multipoint_cls):
        return discrete.points
    assert isinstance(discrete, context.empty_cls), discrete
    return []


def multicontour_has_coordinates_in_range(
    multicontour: Multicontour[ScalarT],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return all(
        contour_has_coordinates_in_range(
            contour,
            min_x_value=min_x_value,
            max_x_value=max_x_value,
            min_y_value=min_y_value,
            max_y_value=max_y_value,
        )
        for contour in multicontour
    )


def multipoint_has_coordinates_in_range(
    multipoint: Multipoint[ScalarT],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return points_have_coordinates_in_range(
        multipoint.points,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )


def multipolygon_has_coordinates_in_range(
    multipolygon: Multipolygon[ScalarT],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return polygons_have_coordinates_in_range(
        multipolygon.polygons,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )


def multisegment_has_coordinates_in_range(
    multisegment: Multisegment[ScalarT],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return segments_have_coordinates_in_range(
        multisegment.segments,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )


def points_have_coordinates_in_range(
    points: Iterable[Point[ScalarT]],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return all(
        point_has_coordinates_in_range(
            point,
            min_x_value=min_x_value,
            max_x_value=max_x_value,
            min_y_value=min_y_value,
            max_y_value=max_y_value,
        )
        for point in points
    )


def polygons_have_coordinates_in_range(
    polygons: Iterable[Polygon[ScalarT]],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return all(
        polygon_has_coordinates_in_range(
            polygon,
            min_x_value=min_x_value,
            max_x_value=max_x_value,
            min_y_value=min_y_value,
            max_y_value=max_y_value,
        )
        for polygon in polygons
    )


def segments_have_coordinates_in_range(
    segments: Iterable[Segment[ScalarT]],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return all(
        segment_has_coordinates_in_range(
            segment,
            min_x_value=min_x_value,
            max_x_value=max_x_value,
            min_y_value=min_y_value,
            max_y_value=max_y_value,
        )
        for segment in segments
    )


def point_has_coordinates_in_range(
    point: Point[ScalarT],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return is_coordinate_in_range(
        point.x, min_value=min_x_value, max_value=max_x_value
    ) and is_coordinate_in_range(
        point.y, min_value=min_y_value, max_value=max_y_value
    )


def polygon_has_coordinates_in_range(
    polygon: Polygon[ScalarT],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return contour_has_coordinates_in_range(
        polygon.border,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    ) and multicontour_has_coordinates_in_range(
        polygon.holes,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )


def segment_has_coordinates_in_range(
    segment: Segment[ScalarT],
    *,
    min_x_value: ScalarT,
    max_x_value: ScalarT | None,
    min_y_value: ScalarT,
    max_y_value: ScalarT | None,
) -> bool:
    return point_has_coordinates_in_range(
        segment.start,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    ) and point_has_coordinates_in_range(
        segment.end,
        min_x_value=min_x_value,
        max_x_value=max_x_value,
        min_y_value=min_y_value,
        max_y_value=max_y_value,
    )


def is_coordinate_in_range(
    coordinate: ScalarT, *, min_value: ScalarT, max_value: ScalarT | None
) -> bool:
    return min_value <= coordinate and (
        max_value is None or coordinate <= max_value
    )


def box_has_coordinate_types(
    box: Box[ScalarT], *, x_type: type[ScalarT], y_type: type[ScalarT]
) -> bool:
    return (
        isinstance(box.min_x, x_type)
        and isinstance(box.max_x, x_type)
        and isinstance(box.min_y, y_type)
        and isinstance(box.max_y, y_type)
    )


def contour_has_coordinate_types(
    contour: Contour[ScalarT], *, x_type: type[ScalarT], y_type: type[ScalarT]
) -> bool:
    return all(
        point_has_coordinate_types(vertex, x_type=x_type, y_type=y_type)
        for vertex in contour.vertices
    )


def mix_has_coordinate_types(
    mix: Mix[ScalarT], *, x_type: type[ScalarT], y_type: type[ScalarT]
) -> bool:
    return (
        points_have_coordinate_types(
            mix_to_points(mix), x_type=x_type, y_type=y_type
        )
        and segments_have_coordinate_types(
            mix_to_segments(mix), x_type=x_type, y_type=y_type
        )
        and polygons_have_coordinate_types(
            mix_to_polygons(mix), x_type=x_type, y_type=y_type
        )
    )


def multicontour_has_coordinate_types(
    multicontour: Multicontour[ScalarT],
    *,
    x_type: type[ScalarT],
    y_type: type[ScalarT],
) -> bool:
    return all(
        contour_has_coordinate_types(contour, x_type=x_type, y_type=y_type)
        for contour in multicontour
    )


def multipoint_has_coordinate_types(
    multipoint: Multipoint[ScalarT],
    *,
    x_type: type[ScalarT],
    y_type: type[ScalarT],
) -> bool:
    return points_have_coordinate_types(
        multipoint.points, x_type=x_type, y_type=y_type
    )


def multipolygon_has_coordinate_types(
    multipolygon: Multipolygon[ScalarT],
    *,
    x_type: type[ScalarT],
    y_type: type[ScalarT],
) -> bool:
    return polygons_have_coordinate_types(
        multipolygon.polygons, x_type=x_type, y_type=y_type
    )


def multisegment_has_coordinate_types(
    multisegment: Multisegment[ScalarT],
    *,
    x_type: type[ScalarT],
    y_type: type[ScalarT],
) -> bool:
    return segments_have_coordinate_types(
        multisegment.segments, x_type=x_type, y_type=y_type
    )


def point_has_coordinate_types(
    point: Point[ScalarT], *, x_type: type[ScalarT], y_type: type[ScalarT]
) -> bool:
    return isinstance(point.x, x_type) and isinstance(point.y, y_type)


def points_have_coordinate_types(
    points: Iterable[Point[ScalarT]],
    *,
    x_type: type[ScalarT],
    y_type: type[ScalarT],
) -> bool:
    return all(
        point_has_coordinate_types(point, x_type=x_type, y_type=y_type)
        for point in points
    )


def polygon_has_coordinate_types(
    polygon: Polygon[ScalarT], *, x_type: type[ScalarT], y_type: type[ScalarT]
) -> bool:
    return contour_has_coordinate_types(
        polygon.border, x_type=x_type, y_type=y_type
    ) and multicontour_has_coordinate_types(
        polygon.holes, x_type=x_type, y_type=y_type
    )


def polygons_have_coordinate_types(
    polygons: Iterable[Polygon[ScalarT]],
    *,
    x_type: type[ScalarT],
    y_type: type[ScalarT],
) -> bool:
    return all(
        polygon_has_coordinate_types(polygon, x_type=x_type, y_type=y_type)
        for polygon in polygons
    )


def segment_has_coordinate_types(
    segment: Segment[ScalarT], *, x_type: type[ScalarT], y_type: type[ScalarT]
) -> bool:
    return point_has_coordinate_types(
        segment.start, x_type=x_type, y_type=y_type
    ) and point_has_coordinate_types(segment.end, x_type=x_type, y_type=y_type)


def segments_have_coordinate_types(
    segments: Iterable[Segment[ScalarT]],
    *,
    x_type: type[ScalarT],
    y_type: type[ScalarT],
) -> bool:
    return all(
        segment_has_coordinate_types(segment, x_type=x_type, y_type=y_type)
        for segment in segments
    )


def is_contour_counterclockwise(contour: Contour[ScalarT]) -> bool:
    vertices = contour.vertices
    index_min = min(range(len(vertices)), key=vertices.__getitem__)
    return (
        context.angle_orientation(
            vertices[index_min - 1],
            vertices[index_min],
            vertices[(index_min + 1) % len(vertices)],
        )
        is Orientation.COUNTERCLOCKWISE
    )


_HashableT = TypeVar('_HashableT', bound=Hashable)


def all_unique(iterable: Iterable[_HashableT]) -> bool:
    seen: set[_HashableT] = set()
    seen_add = seen.add
    for element in iterable:
        if element in seen:
            return False
        seen_add(element)
    return True


def is_multicontour(object_: Any) -> bool:
    return isinstance(object_, list) and all(
        isinstance(element, context.contour_cls) for element in object_
    )


def is_contour_non_self_intersecting(contour: Contour[ScalarT]) -> bool:
    return not contour_self_intersects(contour)


to_contour_segments = context.contour_segments


def is_star_contour(contour: Contour[ScalarT]) -> bool:
    return _segments_do_not_cross_or_overlap(
        list(
            chain(
                contour_to_star_segments(contour), to_contour_segments(contour)
            )
        )
    )


def contour_to_star_segments(
    contour: Contour[ScalarT],
) -> Sequence[Segment[ScalarT]]:
    centroid = context.region_centroid(contour)
    return [
        context.segment_cls(centroid, vertex)
        for vertex in contour.vertices
        if vertex != centroid
    ]


def mix_discrete_component_is_disjoint_with_others(mix: Mix[ScalarT]) -> bool:
    points = mix_to_points(mix)
    return not (
        any(
            context.segment_contains_point(segment, point)
            for point in points
            for segment in mix_to_segments(mix)
        )
        or any(
            polygon_contains_point(polygon, point)
            for point in points
            for polygon in mix_to_polygons(mix)
        )
    )


def polygon_contains_point(
    polygon: Polygon[ScalarT], point: Point[ScalarT]
) -> bool:
    location_without_holes = point_in_region(point, polygon.border)
    if location_without_holes is Location.INTERIOR:
        for hole in polygon.holes:
            relation_with_hole = point_in_region(point, hole)
            if relation_with_hole is Location.INTERIOR:
                return False
            if relation_with_hole is Location.BOUNDARY:
                return True
    return location_without_holes is not Location.EXTERIOR


def point_in_region(
    point: Point[ScalarT], region: Contour[ScalarT]
) -> Location:
    result = False
    point_y = point.y
    for edge in context.contour_segments(region):
        if context.segment_contains_point(edge, point):
            return Location.BOUNDARY
        start, end = edge.start, edge.end
        if (start.y > point_y) is not (end.y > point_y) and (
            (end.y > start.y)
            is (
                context.angle_orientation(start, end, point)
                is Orientation.COUNTERCLOCKWISE
            )
        ):
            result = not result
    return Location.INTERIOR if result else Location.EXTERIOR


def mix_segments_do_not_cross_or_overlap(mix: Mix[ScalarT]) -> bool:
    return _segments_do_not_cross_or_overlap(
        list(
            chain(
                mix_to_segments(mix),
                flatten(
                    chain(
                        to_contour_segments(polygon.border),
                        flatten(
                            to_contour_segments(hole) for hole in polygon.holes
                        ),
                    )
                    for polygon in mix_to_polygons(mix)
                ),
            )
        )
    )


def contours_do_not_cross_or_overlap(
    contours: Sequence[Contour[ScalarT]], /
) -> bool:
    return _contours_do_not_cross_or_overlap(
        contours, context.contour_segments
    )


are_vertices_non_convex = partial(
    _are_vertices_non_convex, orienteer=context.angle_orientation
)
are_vertices_strict = partial(
    _are_vertices_strict, orienteer=context.angle_orientation
)


def is_contour_strict(contour: Contour[ScalarT]) -> bool:
    return are_vertices_strict(contour.vertices)


def is_multicontour_strict(multicontour: Multicontour[ScalarT]) -> bool:
    return all(is_contour_strict(contour) for contour in multicontour)


def is_multipolygon_strict(multipolygon: Multipolygon[ScalarT]) -> bool:
    return all(is_polygon_strict(polygon) for polygon in multipolygon.polygons)


def is_polygon_strict(polygon: Polygon[ScalarT]) -> bool:
    return is_contour_strict(polygon.border) and is_multicontour_strict(
        polygon.holes
    )
