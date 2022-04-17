from functools import partial
from itertools import chain
from typing import (Any,
                    Callable,
                    Hashable,
                    Iterable,
                    Optional,
                    Sequence,
                    Tuple,
                    Type,
                    TypeVar)

from bentley_ottmann.planar import (contour_self_intersects,
                                    segments_cross_or_overlap)
from ground.base import (Location,
                         Orientation,
                         get_context)
from ground.hints import Scalar
from hypothesis import strategies

from hypothesis_geometry.core.contracts import (
    are_vertices_non_convex as _are_vertices_non_convex,
    are_vertices_strict as _are_vertices_strict,
    has_valid_size,
    multicontour_has_valid_sizes)
from hypothesis_geometry.core.utils import flatten
from hypothesis_geometry.hints import (Multicontour,
                                       Strategy)

has_valid_size = has_valid_size
Domain = TypeVar('Domain')
Range = TypeVar('Range')
Key = Callable[[Domain], Any]
Limits = Tuple[Scalar, Optional[Scalar]]
ScalarsLimitsType = Tuple[Tuple[Strategy[Scalar], Limits],
                          Type[Scalar]]
SizesPair = Tuple[int, Optional[int]]
context = get_context()
Box = context.box_cls
Contour = context.contour_cls
Empty = context.empty_cls
Mix = context.mix_cls
Multipoint = context.multipoint_cls
Multipolygon = context.multipolygon_cls
Multisegment = context.multisegment_cls
Point = context.point_cls
Polygon = context.polygon_cls
Segment = context.segment_cls


def identity(argument: Domain) -> Domain:
    return argument


def to_pairs(strategy: Strategy[Domain]) -> Strategy[Tuple[Domain, Domain]]:
    return strategies.tuples(strategy, strategy)


def pack(function: Callable[..., Range]
         ) -> Callable[[Iterable[Domain]], Range]:
    return partial(apply, function)


def apply(function: Callable[..., Range], args: Iterable[Domain]) -> Range:
    return function(*args)


def box_has_coordinates_in_range(box: Box,
                                 *,
                                 min_x_value: Scalar,
                                 max_x_value: Optional[Scalar],
                                 min_y_value: Scalar,
                                 max_y_value: Optional[Scalar]) -> bool:
    return (is_coordinate_in_range(box.min_x,
                                   min_value=min_x_value,
                                   max_value=max_x_value)
            and is_coordinate_in_range(box.max_x,
                                       min_value=min_x_value,
                                       max_value=max_x_value)
            and is_coordinate_in_range(box.min_y,
                                       min_value=min_y_value,
                                       max_value=max_y_value)
            and is_coordinate_in_range(box.max_y,
                                       min_value=min_y_value,
                                       max_value=max_y_value))


def contour_has_valid_sizes(contour: Contour,
                            *,
                            min_size: int,
                            max_size: Optional[int]) -> bool:
    return has_valid_size(contour.vertices,
                          min_size=min_size,
                          max_size=max_size)


def mix_has_valid_sizes(mix: Mix,
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
                        max_polygon_hole_size: Optional[int]) -> bool:
    return (has_valid_size(mix_to_points(mix),
                           min_size=min_points_size,
                           max_size=max_points_size)
            and has_valid_size(mix_to_segments(mix),
                               min_size=min_segments_size,
                               max_size=max_segments_size)
            and polygons_have_valid_sizes(
                    mix_to_polygons(mix),
                    min_size=min_polygons_size,
                    max_size=max_polygons_size,
                    min_border_size=min_polygon_border_size,
                    max_border_size=max_polygon_border_size,
                    min_holes_size=min_polygon_holes_size,
                    max_holes_size=max_polygon_holes_size,
                    min_hole_size=min_polygon_hole_size,
                    max_hole_size=max_polygon_hole_size))


def multipolygon_has_valid_sizes(multipolygon: Multipolygon,
                                 *,
                                 min_size: int,
                                 max_size: Optional[int],
                                 min_border_size: int,
                                 max_border_size: Optional[int],
                                 min_holes_size: int,
                                 max_holes_size: Optional[int],
                                 min_hole_size: int,
                                 max_hole_size: Optional[int]) -> bool:
    return polygons_have_valid_sizes(multipolygon.polygons,
                                     min_size=min_size,
                                     max_size=max_size,
                                     min_border_size=min_border_size,
                                     max_border_size=max_border_size,
                                     min_holes_size=min_holes_size,
                                     max_holes_size=max_holes_size,
                                     min_hole_size=min_hole_size,
                                     max_hole_size=max_hole_size)


def polygon_has_valid_sizes(polygon: Polygon,
                            *,
                            min_size: int,
                            max_size: Optional[int],
                            min_holes_size: int,
                            max_holes_size: Optional[int],
                            min_hole_size: int,
                            max_hole_size: Optional[int]) -> bool:
    return (contour_has_valid_sizes(polygon.border,
                                    min_size=min_size,
                                    max_size=max_size)
            and multicontour_has_valid_sizes(polygon.holes,
                                             min_size=min_holes_size,
                                             max_size=max_holes_size,
                                             min_contour_size=min_hole_size,
                                             max_contour_size=max_hole_size))


def polygons_have_valid_sizes(polygons: Sequence[Polygon],
                              *,
                              min_size: int,
                              max_size: Optional[int],
                              min_border_size: int,
                              max_border_size: Optional[int],
                              min_holes_size: int,
                              max_holes_size: Optional[int],
                              min_hole_size: int,
                              max_hole_size: Optional[int]) -> bool:
    return (has_valid_size(polygons,
                           min_size=min_size,
                           max_size=max_size)
            and all(polygon_has_valid_sizes(polygon,
                                            min_size=min_border_size,
                                            max_size=max_border_size,
                                            min_holes_size=min_holes_size,
                                            max_holes_size=max_holes_size,
                                            min_hole_size=min_hole_size,
                                            max_hole_size=max_hole_size)
                    for polygon in polygons))


def contour_has_coordinates_in_range(contour: Contour,
                                     *,
                                     min_x_value: Scalar,
                                     max_x_value: Optional[Scalar],
                                     min_y_value: Scalar,
                                     max_y_value: Optional[Scalar]
                                     ) -> bool:
    return all(point_has_coordinates_in_range(vertex,
                                              min_x_value=min_x_value,
                                              max_x_value=max_x_value,
                                              min_y_value=min_y_value,
                                              max_y_value=max_y_value)
               for vertex in contour.vertices)


def mix_has_coordinates_in_range(mix: Mix,
                                 *,
                                 min_x_value: Scalar,
                                 max_x_value: Optional[Scalar],
                                 min_y_value: Scalar,
                                 max_y_value: Optional[Scalar]) -> bool:
    return (points_have_coordinates_in_range(mix_to_points(mix),
                                             min_x_value=min_x_value,
                                             max_x_value=max_x_value,
                                             min_y_value=min_y_value,
                                             max_y_value=max_y_value)
            and segments_have_coordinates_in_range(mix_to_segments(mix),
                                                   min_x_value=min_x_value,
                                                   max_x_value=max_x_value,
                                                   min_y_value=min_y_value,
                                                   max_y_value=max_y_value)
            and polygons_have_coordinates_in_range(mix_to_polygons(mix),
                                                   min_x_value=min_x_value,
                                                   max_x_value=max_x_value,
                                                   min_y_value=min_y_value,
                                                   max_y_value=max_y_value))


def mix_to_segments(mix: Mix) -> Sequence[Segment]:
    linear = mix.linear
    return ([]
            if is_empty(linear)
            else ([linear]
                  if isinstance(linear, Segment)
                  else (linear.segments
                        if isinstance(linear, Multisegment)
                        else context.contour_segments(linear))))


def mix_to_polygons(mix: Mix) -> Sequence[Polygon]:
    shaped = mix.shaped
    return ([]
            if is_empty(shaped)
            else ([shaped]
                  if isinstance(shaped, Polygon)
                  else shaped.polygons))


def mix_to_points(mix: Mix) -> Sequence[Point]:
    discrete = mix.discrete
    return [] if is_empty(discrete) else discrete.points


def multicontour_has_coordinates_in_range(multicontour: Multicontour,
                                          *,
                                          min_x_value: Scalar,
                                          max_x_value: Optional[Scalar],
                                          min_y_value: Scalar,
                                          max_y_value: Optional[Scalar]
                                          ) -> bool:
    return all(contour_has_coordinates_in_range(contour,
                                                min_x_value=min_x_value,
                                                max_x_value=max_x_value,
                                                min_y_value=min_y_value,
                                                max_y_value=max_y_value)
               for contour in multicontour)


def multipoint_has_coordinates_in_range(multipoint: Multipoint,
                                        *,
                                        min_x_value: Scalar,
                                        max_x_value: Optional[Scalar],
                                        min_y_value: Scalar,
                                        max_y_value: Optional[Scalar]
                                        ) -> bool:
    return points_have_coordinates_in_range(multipoint.points,
                                            min_x_value=min_x_value,
                                            max_x_value=max_x_value,
                                            min_y_value=min_y_value,
                                            max_y_value=max_y_value)


def multipolygon_has_coordinates_in_range(multipolygon: Multipolygon,
                                          *,
                                          min_x_value: Scalar,
                                          max_x_value: Optional[Scalar],
                                          min_y_value: Scalar,
                                          max_y_value: Optional[Scalar]
                                          ) -> bool:
    return polygons_have_coordinates_in_range(multipolygon.polygons,
                                              min_x_value=min_x_value,
                                              max_x_value=max_x_value,
                                              min_y_value=min_y_value,
                                              max_y_value=max_y_value)


def multisegment_has_coordinates_in_range(multisegment: Multisegment,
                                          *,
                                          min_x_value: Scalar,
                                          max_x_value: Optional[Scalar],
                                          min_y_value: Scalar,
                                          max_y_value: Optional[Scalar]
                                          ) -> bool:
    return segments_have_coordinates_in_range(multisegment.segments,
                                              min_x_value=min_x_value,
                                              max_x_value=max_x_value,
                                              min_y_value=min_y_value,
                                              max_y_value=max_y_value)


def points_have_coordinates_in_range(points: Iterable[Point],
                                     *,
                                     min_x_value: Scalar,
                                     max_x_value: Optional[Scalar],
                                     min_y_value: Scalar,
                                     max_y_value: Optional[Scalar]) -> bool:
    return all(point_has_coordinates_in_range(point,
                                              min_x_value=min_x_value,
                                              max_x_value=max_x_value,
                                              min_y_value=min_y_value,
                                              max_y_value=max_y_value)
               for point in points)


def polygons_have_coordinates_in_range(polygons: Iterable[Polygon],
                                       *,
                                       min_x_value: Scalar,
                                       max_x_value: Optional[Scalar],
                                       min_y_value: Scalar,
                                       max_y_value: Optional[Scalar]) -> bool:
    return all(polygon_has_coordinates_in_range(polygon,
                                                min_x_value=min_x_value,
                                                max_x_value=max_x_value,
                                                min_y_value=min_y_value,
                                                max_y_value=max_y_value)
               for polygon in polygons)


def segments_have_coordinates_in_range(segments: Iterable[Segment],
                                       *,
                                       min_x_value: Scalar,
                                       max_x_value: Optional[Scalar],
                                       min_y_value: Scalar,
                                       max_y_value: Optional[Scalar]) -> bool:
    return all(segment_has_coordinates_in_range(segment,
                                                min_x_value=min_x_value,
                                                max_x_value=max_x_value,
                                                min_y_value=min_y_value,
                                                max_y_value=max_y_value)
               for segment in segments)


def point_has_coordinates_in_range(point: Point,
                                   *,
                                   min_x_value: Scalar,
                                   max_x_value: Optional[Scalar],
                                   min_y_value: Scalar,
                                   max_y_value: Optional[Scalar]) -> bool:
    return (is_coordinate_in_range(point.x,
                                   min_value=min_x_value,
                                   max_value=max_x_value)
            and is_coordinate_in_range(point.y,
                                       min_value=min_y_value,
                                       max_value=max_y_value))


def polygon_has_coordinates_in_range(polygon: Polygon,
                                     *,
                                     min_x_value: Scalar,
                                     max_x_value: Optional[Scalar],
                                     min_y_value: Scalar,
                                     max_y_value: Optional[Scalar]
                                     ) -> bool:
    return (contour_has_coordinates_in_range(polygon.border,
                                             min_x_value=min_x_value,
                                             max_x_value=max_x_value,
                                             min_y_value=min_y_value,
                                             max_y_value=max_y_value)
            and multicontour_has_coordinates_in_range(polygon.holes,
                                                      min_x_value=min_x_value,
                                                      max_x_value=max_x_value,
                                                      min_y_value=min_y_value,
                                                      max_y_value=max_y_value))


def segment_has_coordinates_in_range(segment: Segment,
                                     *,
                                     min_x_value: Scalar,
                                     max_x_value: Optional[Scalar],
                                     min_y_value: Scalar,
                                     max_y_value: Optional[Scalar]
                                     ) -> bool:
    return (point_has_coordinates_in_range(segment.start,
                                           min_x_value=min_x_value,
                                           max_x_value=max_x_value,
                                           min_y_value=min_y_value,
                                           max_y_value=max_y_value)
            and point_has_coordinates_in_range(segment.end,
                                               min_x_value=min_x_value,
                                               max_x_value=max_x_value,
                                               min_y_value=min_y_value,
                                               max_y_value=max_y_value))


def is_coordinate_in_range(coordinate: Scalar,
                           *,
                           min_value: Scalar,
                           max_value: Optional[Scalar]) -> bool:
    return (min_value <= coordinate
            and (max_value is None or coordinate <= max_value))


def box_has_coordinates_types(box: Box,
                              *,
                              x_type: Type[Scalar],
                              y_type: Type[Scalar]) -> bool:
    return (isinstance(box.min_x, x_type) and isinstance(box.max_x, x_type)
            and isinstance(box.min_y, y_type)
            and isinstance(box.max_y, y_type))


def contour_has_coordinates_types(contour: Contour,
                                  *,
                                  x_type: Type[Scalar],
                                  y_type: Type[Scalar]) -> bool:
    return all(point_has_coordinates_types(vertex,
                                           x_type=x_type,
                                           y_type=y_type)
               for vertex in contour.vertices)


def mix_has_coordinates_types(mix: Mix,
                              *,
                              x_type: Type[Scalar],
                              y_type: Type[Scalar]) -> bool:
    return (points_have_coordinates_types(mix_to_points(mix),
                                          x_type=x_type,
                                          y_type=y_type)
            and segments_have_coordinates_types(mix_to_segments(mix),
                                                x_type=x_type,
                                                y_type=y_type)
            and polygons_have_coordinates_types(mix_to_polygons(mix),
                                                x_type=x_type,
                                                y_type=y_type))


def multicontour_has_coordinates_types(multicontour: Multicontour,
                                       *,
                                       x_type: Type[Scalar],
                                       y_type: Type[Scalar]) -> bool:
    return all(contour_has_coordinates_types(contour,
                                             x_type=x_type,
                                             y_type=y_type)
               for contour in multicontour)


def multipoint_has_coordinates_types(multipoint: Multipoint,
                                     *,
                                     x_type: Type[Scalar],
                                     y_type: Type[Scalar]) -> bool:
    return points_have_coordinates_types(multipoint.points,
                                         x_type=x_type,
                                         y_type=y_type)


def multipolygon_has_coordinates_types(multipolygon: Multipolygon,
                                       *,
                                       x_type: Type[Scalar],
                                       y_type: Type[Scalar]) -> bool:
    return polygons_have_coordinates_types(multipolygon.polygons,
                                           x_type=x_type,
                                           y_type=y_type)


def multisegment_has_coordinates_types(multisegment: Multisegment,
                                       *,
                                       x_type: Type[Scalar],
                                       y_type: Type[Scalar]) -> bool:
    return segments_have_coordinates_types(multisegment.segments,
                                           x_type=x_type,
                                           y_type=y_type)


def point_has_coordinates_types(point: Point,
                                *,
                                x_type: Type[Scalar],
                                y_type: Type[Scalar]) -> bool:
    return isinstance(point.x, x_type) and isinstance(point.y, y_type)


def points_have_coordinates_types(points: Iterable[Point],
                                  *,
                                  x_type: Type[Scalar],
                                  y_type: Type[Scalar]) -> bool:
    return all(point_has_coordinates_types(point,
                                           x_type=x_type,
                                           y_type=y_type)
               for point in points)


def polygon_has_coordinates_types(polygon: Polygon,
                                  *,
                                  x_type: Type[Scalar],
                                  y_type: Type[Scalar]) -> bool:
    return (contour_has_coordinates_types(polygon.border,
                                          x_type=x_type,
                                          y_type=y_type)
            and multicontour_has_coordinates_types(polygon.holes,
                                                   x_type=x_type,
                                                   y_type=y_type))


def polygons_have_coordinates_types(polygons: Iterable[Polygon],
                                    *,
                                    x_type: Type[Scalar],
                                    y_type: Type[Scalar]) -> bool:
    return all(polygon_has_coordinates_types(polygon,
                                             x_type=x_type,
                                             y_type=y_type)
               for polygon in polygons)


def segment_has_coordinates_types(segment: Segment,
                                  *,
                                  x_type: Type[Scalar],
                                  y_type: Type[Scalar]) -> bool:
    return (point_has_coordinates_types(segment.start,
                                        x_type=x_type,
                                        y_type=y_type)
            and point_has_coordinates_types(segment.end,
                                            x_type=x_type,
                                            y_type=y_type))


def segments_have_coordinates_types(segments: Iterable[Segment],
                                    *,
                                    x_type: Type[Scalar],
                                    y_type: Type[Scalar]) -> bool:
    return all(segment_has_coordinates_types(segment,
                                             x_type=x_type,
                                             y_type=y_type)
               for segment in segments)


def is_contour_counterclockwise(contour: Contour) -> bool:
    vertices = contour.vertices
    index_min = min(range(len(vertices)),
                    key=vertices.__getitem__)
    return (context.angle_orientation(vertices[index_min - 1],
                                      vertices[index_min],
                                      vertices[(index_min + 1)
                                               % len(vertices)])
            is Orientation.COUNTERCLOCKWISE)


def all_unique(iterable: Iterable[Hashable]) -> bool:
    seen = set()
    seen_add = seen.add
    for element in iterable:
        if element in seen:
            return False
        seen_add(element)
    return True


is_box = Box.__instancecheck__
is_contour = Contour.__instancecheck__
is_empty = Empty.__instancecheck__
is_mix = Mix.__instancecheck__


def is_multicontour(object_: Any) -> bool:
    return isinstance(object_, list) and all(map(is_contour, object_))


is_multipoint = Multipoint.__instancecheck__
is_multipolygon = Multipolygon.__instancecheck__
is_multisegment = Multisegment.__instancecheck__
is_point = Point.__instancecheck__
is_polygon = Polygon.__instancecheck__
is_segment = Segment.__instancecheck__


def is_contour_non_self_intersecting(contour: Contour) -> bool:
    return not contour_self_intersects(contour)


to_contour_segments = context.contour_segments


def is_star_contour(contour: Contour) -> bool:
    return segments_do_not_cross_or_overlap(
            list(chain(contour_to_star_segments(contour),
                       to_contour_segments(contour))))


def contour_to_star_segments(contour: Contour) -> Sequence[Segment]:
    centroid = context.region_centroid(contour)
    return [Segment(centroid, vertex)
            for vertex in contour.vertices
            if vertex != centroid]


def mix_discrete_component_is_disjoint_with_others(mix: Mix) -> bool:
    points = mix_to_points(mix)
    return not (any(context.segment_contains_point(segment, point)
                    for point in points
                    for segment in mix_to_segments(mix))
                or any(polygon_contains_point(polygon, point)
                       for point in points
                       for polygon in mix_to_polygons(mix)))


def polygon_contains_point(polygon: Polygon, point: Point) -> bool:
    location_without_holes = point_in_region(point, polygon.border)
    if location_without_holes is Location.INTERIOR:
        for hole in polygon.holes:
            relation_with_hole = point_in_region(point, hole)
            if relation_with_hole is Location.INTERIOR:
                return False
            elif relation_with_hole is Location.BOUNDARY:
                return True
    return location_without_holes is not Location.EXTERIOR


def point_in_region(point: Point, region: Contour) -> Location:
    result = False
    point_y = point.y
    for index, edge in enumerate(context.contour_segments(region)):
        if context.segment_contains_point(edge, point):
            return Location.BOUNDARY
        start, end = edge.start, edge.end
        if ((start.y > point_y) is not (end.y > point_y)
                and ((end.y > start.y)
                     is (context.angle_orientation(start, end, point)
                         is Orientation.COUNTERCLOCKWISE))):
            result = not result
    return Location.INTERIOR if result else Location.EXTERIOR


def mix_segments_do_not_cross_or_overlap(mix: Mix) -> bool:
    return segments_do_not_cross_or_overlap(
            list(chain(mix_to_segments(mix),
                       flatten(chain(to_contour_segments(polygon.border),
                                     flatten(to_contour_segments(hole)
                                             for hole in polygon.holes))
                               for polygon in mix_to_polygons(mix))))
    )


def contours_do_not_cross_or_overlap(contours: Sequence[Contour]) -> bool:
    return segments_do_not_cross_or_overlap(list(flatten(
            to_contour_segments(contour) for contour in contours)))


def segments_do_not_cross_or_overlap(segments: Sequence[Segment]) -> bool:
    return not segments_cross_or_overlap(segments)


are_vertices_non_convex = partial(_are_vertices_non_convex,
                                  orienteer=context.angle_orientation)
are_vertices_strict = partial(_are_vertices_strict,
                              orienteer=context.angle_orientation)


def is_contour_strict(contour: Contour) -> bool:
    return are_vertices_strict(contour.vertices)


def is_multicontour_strict(multicontour: Multicontour) -> bool:
    return all(is_contour_strict(contour) for contour in multicontour)


def is_multipolygon_strict(multipolygon: Multipolygon) -> bool:
    return all(is_polygon_strict(polygon) for polygon in multipolygon.polygons)


def is_polygon_strict(polygon: Polygon) -> bool:
    return (is_contour_strict(polygon.border)
            and is_multicontour_strict(polygon.holes))
