from itertools import (chain,
                       groupby)
from typing import (Any,
                    Callable,
                    Hashable,
                    Iterable,
                    List,
                    Optional,
                    Tuple,
                    Type,
                    TypeVar)

from bentley_ottmann.planar import (edges_intersect,
                                    segments_cross_or_overlap)
from ground.base import get_context
from hypothesis import strategies

from hypothesis_geometry.core.contracts import is_contour_strict
from hypothesis_geometry.core.utils import (Orientation,
                                            contour_to_centroid,
                                            flatten,
                                            orientation)
from hypothesis_geometry.hints import (Contour,
                                       Coordinate,
                                       Mix,
                                       Multicontour,
                                       Multipoint,
                                       Multipolygon,
                                       Multisegment,
                                       Polygon,
                                       Segment,
                                       Strategy)
from hypothesis_geometry.planar import (MIN_POLYLINE_SIZE,
                                        TRIANGULAR_CONTOUR_SIZE,
                                        _has_valid_size)
from hypothesis_geometry.utils import contour_to_multisegment

has_valid_size = _has_valid_size
Domain = TypeVar('Domain')
Key = Callable[[Domain], Any]
Limits = Tuple[Coordinate, Optional[Coordinate]]
CoordinatesLimitsType = Tuple[Tuple[Strategy[Coordinate], Limits],
                              Type[Coordinate]]
SizesPair = Tuple[int, Optional[int]]
context = get_context()
Box = context.box_cls
Point = context.point_cls


def identity(argument: Domain) -> Domain:
    return argument


def to_pairs(strategy: Strategy[Domain]) -> Strategy[Tuple[Domain, Domain]]:
    return strategies.tuples(strategy, strategy)


def box_has_coordinates_in_range(box: Box,
                                 *,
                                 min_x_value: Coordinate,
                                 max_x_value: Optional[Coordinate],
                                 min_y_value: Coordinate,
                                 max_y_value: Optional[Coordinate]) -> bool:
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


def mix_has_valid_sizes(mix: Mix,
                        *,
                        min_multipoint_size: int,
                        max_multipoint_size: Optional[int],
                        min_multisegment_size: int,
                        max_multisegment_size: Optional[int],
                        min_multipolygon_size: int,
                        max_multipolygon_size: Optional[int],
                        min_multipolygon_border_size: int,
                        max_multipolygon_border_size: Optional[int],
                        min_multipolygon_holes_size: int,
                        max_multipolygon_holes_size: Optional[int],
                        min_multipolygon_hole_size: int,
                        max_multipolygon_hole_size: Optional[int]
                        ) -> bool:
    multipoint, multisegment, multipolygon = mix
    return (has_valid_size(multipoint,
                           min_size=min_multipoint_size,
                           max_size=max_multipoint_size)
            and has_valid_size(multisegment,
                               min_size=min_multisegment_size,
                               max_size=max_multisegment_size)
            and multipolygon_has_valid_sizes(
                    multipolygon,
                    min_size=min_multipolygon_size,
                    max_size=max_multipolygon_size,
                    min_border_size=min_multipolygon_border_size,
                    max_border_size=max_multipolygon_border_size,
                    min_holes_size=min_multipolygon_holes_size,
                    max_holes_size=max_multipolygon_holes_size,
                    min_hole_size=min_multipolygon_hole_size,
                    max_hole_size=max_multipolygon_hole_size))


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
    return (has_valid_size(multipolygon,
                           min_size=min_size,
                           max_size=max_size)
            and all(polygon_has_valid_sizes(polygon,
                                            min_size=min_border_size,
                                            max_size=max_border_size,
                                            min_holes_size=min_holes_size,
                                            max_holes_size=max_holes_size,
                                            min_hole_size=min_hole_size,
                                            max_hole_size=max_hole_size)
                    for polygon in multipolygon))


def polygon_has_valid_sizes(polygon: Polygon,
                            *,
                            min_size: int,
                            max_size: Optional[int],
                            min_holes_size: int,
                            max_holes_size: Optional[int],
                            min_hole_size: int,
                            max_hole_size: Optional[int]) -> bool:
    border, holes = polygon
    return (has_valid_size(border,
                           min_size=min_size,
                           max_size=max_size)
            and has_valid_size(holes,
                               min_size=min_holes_size,
                               max_size=max_holes_size)
            and all(has_valid_size(hole,
                                   min_size=min_hole_size,
                                   max_size=max_hole_size)
                    for hole in holes))


def contour_has_coordinates_in_range(contour: Contour,
                                     *,
                                     min_x_value: Coordinate,
                                     max_x_value: Optional[Coordinate],
                                     min_y_value: Coordinate,
                                     max_y_value: Optional[Coordinate]
                                     ) -> bool:
    return all(point_has_coordinates_in_range(vertex,
                                              min_x_value=min_x_value,
                                              max_x_value=max_x_value,
                                              min_y_value=min_y_value,
                                              max_y_value=max_y_value)
               for vertex in contour)


def mix_has_coordinates_in_range(mix: Mix,
                                 *,
                                 min_x_value: Coordinate,
                                 max_x_value: Optional[Coordinate],
                                 min_y_value: Coordinate,
                                 max_y_value: Optional[Coordinate]
                                 ) -> bool:
    multipoint, multisegment, multipolygon = mix
    return (multipoint_has_coordinates_in_range(multipoint,
                                                min_x_value=min_x_value,
                                                max_x_value=max_x_value,
                                                min_y_value=min_y_value,
                                                max_y_value=max_y_value)
            and multisegment_has_coordinates_in_range(multisegment,
                                                      min_x_value=min_x_value,
                                                      max_x_value=max_x_value,
                                                      min_y_value=min_y_value,
                                                      max_y_value=max_y_value)
            and multipolygon_has_coordinates_in_range(multipolygon,
                                                      min_x_value=min_x_value,
                                                      max_x_value=max_x_value,
                                                      min_y_value=min_y_value,
                                                      max_y_value=max_y_value))


def multicontour_has_coordinates_in_range(multicontour: Multicontour,
                                          *,
                                          min_x_value: Coordinate,
                                          max_x_value: Optional[Coordinate],
                                          min_y_value: Coordinate,
                                          max_y_value: Optional[Coordinate]
                                          ) -> bool:
    return all(contour_has_coordinates_in_range(contour,
                                                min_x_value=min_x_value,
                                                max_x_value=max_x_value,
                                                min_y_value=min_y_value,
                                                max_y_value=max_y_value)
               for contour in multicontour)


def multipoint_has_coordinates_in_range(multipoint: Multipoint,
                                        *,
                                        min_x_value: Coordinate,
                                        max_x_value: Optional[Coordinate],
                                        min_y_value: Coordinate,
                                        max_y_value: Optional[Coordinate]
                                        ) -> bool:
    return all(point_has_coordinates_in_range(point,
                                              min_x_value=min_x_value,
                                              max_x_value=max_x_value,
                                              min_y_value=min_y_value,
                                              max_y_value=max_y_value)
               for point in multipoint)


def multipolygon_has_coordinates_in_range(multipolygon: Multipolygon,
                                          *,
                                          min_x_value: Coordinate,
                                          max_x_value: Optional[Coordinate],
                                          min_y_value: Coordinate,
                                          max_y_value: Optional[Coordinate]
                                          ) -> bool:
    return all(polygon_has_coordinates_in_range(polygon,
                                                min_x_value=min_x_value,
                                                max_x_value=max_x_value,
                                                min_y_value=min_y_value,
                                                max_y_value=max_y_value)
               for polygon in multipolygon)


def multisegment_has_coordinates_in_range(multisegment: Multisegment,
                                          *,
                                          min_x_value: Coordinate,
                                          max_x_value: Optional[Coordinate],
                                          min_y_value: Coordinate,
                                          max_y_value: Optional[Coordinate]
                                          ) -> bool:
    return all(segment_has_coordinates_in_range(segment,
                                                min_x_value=min_x_value,
                                                max_x_value=max_x_value,
                                                min_y_value=min_y_value,
                                                max_y_value=max_y_value)
               for segment in multisegment)


def point_has_coordinates_in_range(point: Point,
                                   *,
                                   min_x_value: Coordinate,
                                   max_x_value: Optional[Coordinate],
                                   min_y_value: Coordinate,
                                   max_y_value: Optional[Coordinate]) -> bool:
    return (is_coordinate_in_range(point.x,
                                   min_value=min_x_value,
                                   max_value=max_x_value)
            and is_coordinate_in_range(point.y,
                                       min_value=min_y_value,
                                       max_value=max_y_value))


def polygon_has_coordinates_in_range(polygon: Polygon,
                                     *,
                                     min_x_value: Coordinate,
                                     max_x_value: Optional[Coordinate],
                                     min_y_value: Coordinate,
                                     max_y_value: Optional[Coordinate]
                                     ) -> bool:
    border, holes = polygon
    return (contour_has_coordinates_in_range(border,
                                             min_x_value=min_x_value,
                                             max_x_value=max_x_value,
                                             min_y_value=min_y_value,
                                             max_y_value=max_y_value)
            and multicontour_has_coordinates_in_range(holes,
                                                      min_x_value=min_x_value,
                                                      max_x_value=max_x_value,
                                                      min_y_value=min_y_value,
                                                      max_y_value=max_y_value))


def segment_has_coordinates_in_range(segment: Segment,
                                     *,
                                     min_x_value: Coordinate,
                                     max_x_value: Optional[Coordinate],
                                     min_y_value: Coordinate,
                                     max_y_value: Optional[Coordinate]
                                     ) -> bool:
    start, end = segment
    return (point_has_coordinates_in_range(start,
                                           min_x_value=min_x_value,
                                           max_x_value=max_x_value,
                                           min_y_value=min_y_value,
                                           max_y_value=max_y_value)
            and point_has_coordinates_in_range(end,
                                               min_x_value=min_x_value,
                                               max_x_value=max_x_value,
                                               min_y_value=min_y_value,
                                               max_y_value=max_y_value))


def is_coordinate_in_range(coordinate: Coordinate,
                           *,
                           min_value: Coordinate,
                           max_value: Optional[Coordinate]) -> bool:
    return (min_value <= coordinate
            and (max_value is None or coordinate <= max_value))


def box_has_coordinates_types(box: Box,
                              *,
                              x_type: Type[Coordinate],
                              y_type: Type[Coordinate]) -> bool:
    return (isinstance(box.min_x, x_type) and isinstance(box.max_x, x_type)
            and isinstance(box.min_y, y_type)
            and isinstance(box.max_y, y_type))


def contour_has_coordinates_types(contour: Contour,
                                  *,
                                  x_type: Type[Coordinate],
                                  y_type: Type[Coordinate]) -> bool:
    return all(point_has_coordinates_types(vertex,
                                           x_type=x_type,
                                           y_type=y_type)
               for vertex in contour)


def mix_has_coordinates_types(mix: Mix,
                              *,
                              x_type: Type[Coordinate],
                              y_type: Type[Coordinate]) -> bool:
    multipoint, multisegment, multipolygon = mix
    return (multipoint_has_coordinates_types(multipoint,
                                             x_type=x_type,
                                             y_type=y_type)
            and multisegment_has_coordinates_types(multisegment,
                                                   x_type=x_type,
                                                   y_type=y_type)
            and multipolygon_has_coordinates_types(multipolygon,
                                                   x_type=x_type,
                                                   y_type=y_type))


def multicontour_has_coordinates_types(multicontour: Multicontour,
                                       *,
                                       x_type: Type[Coordinate],
                                       y_type: Type[Coordinate]) -> bool:
    return all(contour_has_coordinates_types(contour,
                                             x_type=x_type,
                                             y_type=y_type)
               for contour in multicontour)


def multipoint_has_coordinates_types(multipoint: Multipoint,
                                     *,
                                     x_type: Type[Coordinate],
                                     y_type: Type[Coordinate]) -> bool:
    return all(point_has_coordinates_types(point,
                                           x_type=x_type,
                                           y_type=y_type)
               for point in multipoint)


def multipolygon_has_coordinates_types(multipolygon: Multipolygon,
                                       *,
                                       x_type: Type[Coordinate],
                                       y_type: Type[Coordinate]) -> bool:
    return all(polygon_has_coordinates_types(polygon,
                                             x_type=x_type,
                                             y_type=y_type)
               for polygon in multipolygon)


def multisegment_has_coordinates_types(multisegment: Multisegment,
                                       *,
                                       x_type: Type[Coordinate],
                                       y_type: Type[Coordinate]):
    return all(segment_has_coordinates_types(segment,
                                             x_type=x_type,
                                             y_type=y_type)
               for segment in multisegment)


def point_has_coordinates_types(point: Point,
                                *,
                                x_type: Type[Coordinate],
                                y_type: Type[Coordinate]) -> bool:
    return isinstance(point.x, x_type) and isinstance(point.y, y_type)


def polygon_has_coordinates_types(polygon: Polygon,
                                  *,
                                  x_type: Type[Coordinate],
                                  y_type: Type[Coordinate]) -> bool:
    border, holes = polygon
    return (contour_has_coordinates_types(border,
                                          x_type=x_type,
                                          y_type=y_type)
            and multicontour_has_coordinates_types(holes,
                                                   x_type=x_type,
                                                   y_type=y_type))


def segment_has_coordinates_types(segment: Segment,
                                  *,
                                  x_type: Type[Coordinate],
                                  y_type: Type[Coordinate]) -> bool:
    start, end = segment
    return (point_has_coordinates_types(start,
                                        x_type=x_type,
                                        y_type=y_type)
            and point_has_coordinates_types(end,
                                            x_type=x_type,
                                            y_type=y_type))


def has_no_consecutive_repetitions(iterable: Iterable[Domain]) -> bool:
    return any(capacity(group) == 1
               for _, group in groupby(iterable))


def is_counterclockwise_contour(contour: Contour) -> bool:
    index_min = to_index_min(contour)
    return (orientation(contour[index_min - 1], contour[index_min],
                        contour[(index_min + 1) % len(contour)])
            is Orientation.CLOCKWISE)


_sentinel = object()


def to_index_min(values: Iterable[Domain],
                 *,
                 key: Optional[Key] = None,
                 default: Any = _sentinel) -> int:
    kwargs = {}
    if key is not None:
        kwargs['key'] = lambda value_with_index: key(value_with_index[0])
    if default is not _sentinel:
        kwargs['default'] = default
    return min(((value, index)
                for index, value in enumerate(values)),
               **kwargs)[1]


def capacity(iterable: Iterable[Domain]) -> int:
    return sum(1 for _ in iterable)


def all_unique(iterable: Iterable[Hashable]) -> bool:
    seen = set()
    seen_add = seen.add
    for element in iterable:
        if element in seen:
            return False
        seen_add(element)
    return True


is_box = Box.__instancecheck__


def is_contour(object_: Any) -> bool:
    return (isinstance(object_, list)
            and len(object_) >= TRIANGULAR_CONTOUR_SIZE
            and all(map(is_point, object_)))


def is_multipoint(object_: Any) -> bool:
    return isinstance(object_, list) and all(map(is_point, object_))


def is_mix(object_: Any) -> bool:
    return (isinstance(object_, tuple)
            and is_multipoint(object_[0])
            and is_multisegment(object_[1])
            and is_multipolygon(object_[2]))


def is_multipolygon(object_: Any) -> bool:
    return isinstance(object_, list) and all(map(is_polygon, object_))


def is_multisegment(object_: Any) -> bool:
    return isinstance(object_, list) and all(map(is_segment, object_))


def is_multicontour(object_: Any) -> bool:
    return isinstance(object_, list) and all(map(is_contour, object_))


is_point = Point.__instancecheck__


def is_polygon(object_: Any) -> bool:
    return (isinstance(object_, tuple)
            and len(object_) == 2
            and is_contour(object_[0])
            and is_multicontour(object_[1]))


def is_polyline(object_: Any) -> bool:
    return (isinstance(object_, list)
            and len(object_) >= MIN_POLYLINE_SIZE
            and all(map(is_point, object_)))


def is_segment(object_: Any) -> bool:
    return (isinstance(object_, tuple)
            and len(object_) == 2
            and all(map(is_point, object_))
            and len(set(object_)) == 2)


def is_non_self_intersecting_contour(contour: Contour) -> bool:
    return not edges_intersect(context.contour_cls(contour))


def is_star_contour(contour: Contour) -> bool:
    return segments_do_not_cross_or_overlap(
            contour_to_star_multisegment(contour)
            + contour_to_multisegment(contour))


def contour_to_star_multisegment(contour: Contour) -> Multisegment:
    centroid = contour_to_centroid(contour)
    return [(centroid, vertex)
            for vertex in contour
            if vertex != centroid]


def mix_segments_do_not_cross_or_overlap(mix: Mix) -> bool:
    _, multisegment, multipolygon = mix
    return segments_do_not_cross_or_overlap(
            multisegment
            + list(flatten(chain(contour_to_multisegment(border),
                                 flatten(contour_to_multisegment(hole)
                                         for hole in holes))
                           for border, holes in multipolygon)))


def contours_do_not_cross_or_overlap(contours: List[Contour]) -> bool:
    return segments_do_not_cross_or_overlap(list(flatten(
            contour_to_multisegment(contour) for contour in contours)))


def segments_do_not_cross_or_overlap(segments: List[Segment]) -> bool:
    return not segments_cross_or_overlap([context.segment_cls(
            context.point_cls(*start), context.point_cls(*end))
        for start, end in segments])


def is_multicontour_strict(multicontour: Multicontour) -> bool:
    return all(is_contour_strict(contour) for contour in multicontour)
