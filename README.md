hypothesis_geometry
===================

[![](https://dev.azure.com/lycantropos/hypothesis_geometry/_apis/build/status/lycantropos.hypothesis_geometry?branchName=master)](https://dev.azure.com/lycantropos/hypothesis_geometry/_build/latest?definitionId=19&branchName=master "Azure Pipelines")
[![](https://readthedocs.org/projects/hypothesis_geometry/badge/?version=latest)](https://hypothesis-geometry.readthedocs.io/en/latest "Documentation")
[![](https://codecov.io/gh/lycantropos/hypothesis_geometry/branch/master/graph/badge.svg)](https://codecov.io/gh/lycantropos/hypothesis_geometry "Codecov")
[![](https://img.shields.io/github/license/lycantropos/hypothesis_geometry.svg)](https://github.com/lycantropos/hypothesis_geometry/blob/master/LICENSE "License")
[![](https://badge.fury.io/py/hypothesis-geometry.svg)](https://badge.fury.io/py/hypothesis-geometry "PyPI")

In what follows `python` is an alias for `python3.5` or `pypy3.5`
or any later version (`python3.6`, `pypy3.6` and so on).

Installation
------------

Install the latest `pip` & `setuptools` packages versions
```bash
python -m pip install --upgrade pip setuptools
```

### User

Download and install the latest stable version from `PyPI` repository:
```bash
python -m pip install --upgrade hypothesis_geometry
```

### Developer

Download the latest version from `GitHub` repository
```bash
git clone https://github.com/lycantropos/hypothesis_geometry.git
cd hypothesis_geometry
```

Install dependencies
```bash
python -m pip install --force-reinstall -r requirements.txt
```

Install
```bash
python setup.py install
```

Usage
-----
With setup
```python
>>> from ground.base import get_context
>>> from hypothesis import strategies
>>> from hypothesis_geometry import planar
>>> context = get_context()
>>> Contour = context.contour_cls
>>> Empty = context.empty_cls
>>> Mix = context.mix_cls
>>> Multipoint = context.multipoint_cls
>>> Multipolygon = context.multipolygon_cls
>>> Multisegment = context.multisegment_cls
>>> Point = context.point_cls
>>> Polygon = context.polygon_cls
>>> Segment = context.segment_cls
>>> min_coordinate, max_coordinate = -100, 100
>>> coordinates_type = int
>>> coordinates = strategies.integers(min_coordinate, max_coordinate)
>>> import warnings
>>> from hypothesis.errors import NonInteractiveExampleWarning
>>> # ignore hypothesis warnings caused by `example` method call
... warnings.filterwarnings('ignore', category=NonInteractiveExampleWarning)

```
let's take a look at what can be generated and how.

### Empty geometries
```python
>>> empty_geometries = planar.empty_geometries()
>>> empty = empty_geometries.example()
>>> isinstance(empty, Empty)
True

```

### Points
```python
>>> points = planar.points(coordinates)
>>> point = points.example()
>>> isinstance(point, Point)
True
>>> (isinstance(point.x, coordinates_type)
...  and isinstance(point.y, coordinates_type))
True
>>> (min_coordinate <= point.x <= max_coordinate
...  and min_coordinate <= point.y <= max_coordinate)
True

```

### Multipoints
```python
>>> min_size, max_size = 5, 10
>>> multipoints = planar.multipoints(coordinates,
...                                  min_size=min_size,
...                                  max_size=max_size)
>>> multipoint = multipoints.example()
>>> isinstance(multipoint, Multipoint)
True
>>> min_size <= len(multipoint.points) <= max_size
True
>>> all(isinstance(point.x, coordinates_type)
...     and isinstance(point.y, coordinates_type)
...     for point in multipoint.points)
True
>>> all(min_coordinate <= point.x <= max_coordinate
...     and min_coordinate <= point.y <= max_coordinate
...     for point in multipoint.points)
True

```

### Segments
```python
>>> segments = planar.segments(coordinates)
>>> segment = segments.example()
>>> isinstance(segment, Segment)
True
>>> (isinstance(segment.start.x, coordinates_type)
...  and isinstance(segment.start.y, coordinates_type)
...  and isinstance(segment.end.x, coordinates_type)
...  and isinstance(segment.end.y, coordinates_type))
True
>>> (min_coordinate <= segment.start.x <= max_coordinate 
...  and min_coordinate <= segment.start.y <= max_coordinate
...  and min_coordinate <= segment.end.x <= max_coordinate 
...  and min_coordinate <= segment.end.y <= max_coordinate)
True

```

### Multisegments
```python
>>> min_size, max_size = 5, 10
>>> multisegments = planar.multisegments(coordinates, 
...                                      min_size=min_size,
...                                      max_size=max_size)
>>> multisegment = multisegments.example()
>>> isinstance(multisegment, Multisegment)
True
>>> min_size <= len(multisegment.segments) <= max_size
True
>>> all(isinstance(segment.start.x, coordinates_type)
...     and isinstance(segment.start.y, coordinates_type)
...     and isinstance(segment.end.x, coordinates_type)
...     and isinstance(segment.end.y, coordinates_type)
...     for segment in multisegment.segments)
True
>>> all(min_coordinate <= segment.start.x <= max_coordinate
...     and min_coordinate <= segment.start.y <= max_coordinate
...     and min_coordinate <= segment.end.x <= max_coordinate
...     and min_coordinate <= segment.end.y <= max_coordinate
...     for segment in multisegment.segments)
True

```

### Contours
```python
>>> min_size, max_size = 5, 10
>>> contours = planar.contours(coordinates, 
...                            min_size=min_size,
...                            max_size=max_size)
>>> contour = contours.example()
>>> isinstance(contour, Contour)
True
>>> min_size <= len(contour.vertices) <= max_size
True
>>> all(isinstance(vertex.x, coordinates_type)
...     and isinstance(vertex.y, coordinates_type)
...     for vertex in contour.vertices)
True
>>> all(min_coordinate <= vertex.x <= max_coordinate
...     and min_coordinate <= vertex.y <= max_coordinate
...     for vertex in contour.vertices)
True

```
also `planar.concave_contours` & `planar.convex_contours` options are available.

### Multicontours
```python
>>> min_size, max_size = 5, 10
>>> min_contour_size, max_contour_size = 4, 8
>>> multicontours = planar.multicontours(coordinates, 
...                                      min_size=min_size,
...                                      max_size=max_size,
...                                      min_contour_size=min_contour_size,
...                                      max_contour_size=max_contour_size)
>>> multicontour = multicontours.example()
>>> isinstance(multicontour, list)
True
>>> all(isinstance(contour, Contour) for contour in multicontour)
True
>>> min_size <= len(multicontour) <= max_size
True
>>> all(min_contour_size <= len(contour.vertices) <= max_contour_size
...     for contour in multicontour)
True
>>> all(isinstance(vertex.x, coordinates_type)
...     and isinstance(vertex.y, coordinates_type)
...     for contour in multicontour
...     for vertex in contour.vertices)
True
>>> all(min_coordinate <= vertex.x <= max_coordinate
...     and min_coordinate <= vertex.y <= max_coordinate
...     for contour in multicontour
...     for vertex in contour.vertices)
True

```

### Polygons
```python
>>> min_size, max_size = 5, 10
>>> min_holes_size, max_holes_size = 1, 3
>>> min_hole_size, max_hole_size = 4, 8
>>> polygons = planar.polygons(coordinates, 
...                            min_size=min_size,
...                            max_size=max_size,
...                            min_holes_size=min_holes_size,
...                            max_holes_size=max_holes_size,
...                            min_hole_size=min_hole_size,
...                            max_hole_size=max_hole_size)
>>> polygon = polygons.example()
>>> isinstance(polygon, Polygon)
True
>>> min_size <= len(polygon.border.vertices) <= max_size
True
>>> min_holes_size <= len(polygon.holes) <= max_holes_size
True
>>> all(min_hole_size <= len(hole.vertices) <= max_hole_size for hole in polygon.holes)
True
>>> polygon_contours = [polygon.border, *polygon.holes]
>>> all(isinstance(vertex.x, coordinates_type)
...     and isinstance(vertex.y, coordinates_type)
...     for contour in polygon_contours
...     for vertex in contour.vertices)
True
>>> all(min_coordinate <= vertex.x <= max_coordinate
...     and min_coordinate <= vertex.y <= max_coordinate
...     for contour in polygon_contours
...     for vertex in contour.vertices)
True

```

### Multipolygons
```python
>>> min_size, max_size = 2, 5
>>> min_border_size, max_border_size = 5, 10
>>> min_holes_size, max_holes_size = 1, 3
>>> min_hole_size, max_hole_size = 4, 8
>>> multipolygons = planar.multipolygons(coordinates, 
...                                      min_size=min_size,
...                                      max_size=max_size,
...                                      min_border_size=min_border_size,
...                                      max_border_size=max_border_size,
...                                      min_holes_size=min_holes_size,
...                                      max_holes_size=max_holes_size,
...                                      min_hole_size=min_hole_size,
...                                      max_hole_size=max_hole_size)
>>> multipolygon = multipolygons.example()
>>> isinstance(multipolygon, Multipolygon)
True
>>> min_size <= len(multipolygon.polygons) <= max_size
True
>>> all(min_border_size <= len(polygon.border.vertices) <= max_border_size
...     and min_holes_size <= len(polygon.holes) <= max_holes_size
...     and all(min_hole_size <= len(hole.vertices) <= max_hole_size
...             for hole in polygon.holes)
...     for polygon in multipolygon.polygons)
True
>>> all(all(isinstance(vertex.x, coordinates_type)
...         and isinstance(vertex.y, coordinates_type)
...         for vertex in polygon.border.vertices)
...     and all(isinstance(vertex.x, coordinates_type)
...             and isinstance(vertex.y, coordinates_type)
...             for hole in polygon.holes
...             for vertex in hole.vertices)
...     for polygon in multipolygon.polygons)
True
>>> all(all(min_coordinate <= vertex.x <= max_coordinate
...         and min_coordinate <= vertex.y <= max_coordinate
...         for vertex in polygon.border.vertices)
...     and all(min_coordinate <= vertex.x <= max_coordinate
...             and min_coordinate <= vertex.y <= max_coordinate
...             for hole in polygon.holes
...             for vertex in hole.vertices)
...     for polygon in multipolygon.polygons)
True

```

### Mixes
```python
>>> min_points_size, max_points_size = 2, 3
>>> min_segments_size, max_segments_size = 1, 4
>>> min_polygons_size, max_polygons_size = 0, 5
>>> min_polygon_border_size, max_polygon_border_size = 5, 10
>>> min_polygon_holes_size, max_polygon_holes_size = 1, 4
>>> min_polygon_hole_size, max_polygon_hole_size = 3, 5
>>> mixes = planar.mixes(coordinates,
...                      min_points_size=min_points_size,
...                      max_points_size=max_points_size,
...                      min_segments_size=min_segments_size,
...                      max_segments_size=max_segments_size,
...                      min_polygons_size=min_polygons_size,
...                      max_polygons_size=max_polygons_size,
...                      min_polygon_border_size=min_polygon_border_size,
...                      max_polygon_border_size=max_polygon_border_size,
...                      min_polygon_holes_size=min_polygon_holes_size,
...                      max_polygon_holes_size=max_polygon_holes_size,
...                      min_polygon_hole_size=min_polygon_hole_size,
...                      max_polygon_hole_size=max_polygon_hole_size)
>>> mix = mixes.example()
>>> isinstance(mix, Mix)
True
>>> isinstance(mix.discrete, (Empty, Multipoint))
True
>>> points = [] if isinstance(mix.discrete, Empty) else mix.discrete.points
>>> min_points_size <= len(points) <= max_points_size
True
>>> all(isinstance(point.x, coordinates_type)
...     and isinstance(point.y, coordinates_type)
...     for point in points)
True
>>> all(min_coordinate <= point.x <= max_coordinate
...     and min_coordinate <= point.y <= max_coordinate
...     for point in points)
True
>>> isinstance(mix.linear, (Empty, Segment, Contour, Multisegment))
True
>>> segments = ([]
...             if isinstance(mix.linear, Empty)
...             else ([mix.linear]
...                   if isinstance(mix.linear, Segment)
...                   else (mix.linear.segments
...                         if isinstance(mix.linear, Multisegment)
...                         else context.contour_edges(mix.linear))))
>>> min_segments_size <= len(segments) <= max_segments_size
True
>>> all(isinstance(segment.start.x, coordinates_type)
...     and isinstance(segment.start.y, coordinates_type)
...     and isinstance(segment.end.x, coordinates_type)
...     and isinstance(segment.end.y, coordinates_type)
...     for segment in segments)
True
>>> all(min_coordinate <= segment.start.x <= max_coordinate
...     and min_coordinate <= segment.start.y <= max_coordinate
...     and min_coordinate <= segment.end.x <= max_coordinate
...     and min_coordinate <= segment.end.y <= max_coordinate
...     for segment in segments)
True
>>> isinstance(mix.shaped, (Empty, Polygon, Multipolygon))
True
>>> polygons = ([]
...             if isinstance(mix.shaped, Empty)
...             else ([mix.shaped]
...                   if isinstance(mix.shaped, Polygon)
...                   else mix.shaped.polygons))
>>> min_polygons_size <= len(polygons) <= max_polygons_size
True
>>> all(min_polygon_border_size
...     <= len(polygon.border.vertices)
...     <= max_polygon_border_size
...     and (min_polygon_holes_size
...          <= len(polygon.holes)
...          <= max_polygon_holes_size)
...     and all(min_polygon_hole_size
...             <= len(hole.vertices)
...             <= max_polygon_hole_size
...             for hole in polygon.holes)
...     for polygon in polygons)
True
>>> all(all(isinstance(vertex.x, coordinates_type)
...         and isinstance(vertex.y, coordinates_type)
...         for vertex in polygon.border.vertices)
...     and all(isinstance(vertex.x, coordinates_type)
...             and isinstance(vertex.y, coordinates_type)
...             for hole in polygon.holes
...             for vertex in hole.vertices)
...     for polygon in polygons)
True
>>> all(all(min_coordinate <= vertex.x <= max_coordinate
...         and min_coordinate <= vertex.y <= max_coordinate
...         for vertex in polygon.border.vertices)
...     and all(min_coordinate <= vertex.x <= max_coordinate
...             and min_coordinate <= vertex.y <= max_coordinate
...             for hole in polygon.holes
...             for vertex in hole.vertices)
...     for polygon in polygons)
True

```

#### Caveats
- Strategies may be slow depending on domain,
so it may be necessary to add `HealthCheck.filter_too_much`, `HealthCheck.too_slow`
in [`suppress_health_check`](https://hypothesis.readthedocs.io/en/latest/settings.html#hypothesis.settings.suppress_health_check) 
and set [`deadline`](https://hypothesis.readthedocs.io/en/latest/settings.html#hypothesis.settings.deadline) to `None`.

- Unbounded floating point strategies for coordinates 
(like [`hypothesis.strategies.floats`](https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.floats)
with unset `min_value`/`max_value`) do not play well with bounded sizes 
and may cause a lot of searching iterations with no success,
so it is recommended to use bounded floating point coordinates with bounded sizes
or unbounded coordinates with unbounded sizes.

- [`decimal.Decimal`](https://docs.python.org/library/decimal.html) coordinates are not supported, because 
they seem to be too hard to work with correctly (e.g. sometimes self-intersecting contours arise), 
so it is suggested to use `float`  or [`fractions.Fraction`](https://docs.python.org/library/fractions.html) instead.

Development
-----------

### Bumping version

#### Preparation

Install
[bump2version](https://github.com/c4urself/bump2version#installation).

#### Pre-release

Choose which version number category to bump following [semver
specification](http://semver.org/).

Test bumping version
```bash
bump2version --dry-run --verbose $CATEGORY
```

where `$CATEGORY` is the target version number category name, possible
values are `patch`/`minor`/`major`.

Bump version
```bash
bump2version --verbose $CATEGORY
```

This will set version to `major.minor.patch-alpha`. 

#### Release

Test bumping version
```bash
bump2version --dry-run --verbose release
```

Bump version
```bash
bump2version --verbose release
```

This will set version to `major.minor.patch`.

### Running tests

Install dependencies
```bash
python -m pip install --force-reinstall -r requirements-tests.txt
```

Plain
```bash
pytest
```

Inside `Docker` container:
- with `CPython`
  ```bash
  docker-compose --file docker-compose.cpython.yml up
  ```
- with `PyPy`
  ```bash
  docker-compose --file docker-compose.pypy.yml up
  ```

`Bash` script (e.g. can be used in `Git` hooks):
- with `CPython`
  ```bash
  ./run-tests.sh
  ```
  or
  ```bash
  ./run-tests.sh cpython
  ```

- with `PyPy`
  ```bash
  ./run-tests.sh pypy
  ```

`PowerShell` script (e.g. can be used in `Git` hooks):
- with `CPython`
  ```powershell
  .\run-tests.ps1
  ```
  or
  ```powershell
  .\run-tests.ps1 cpython
  ```
- with `PyPy`
  ```powershell
  .\run-tests.ps1 pypy
  ```
