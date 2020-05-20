hypothesis_geometry
===================

[![](https://travis-ci.com/lycantropos/hypothesis_geometry.svg?branch=master)](https://travis-ci.com/lycantropos/hypothesis_geometry "Travis CI")
[![](https://dev.azure.com/lycantropos/hypothesis_geometry/_apis/build/status/lycantropos.hypothesis_geometry?branchName=master)](https://dev.azure.com/lycantropos/hypothesis_geometry/_build/latest?definitionId=19&branchName=master "Azure Pipelines")
[![](https://readthedocs.org/projects/hypothesis_geometry/badge/?version=latest)](https://hypothesis-geometry.readthedocs.io/en/latest "Documentation")
[![](https://codecov.io/gh/lycantropos/hypothesis_geometry/branch/master/graph/badge.svg)](https://codecov.io/gh/lycantropos/hypothesis_geometry "Codecov")
[![](https://img.shields.io/github/license/lycantropos/hypothesis_geometry.svg)](https://github.com/lycantropos/hypothesis_geometry/blob/master/LICENSE "License")
[![](https://badge.fury.io/py/hypothesis-geometry.svg)](https://badge.fury.io/py/hypothesis-geometry "PyPI")

In what follows
- `python` is an alias for `python3.5` or any later
version (`python3.6` and so on),
- `pypy` is an alias for `pypy3.5` or any later
version (`pypy3.6` and so on).

Installation
------------

Install the latest `pip` & `setuptools` packages versions:
- with `CPython`
  ```bash
  python -m pip install --upgrade pip setuptools
  ```
- with `PyPy`
  ```bash
  pypy -m pip install --upgrade pip setuptools
  ```

### User

Download and install the latest stable version from `PyPI` repository:
- with `CPython`
  ```bash
  python -m pip install --upgrade hypothesis_geometry
  ```
- with `PyPy`
  ```bash
  pypy -m pip install --upgrade hypothesis_geometry
  ```

### Developer

Download the latest version from `GitHub` repository
```bash
git clone https://github.com/lycantropos/hypothesis_geometry.git
cd hypothesis_geometry
```

Install dependencies:
- with `CPython`
  ```bash
  python -m pip install --force-reinstall -r requirements.txt
  ```
- with `PyPy`
  ```bash
  pypy -m pip install --force-reinstall -r requirements.txt
  ```

Install:
- with `CPython`
  ```bash
  python setup.py install
  ```
- with `PyPy`
  ```bash
  pypy setup.py install
  ```

Usage
-----
With setup
```python
>>> from hypothesis import strategies
>>> from hypothesis_geometry import planar
>>> min_coordinate, max_coordinate = -100, 100
>>> coordinates_type = int
>>> coordinates = strategies.integers(min_coordinate, max_coordinate)
>>> import warnings
>>> from hypothesis.errors import NonInteractiveExampleWarning
>>> # ignore hypothesis warnings caused by `example` method call
... warnings.filterwarnings('ignore', category=NonInteractiveExampleWarning)

```
let's take a look at what can be generated and how.

### Points
```python
>>> points = planar.points(coordinates)
>>> point = points.example()
>>> isinstance(point, tuple)
True
>>> len(point) == 2
True
>>> all(isinstance(coordinate, coordinates_type) for coordinate in point)
True
>>> all(min_coordinate <= coordinate <= max_coordinate for coordinate in point)
True

```

### Segments
```python
>>> segments = planar.segments(coordinates)
>>> segment = segments.example()
>>> isinstance(segment, tuple)
True
>>> len(segment) == 2
True
>>> all(isinstance(endpoint, tuple) for endpoint in segment)
True
>>> all(len(endpoint) == 2 for endpoint in segment)
True
>>> all(isinstance(coordinate, coordinates_type) 
...     for endpoint in segment
...     for coordinate in endpoint)
True
>>> all(min_coordinate <= coordinate <= max_coordinate 
...     for endpoint in segment
...     for coordinate in endpoint)
True

```

### Multisegments
```python
>>> min_size, max_size = 5, 10
>>> multisegments = planar.multisegments(coordinates, 
...                                      min_size=min_size,
...                                      max_size=max_size)
>>> multisegment = multisegments.example()
>>> isinstance(multisegment, list)
True
>>> min_size <= len(multisegment) <= max_size
True
>>> all(isinstance(segment, tuple)
...     for segment in multisegment)
True
>>> all(isinstance(endpoint, tuple)
...     for segment in multisegment
...     for endpoint in segment)
True
>>> all(len(segment) == 2 for segment in multisegment)
True
>>> all(len(endpoint) == 2
...     for segment in multisegment
...     for endpoint in segment)
True
>>> all(isinstance(coordinate, coordinates_type)
...     for segment in multisegment
...     for endpoint in segment
...     for coordinate in endpoint)
True
>>> all(min_coordinate <= coordinate <= max_coordinate
...     for segment in multisegment
...     for endpoint in segment
...     for coordinate in endpoint)
True

```

### Polylines
```python
>>> min_size, max_size = 5, 10
>>> polylines = planar.polylines(coordinates, 
...                              min_size=min_size,
...                              max_size=max_size)
>>> polyline = polylines.example()
>>> isinstance(polyline, list)
True
>>> min_size <= len(polyline) <= max_size
True
>>> all(isinstance(vertex, tuple) for vertex in polyline)
True
>>> all(len(vertex) == 2 for vertex in polyline)
True
>>> all(isinstance(coordinate, coordinates_type)
...     for vertex in polyline
...     for coordinate in vertex)
True
>>> all(min_coordinate <= coordinate <= max_coordinate 
...     for vertex in polyline
...     for coordinate in vertex)
True

```

### Contours
```python
>>> min_size, max_size = 5, 10
>>> contours = planar.contours(coordinates, 
...                            min_size=min_size,
...                            max_size=max_size)
>>> contour = contours.example()
>>> isinstance(contour, list)
True
>>> min_size <= len(contour) <= max_size
True
>>> all(isinstance(vertex, tuple) for vertex in contour)
True
>>> all(len(vertex) == 2 for vertex in contour)
True
>>> all(isinstance(coordinate, coordinates_type)
...     for vertex in contour
...     for coordinate in vertex)
True
>>> all(min_coordinate <= coordinate <= max_coordinate
...     for vertex in contour
...     for coordinate in vertex)
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
>>> all(isinstance(contour, list) for contour in multicontour)
True
>>> min_size <= len(multicontour) <= max_size
True
>>> all(min_contour_size <= len(contour) <= max_contour_size
...     for contour in multicontour)
True
>>> all(isinstance(vertex, tuple)
...     for contour in multicontour
...     for vertex in contour)
True
>>> all(len(vertex) == 2
...     for contour in multicontour
...     for vertex in contour)
True
>>> all(isinstance(coordinate, coordinates_type)
...     for contour in multicontour
...     for vertex in contour
...     for coordinate in vertex)
True
>>> all(min_coordinate <= coordinate <= max_coordinate
...     for contour in multicontour
...     for vertex in contour
...     for coordinate in vertex)
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
>>> isinstance(polygon, tuple)
True
>>> len(polygon) == 2
True
>>> border, holes = polygon
>>> isinstance(border, list)
True
>>> all(isinstance(hole, list) for hole in holes)
True
>>> min_size <= len(border) <= max_size
True
>>> min_holes_size <= len(holes) <= max_holes_size
True
>>> all(min_hole_size <= len(hole) <= max_hole_size for hole in holes)
True
>>> contours = [border, *holes]
>>> all(isinstance(vertex, tuple)
...     for contour in contours
...     for vertex in contour)
True
>>> all(len(vertex) == 2
...     for contour in contours
...     for vertex in contour)
True
>>> all(isinstance(coordinate, coordinates_type)
...     for contour in contours
...     for vertex in contour
...     for coordinate in vertex)
True
>>> all(min_coordinate <= coordinate <= max_coordinate
...     for contour in contours
...     for vertex in contour
...     for coordinate in vertex)
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

Install dependencies:
- with `CPython`
  ```bash
  python -m pip install --force-reinstall -r requirements-tests.txt
  ```
- with `PyPy`
  ```bash
  pypy -m pip install --force-reinstall -r requirements-tests.txt
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
