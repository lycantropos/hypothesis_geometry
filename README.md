hypothesis_geometry
===================

[![](https://travis-ci.com/lycantropos/hypothesis_geometry.svg?branch=master)](https://travis-ci.com/lycantropos/hypothesis_geometry "Travis CI")
[![](https://dev.azure.com/lycantropos/hypothesis_geometry/_apis/build/status/lycantropos.hypothesis_geometry?branchName=master)](https://dev.azure.com/lycantropos/hypothesis_geometry/_build/latest?branchName=master "Azure Pipelines")
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
>>> all(all(isinstance(coordinate, coordinates_type) 
...         for coordinate in endpoint)
...     for endpoint in segment)
True
>>> all(all(min_coordinate <= coordinate <= max_coordinate 
...         for coordinate in endpoint)
...     for endpoint in segment)
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
>>> all(all(isinstance(coordinate, coordinates_type)
...         for coordinate in vertex)
...     for vertex in polyline)
True
>>> all(all(min_coordinate <= coordinate <= max_coordinate 
...         for coordinate in vertex)
...     for vertex in polyline)
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
>>> all(all(isinstance(coordinate, coordinates_type)
...         for coordinate in vertex)
...     for vertex in contour)
True
>>> all(all(min_coordinate <= coordinate <= max_coordinate
...         for coordinate in vertex)
...     for vertex in contour)
True

```
also `planar.concave_contours` & `planar.convex_contours` options are available.

#### Caveats
- Strategies may be slow depending on domain,
so it may be necessary to add `HealthCheck.filter_too_much`, `HealthCheck.too_slow`
in [`suppress_health_check`](https://hypothesis.readthedocs.io/en/latest/settings.html#hypothesis.settings.suppress_health_check) 
and set [`deadline`](https://hypothesis.readthedocs.io/en/latest/settings.html#hypothesis.settings.deadline) to `None`.

- Unbounded floating point strategies for coordinates 
(like [`hypothesis.strategies.floats`](https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.floats)
or [`hypothesis.strategies.decimals`](https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.decimals)
with unset `min_value`/`max_value`) do not play well with bounded sizes 
and may cause a lot of searching iterations with no success,
so it is recommended to use bounded floating point coordinates with bounded sizes
or unbounded coordinates with unbounded sizes.

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
