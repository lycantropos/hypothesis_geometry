ARG IMAGE_NAME
ARG IMAGE_VERSION

FROM ${IMAGE_NAME}:${IMAGE_VERSION}

RUN pip install --upgrade pip setuptools

WORKDIR /opt/hypothesis_geometry

COPY requirements.txt .
RUN pip install --force-reinstall -r requirements.txt

COPY requirements-tests.txt .
RUN pip install --force-reinstall -r requirements-tests.txt

COPY README.md .
COPY pytest.ini .
COPY setup.py .
COPY hypothesis_geometry hypothesis_geometry/
COPY tests/ tests/
