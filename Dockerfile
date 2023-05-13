ARG IMAGE_NAME
ARG IMAGE_VERSION

FROM ${IMAGE_NAME}:${IMAGE_VERSION}

WORKDIR /opt/hypothesis_geometry

COPY pyproject.toml .
COPY README.md .
COPY setup.py .
COPY hypothesis_geometry hypothesis_geometry
COPY tests tests

RUN pip install -e .[tests]
