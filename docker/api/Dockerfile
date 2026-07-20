FROM python:3.14-slim@sha256:cea0e6040540fb2b965b6e7fb5ffa00871e632eef63719f0ea54bca189ce14a6

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OBESITY_MODEL_RUN_DIRECTORY=/models/current \
    OBESITY_PROMOTION_REPORT=/governance/promotion_report.json \
    OBESITY_REQUIRE_PROMOTION=true

WORKDIR /app

COPY requirements-serving.txt pyproject.toml README.md /app/

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements-serving.txt

COPY src /app/src

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps . \
    && addgroup --system inference \
    && adduser --system --ingroup inference --home /home/inference inference

USER inference

EXPOSE 8000

CMD ["obesity-serve"]
