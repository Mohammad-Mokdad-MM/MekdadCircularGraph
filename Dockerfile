FROM debian:bookworm-slim

LABEL org.opencontainers.image.title="Mekdad Circular Graph" \
      org.opencontainers.image.description="Interactive and headless brain connectivity visualization" \
      org.opencontainers.image.source="https://github.com/Mohammad-Mokdad-MM/MekdadCircularGraph" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        python3 \
        python3-tk \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv "$VIRTUAL_ENV"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /output /home/appuser/.config/matplotlib \
    && chown -R appuser:appuser /output /home/appuser

USER appuser

ENTRYPOINT ["python", "main.py"]
CMD ["--output", "/output/circular_graph.png"]
