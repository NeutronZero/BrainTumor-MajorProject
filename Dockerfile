# BrainTumor-MajorProject API — minimal reproducible runtime (REL-002).
#
# Build:  docker build -t braintumor-api:REL-002 .
# Run:    docker run --rm -p 8000:8000 -v ./checkpoints:/app/checkpoints:ro -v ./outputs:/app/outputs:ro braintumor-api:REL-002
# Health: curl http://localhost:8000/health  (status "ok" requires the CLS/SEG
#         checkpoints + frozen calibration/metrics artifacts, mounted read-only)
#
# Rollback = redeploy the previous immutable image tag (docs/deployment/profiles.md).
# Intentionally NOT copied into the image: checkpoints/ and outputs/ (multi-GB,
# LFS-backed, release-hashed). The image ships code + locked deps only; runtime
# artifacts are supplied via read-only mounts so artifact identity stays
# controlled by the release manifest, not the image build cache.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer: installed from the regenerated REL-002 lock. The app
# import path (starlette TestClient transport, pandas/streamlit resolution)
# is already proven against this exact pin set by the clean-venv gate.
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# Application layer (source only; heavy artifacts are runtime mounts)
COPY app/ ./app/
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/reproducibility_check.py ./scripts/reproducibility_check.py

EXPOSE 8000

# Self-audit at boot: identical to the clean-checkout property proven in G2
# (import application + load checkpoints). Fails fast on missing artifacts.
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--app-dir", "app/api", "--host", "0.0.0.0", "--port", "8000"]
