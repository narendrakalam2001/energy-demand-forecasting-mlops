# ============================================================
# DOCKERFILE — Energy Demand Forecast API
# ============================================================
# Multi-stage build:
#   builder  → install deps
#   runtime  → lean production image
# ============================================================

# ── Stage 1: builder ─────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements_api.txt

# ── Stage 2: runtime ─────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY src/        ./src/
COPY serving/    ./serving/
COPY services/   ./services/
COPY energy_models/ ./energy_models/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "serving.energy_demand_api:app", "--host", "0.0.0.0", "--port", "8000"]
