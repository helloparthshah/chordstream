# Inference-only image: numpy + onnxruntime + FastAPI. No PyTorch, no GPU needed.
# Train locally, `chordstream export`, then build with the bundle in models/latest.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

COPY models/latest/config.json models/latest/model.onnx models/latest/init_state.npz models/latest/hmm.npz ./models/latest/

ENV CHORDSTREAM_BUNDLE=/app/models/latest
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "chordstream.server:app", "--host", "0.0.0.0", "--port", "8000"]
