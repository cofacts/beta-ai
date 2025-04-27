FROM python:3.13-slim

WORKDIR /app

# Install uv and dependencies
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv sync

# Copy source code
COPY . .

# Create a non-root user
RUN adduser --disabled-password --gecos "" myuser && \
    chown -R myuser:myuser /app
USER myuser

ENV PATH="/home/myuser/.local/bin:$PATH"
ENV PORT=8080

CMD uv run main.py
