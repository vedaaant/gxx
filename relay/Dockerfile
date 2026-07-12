FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pii.py ratelimit.py app.py ./
# package layout so `relay.app` imports resolve
RUN mkdir -p /app/relay && touch /app/relay/__init__.py && \
    cp pii.py ratelimit.py app.py /app/relay/
ENV PYTHONPATH=/app

# Required env at deploy time (Railway/Fly.io secrets):
#   CONTOUR_RELAY_TOKENS  comma-separated device tokens (empty = open dev mode)
#   TAVILY_API_KEY or SERPER_API_KEY   (search)
#   OPENAI_API_KEY    (opt-in cloud escalation)
#   CONTOUR_RELAY_RATE    per-token requests/minute (default 120)
EXPOSE 8080
CMD ["uvicorn", "relay.app:app", "--host", "0.0.0.0", "--port", "8080"]
