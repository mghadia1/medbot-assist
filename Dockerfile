# Build from the parent projects directory:
# docker build -f medbot-assist/Dockerfile -t medbot-assist .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
COPY medbot-vision ./medbot-vision
COPY surgiarm-sim ./surgiarm-sim
COPY medbot-assist ./medbot-assist
ARG PYTORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir \
      --index-url ${PYTORCH_CPU_INDEX} \
      torch==2.6.0 \
    && python -m pip install --no-cache-dir \
    ./medbot-vision \
    ./surgiarm-sim \
    './medbot-assist[ml]'

WORKDIR /workspace/medbot-assist
ENTRYPOINT ["medbot-assist"]
CMD ["--help"]
