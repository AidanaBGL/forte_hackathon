FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# OpenCV + torch need these shared libs at runtime (ultralytics may pull full opencv-python,
# which links libGL/libxcb/X11). Installing the set avoids ImportError whack-a-mole.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 libgomp1 libgl1 libxcb1 libxext6 libxrender1 libsm6 \
      && rm -rf /var/lib/apt/lists/*

COPY requirements-railway.txt requirements-worker-cpu.txt /app/
# CPU torch/torchvision: required by ultralytics (detector). The ONNX action backend
# does not use torch at inference time, but ultralytics imports it regardless.
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-railway.txt -r requirements-worker-cpu.txt

COPY backend /app/backend
COPY web /app/web
COPY config /app/config
COPY worker /app/worker
# Exported CPU models (best.onnx / action.onnx + action.meta.json). Commit them to git
# so Railway's GitHub build includes them. See README section 3c.
COPY models /app/models
COPY start-allinone.sh /app/

# /data is a Railway persistent Volume (configured in dashboard); harmless if ephemeral.
RUN chmod +x /app/start-allinone.sh && mkdir -p /data && chmod 700 /data
EXPOSE 8000
CMD ["/app/start-allinone.sh"]
