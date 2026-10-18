#!/bin/bash
# All-in-one Railway entrypoint: the API/UI gateway AND the CPU inference worker
# run in the same container. The worker polls the gateway over loopback, so there is
# no second service and no public worker endpoint.
set -euo pipefail

: "${PORT:=8000}"
# The worker talks to the local gateway; run_worker allows http on 127.0.0.1.
export FORTE_GATEWAY_URL="http://127.0.0.1:${PORT}"
export FORTE_DEVICE="${FORTE_DEVICE:-cpu}"
export FORTE_WORKER_ID="${FORTE_WORKER_ID:-railway-cpu-0}"

echo "[start] launching gateway on :${PORT} and CPU worker (device=${FORTE_DEVICE})"

uvicorn backend.main:app --host 0.0.0.0 --port "${PORT}" --workers 1 &
api_pid=$!

# The worker loads the ONNX models once, then retries /internal/claim until the
# gateway is up (transport errors are caught and retried inside run_worker).
python -m worker.run_worker &
worker_pid=$!

# If either process exits, stop the other and let Railway restart the container.
trap 'kill "$api_pid" "$worker_pid" 2>/dev/null || true' EXIT
wait -n "$api_pid" "$worker_pid"
exit_code=$?
echo "[start] a process exited (code ${exit_code}); shutting down container"
exit "$exit_code"
