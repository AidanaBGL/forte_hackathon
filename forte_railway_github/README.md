# Forte Vision — GitHub → Railway + V100 worker

A functioning deployment starter for the **ForteBank Datathon 26** asynchronous video API, web UI, and a **separate GPU inference worker**. This repository reuses the current project's actual YOLO26 object detector and the trained Action Recognition checkpoint; it does **not** fabricate action classes or results.

**Important:** Railway hosts the UI/API/queue/files, **not your GPU models**. Your V100 server must run `python -m worker.run_worker` continuously to process videos. The Railway API can accept videos with the worker offline, but will not produce predictions until the worker is running.

## Architecture

```text
User / Datathon jury
    |
    v
Railway HTTPS API + Web UI (CPU-only Docker container)
    |  POST /api/v1/analyze -> job_id
    |  SQLite job queue + JSON history on Railway Volume /data
    |  .mov/.mp4 stored temporarily; optionally retained
    |
    v
GPU Worker on YOUR server (polls Railway over outgoing HTTPS)
    |  downloads private video using worker key + one-time job lease
    |  YOLO26 detector: cash, cash_bag, counting_machine, person,
    |                   id, open/close laptop, open/close door, safe etc.
    |  process classifier: 0 receipt, 1 counting, 2 entry, 3 exit, 4 cash bag
    |  Action model: loads action label names FROM best.pt['classes']
    |  approximate temporal action windows
    |  visual relations, configurable process-order hypothesis checks
    v
JSON: violation/no_violation, type, timestamp_sec, 1–3 JPEG base64 frames
    |
    v
Railway: GET /api/v1/jobs/{job_id}, history + downloadable reports
```

The action checkpoint's actual classes (such as `cash_handover`, `document_handover`, `cash_storage`, `cash_counting_machine`, `safe_closing`, `cashier_leaving`, `cashbag_handover`, `computer_interaction`, `open door`, `close door`) are used automatically. Their exact spellings come from the checkpoint. **No model needs to be retrained just to use Railway.**

### Important scientific limitations

* Action classifiers trained on labels spanning entire videos provide approximate action windows, not precise temporal segmentation.
* One class at a time can be predicted per action window by the current softmax model. Parallel actions are not explicitly supported.
* People/objects and their bounding boxes do not prove that a financial transaction was registered in the bank's accounting system.
* Rules in `config/procedure_graphs.json` include workflow **hypotheses** from video observations. They must not be represented as established ForteBank policy or automatically proven legal violations.
* The service meets the *shape* of the judging JSON API; accuracy and the <=300-second processing target need to be measured on your V100 and network.

## 1. Push to a NEW PRIVATE GitHub repository

```bash
unzip forte_railway_github.zip
cd forte_railway_github

git init
git add .
git commit -m "Initial Railway API + V100 worker"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_PRIVATE_REPOSITORY.git
git push -u origin main
```

Do **not** commit videos, trained checkpoints, secrets, or bank recordings. The `.gitignore` excludes these. Keep the GitHub repository private.

## 2. Railway setup (one service)

1. Railway → **New Project → Deploy from GitHub Repo** → select your private repo.
2. It will build the included `Dockerfile` with **only FastAPI dependencies**, not PyTorch.
3. In **Variables**, set unique secrets (generate locally with `python -c "import secrets;print(secrets.token_urlsafe(36))"`):

   ```env
   FORTE_API_KEY=<random-strong-secret-for-jury-and-website>
   FORTE_WORKER_KEY=<different-random-strong-secret-for-V100-worker>
   FORTE_STORAGE_DIR=/data
   FORTE_SAVE_VIDEOS=0
   FORTE_MAX_UPLOAD_MB=150
   FORTE_MAX_PENDING=10
   FORTE_JOB_TIMEOUT_SEC=295
   ```

4. Under **Volumes** attach one persistent Railway Volume, mount path **`/data`**. **Required** if you want uploaded video queues, history and JSON reports to survive redeploys. Do not scale the service to multiple replicas when using SQLite.
5. Under **Settings → Networking → Generate Domain** get `https://...up.railway.app`.
6. In Deploy settings, use one replica, health check `/health`. `railway.json` sets the health check, and the Docker image listens on Railway's `$PORT`.
7. Test `https://...up.railway.app/health` and open `/` in a browser. Enter the `FORTE_API_KEY` in the UI.

**Storing videos:** set `FORTE_SAVE_VIDEOS=1` if you want the original `.mov/.mp4` kept and downloadable from history. With `0`, video stays only until the GPU worker finishes; **JSON reports remain saved**. Completed jobs can be deleted by API/UI. Be mindful of video storage cost and retention obligations.

**Security:** one shared `X-API-Key` controls access to all reports. This is acceptable as a Datathon prototype but is not multi-tenant user authentication. Do not make the key public. The browser stores it in `sessionStorage` for that tab only. The `FORTE_WORKER_KEY` must **never** appear in the browser, GitHub, or a public log.

## 3. Start the GPU worker on your V100 server

The worker runs on your existing CUDA 12.6 PyTorch 2.6.0 environment. It makes only **outbound HTTPS** calls to Railway; you do not need an inbound public port on the GPU machine.

```bash
cd forte_railway_github
source /path/to/your/.venv/bin/activate

# Use your already working torch 2.6.0+cu126 and torchvision 0.21.0+cu126.
python -m pip install -r requirements-worker.txt

export FORTE_GATEWAY_URL="https://YOUR-SERVICE.up.railway.app"
export FORTE_WORKER_KEY="THE_SAME_WORKER_SECRET_FROM_RAILWAY"
export FORTE_DETECTOR_WEIGHTS="/absolute/path/yolo26/weights/best.pt"
export FORTE_ACTION_CHECKPOINT="/absolute/path/action_experiments/e02_mobilenet_ce_strong/best.pt"
export FORTE_DEVICE=0
export FORTE_JOB_TIMEOUT_SEC=265

CUDA_VISIBLE_DEVICES=0 python -m worker.run_worker
```

The worker first checks that the detector and action checkpoints exist. It loads them once and uses them for all subsequent jobs. It reports its **actual detector and action label names** through a protected heartbeat, so `GET /api/v1/model-status` can show whether the GPU worker is currently online.

**If you see `Torch 2.14.1+cu130`** on V100, use your working `2.6.0+cu126` environment; do not change the system NVIDIA driver to address the wrong wheel.

**Checkpoint compatibility:** The code expects the checkpoint produced by the earlier `forte_action_benchmark` (`state_dict`, `classes`, `model`, `frames`, `size`, `normalization`). `best.pt` from a different architecture requires a matching loader. It is not interchangeable with the YOLO `.pt`.

On a shared GPU, make sure enough memory is available. You can first run `python -m worker.run_worker --once` to process one queued video, then run continuously via a supervised process (systemd/tmux/supervisor). For background production, use a process manager to restart the worker after a crash.

### 3b. CPU-only worker (ONNX / OpenVINO) — no GPU required

The same worker can run on a plain CPU (including a second Railway service) after you
convert the two models to **OpenVINO** or **ONNX**. Run the exporter **once** on a machine
that still has the trained `best.pt` files plus `torch`/`ultralytics`:

```bash
python -m pip install -r requirements-worker-cpu.txt
# install CPU torch/torchvision first (see the file header)

python -m worker.vision.export \
  --detector /path/yolo26/weights/best.pt \
  --action   /path/action_experiments/e02_mobilenet_ce_strong/best.pt \
  --out models --format openvino --imgsz 640      # or --format onnx / both
```

This writes `models/best_openvino_model/` (detector), `models/action.xml` (+ `action.meta.json`),
and/or `.onnx` equivalents. Then start the worker pointing at the exported artifacts:

```bash
export FORTE_DEVICE=cpu
export FORTE_DETECTOR_IMGSZ=640                    # must match --imgsz used at export
export FORTE_DETECTOR_WEIGHTS="models/best_openvino_model"   # or models/best.onnx
export FORTE_ACTION_CHECKPOINT="models/action.xml"          # or models/action.onnx
# optional: FORTE_ACTION_BACKEND=auto|torch|onnx|openvino, FORTE_CPU_THREADS=<N>
python -m worker.run_worker
```

The detector loads through Ultralytics (which natively serves `.onnx` / `_openvino_model`).
The action net picks a backend from the file extension: `.pt`→PyTorch, `.onnx`→ONNX Runtime,
`.xml`→OpenVINO. No PyTorch is used at inference time for the ONNX/OpenVINO action backends —
all metadata comes from the `*.meta.json` sidecar. Expect a noticeably larger per-video wall
time than the V100; keep `FORTE_MAX_DURATION_SEC` within the ~300 s contest budget and measure.

## 4. Jury API exactly as required

```bash
# Asynchronous upload returns job_id (HTTP 202)
curl -X POST 'https://YOUR-SERVICE.up.railway.app/api/v1/analyze' \
  -H 'X-API-Key: YOUR_CLIENT_KEY' \
  -F 'file=@X-1.mov'

# Poll until done or failed
curl -H 'X-API-Key: YOUR_CLIENT_KEY' \
  'https://YOUR-SERVICE.up.railway.app/api/v1/jobs/JOB_ID'
```

A successful response is shaped as:

```json
{
  "job_id": "d4138cf62da129ac",
  "status": "done",
  "verdict": "no_violation",
  "violations": []
}
```

For suspected violations, `violations` contains `type`, `timecode_sec`, `description`, and `evidence_frames` with `{timestamp_sec, image_base64}`. **This is a format example, not a claim about any particular video.**

Additional protected endpoints:

- `GET /api/v1/jobs/{job_id}/details` — process name, actions/timelines, YOLO counts and timing.
- `GET /api/v1/history` — all jobs, timestamps, status.
- `GET /api/v1/jobs/{job_id}/report` — download complete JSON result (with action details).
- `GET /api/v1/jobs/{job_id}/video` — download video only if retention enabled.
- `DELETE /api/v1/jobs/{job_id}` — delete completed or failed job, report and video.
- `GET /health` — gateway health, **not** an indicator that GPU worker is connected.

### Five processes

| Process | Signal currently used |
|--|--|
| 0 Client cash receipt | two people visible for much of the video |
| 1 Internal counting | one person, counting-machine evidence |
| 2 Cashier entry | transition absent → present |
| 3 Cashier exit | transition present → absent |
| 4 Cash bag processing | trained YOLO26 `cash_bag` class, detected in multiple frames |

The process selection is a set of scoring heuristics, not a trained five-class model. `cash_bag` is **not** detected by green-color thresholding.

## 5. Data and competition restrictions — READ BEFORE PUBLIC DEPLOY

**The uploaded Datathon specification says the provided videos must be used only within the competition, and testing videos must not be published or transferred to third parties.** Routing restricted footage through Railway (third-party cloud hosting and storage) could be disallowed, **even if all ML inference runs locally on your V100**. Confirm authorization for this arrangement before uploading bank footage or jury test videos to Railway. The no-cloud-AI-inference rule is separate from restrictions on transmitting video data.

For material that cannot leave the permitted environment, run the **entire web/API/inference stack on the approved on-premises server**, and use a suitable authorized networking arrangement instead. The earlier `forte_vision_fullstack.zip` implements local inference behind the same public endpoints. Do not send restricted footage to Railway without permission.

## 6. Local API tests (do not need CUDA)

```bash
python -m pip install -r requirements-test.txt
pytest -q tests/
```

A complete API upload→claim→download→report→history roundtrip is tested with mock inference. **GPU model inference cannot be verified until you provide actual checkpoints on the V100**.

## 7. Deployment and operations notes

* The Railway filesystem outside a mounted volume is ephemeral; use `/data` for persistent SQLite, videos, and reports.
* SQLite + one mounted volume is deliberately simple for a 5–10-request demo. For production scale and multiple replicas, replace it with PostgreSQL and an object storage backend; add user auth, encryption, audit logs, retention and access controls.
* If the worker is offline, jobs remain queued (the contest timeout causes them to fail after ~295 seconds). No response is fabricated.
* Very long clips and large vision models need throughput testing. The GPU worker's inference timeout is 265 seconds; remaining time covers transfer and queue overhead.
* Railway's public HTTP upload must complete within its network limit; keep `.mov` files reasonably small.
* Do not expose `/internal` routes to untrusted clients: they require a separate `X-Worker-Key`, and active jobs additionally require a private lease token.
