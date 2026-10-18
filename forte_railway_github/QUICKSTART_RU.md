# Быстрый запуск

1. Загрузи этот проект в **приватный GitHub-репозиторий**.
2. В Railway нажми **Deploy from GitHub**, выбери репозиторий.
3. Задай переменные `FORTE_API_KEY`, `FORTE_WORKER_KEY`, `FORTE_STORAGE_DIR=/data`, `FORTE_SAVE_VIDEOS=1` (или `0` для автоматического удаления файлов после обработки).
4. Прикрепи Railway Volume к `/data`, создай публичный домен и проверь `/health`.
5. На своём сервере V100 в активном CUDA-окружении задай `FORTE_GATEWAY_URL`, `FORTE_WORKER_KEY`, `FORTE_DETECTOR_WEIGHTS`, `FORTE_ACTION_CHECKPOINT` и запусти:

```bash
CUDA_VISIBLE_DEVICES=0 python -m worker.run_worker
```

6. Открой сайт Railway, введи клиентский `FORTE_API_KEY`, загрузи `.mov`, дождись обработки. В «Истории» можно открыть результат и скачать JSON. Видео скачивается, только если включено хранение.

**Все 10 action labels читаются из твоего обученного `best.pt` автоматически.** YOLO26 считывает объекты из своего detector checkpoint. На Railway самих GPU-моделей нет.

## Запуск без GPU (CPU: OpenVINO / ONNX)

Если V100 недоступен, оба модели можно сконвертировать под CPU и запускать worker на обычном процессоре (вплоть до отдельного сервиса Railway).

1. На машине, где ещё лежат обученные `best.pt`, один раз запусти экспорт:

```bash
pip install -r requirements-worker-cpu.txt   # сначала CPU-версии torch/torchvision, см. шапку файла
python -m worker.vision.export \
  --detector /путь/yolo26/weights/best.pt \
  --action   /путь/action/e02_mobilenet_ce_strong/best.pt \
  --out models --format openvino --imgsz 640     # или --format onnx / both
```

2. Укажи worker-у сконвертированные файлы и запусти без CUDA:

```bash
export FORTE_DEVICE=cpu
export FORTE_DETECTOR_IMGSZ=640                   # как в --imgsz при экспорте
export FORTE_DETECTOR_WEIGHTS="models/best_openvino_model"  # или models/best.onnx
export FORTE_ACTION_CHECKPOINT="models/action.xml"         # или models/action.onnx
python -m worker.run_worker
```

Backend для action-модели выбирается по расширению (`.pt`→PyTorch, `.onnx`→ONNX Runtime, `.xml`→OpenVINO); класс/кадры/размер берутся из сопутствующего `*.meta.json`. На CPU одно видео обрабатывается заметно дольше, чем на V100 — держи `FORTE_MAX_DURATION_SEC` в пределах бюджета ~300 c и замерь время.

**Внимание:** перед загрузкой реальных видео ForteBank в сторонний Railway проверь условия соревнования о передаче и хранении файлов у третьих лиц. Для тестовых роликов это может быть запрещено.
