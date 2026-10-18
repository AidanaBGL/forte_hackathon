# ForteBank Datathon 26 — детекция нарушений кассовой дисциплины по видео

Пайплайн для анализа видео с рабочего места банковского кассира: модели находят объекты
и действия, сопоставляют их с каталогом процедур и выдают **кандидатов на нарушения** с
таймкодами и кадрами-доказательствами. Все инференсы локальные. Результат — **материал для
проверки человеком**, а не автоматический юридический вывод.

> ⚠️ Тестовые видео ForteBank нельзя использовать для обучения на стороне, публиковать или
> передавать третьим лицам. Храни данные локально.

---

## Содержание
1. [Архитектура](#архитектура)
2. [Структура репозитория](#структура-репозитория)
3. [Датасеты: откуда и куда класть](#датасеты)
4. [Тренировки](#тренировки)
5. [Экспорт моделей под CPU](#экспорт-моделей-под-cpu)
6. [Сборка и деплой сервиса](#сборка-и-деплой-сервиса)
7. [Переменные окружения и ключи](#переменные-окружения)

---

## Архитектура

Две независимые линии решения — обе сводятся к одному JSON-контракту жюри:

**A. Основной пайплайн (детектор + action-модель) → онлайн-сервис**

```
видео (.mov)
  └─ YOLO26 детектор (объекты: cash, cash_bag, counting_machine, safe, door, laptop, person, id, phone)
  └─ Action-модель (mobilenet_gru / r3d18 / ...): действия в скользящих окнах
        ↓
  геометрия отношений + классификация процесса + проверка гипотез порядка операций
        ↓
  verdict: violation / no_violation + список нарушений (type, timecode_sec, description, 3 кадра)
```

Обученные `best.pt` экспортируются в **ONNX/OpenVINO** и раздаются сервисом на CPU.

**B. VLM-baseline (Qwen):** мультимодальная модель анализирует кадры по промпту и каталогу
правил, возвращает тот же JSON. См. `run_fortebank_qwen.py`.

**Сервис (`forte_railway_github`)** — публичный асинхронный API + веб-UI:

```
Жюри / пользователь ──X-API-Key──▶  Gateway (FastAPI, очередь, UI)  ◀──loopback──  Worker (модели)
                                     POST /api/v1/analyze → job_id
                                     GET  /api/v1/jobs/{id} → verdict/violations
```

Gateway и worker могут жить в **одном контейнере** (CPU, ONNX/OpenVINO) или worker — отдельно
на GPU. Подробности — в [`forte_railway_github/README.md`](forte_railway_github/README.md).

---

## Структура репозитория

| Папка | Назначение |
|---|---|
| `forte_object_detection_benchmark/` | Обучение/бенчмарк **YOLO26-детектора** объектов (Roboflow YOLO-формат). |
| `forte_action_benchmark/` | Обучение/бенчмарк **action-recognition** (Roboflow Video COCO), 5 бэкбонов. |
| `forte_cv_experiments/` | Классические CV-бейзлайны (optical flow, motion, temporal LogisticRegression, E0–E5). |
| `forte_railway_github/` | Продакшн-**сервис**: FastAPI gateway + worker + экспорт моделей под CPU + деплой на Railway. |
| `run_fortebank_qwen.py` | **VLM-baseline** на Qwen: кадры + `fortebank_procedure_rules.json` + промпт → JSON. |
| `fortebank_procedure_rules.json` | Каталог процедур/правил (используется и сервисом, и Qwen). |
| `fortebank_qwen_prompt.md` | Промпт и строгая JSON-схема для VLM. |

---

## Датасеты

Данные **не публичные** — это материалы соревнования. Есть два источника/формата:

### 1. Roboflow-экспорты (для тренировок детектора и action)

Оба бенчмарка рассчитаны на экспорт из Roboflow:

- **Детекция** — экспорт в формате *YOLO detection*: `data.yaml` + `train/valid/test` с
  `images/*.jpg` и `labels/*.txt` (строка метки: `class_id cx cy w h`, нормализованные координаты).
- **Action** — экспорт в формате *Video COCO*: `_annotations.coco.json` + `.mp4/.mov`,
  метки лежат в `segments` (`images: []`/`annotations: []` для видео-экспорта — это норма).

Скачай ZIP из своего Roboflow-проекта и распакуй, например:

```bash
# детекция
unzip roboflow_detection_export.zip -d /path/to/det_dataset
# action
unzip roboflow_video_coco_export.zip -d /path/to/action_dataset
```

### 2. `videos.zip` + `open_labels.xlsx` (для CV-экспериментов и Qwen)

Исходные ролики и Excel с разметкой (бинарный GT + секунды события) выдаются **в чате
соревнования**. Положи их рядом со скриптами:

```
forte_cv_experiments/videos.zip
forte_cv_experiments/open_labels.xlsx
# и/или в корень для Qwen:
videos.zip            # run_fortebank_qwen.py сам распакует нужный ролик в ./videos/
```

> Имена нормализуются автоматически (кириллическая `Х-` → латинская `X-`). Точные дубликаты
> видео определяются по SHA-256 и не растекаются между train/val/test.

---

## Тренировки

Во всех бенчмарках torch/torchvision под твою CUDA ставятся **отдельно** — не переустанавливай
рабочий GPU-torch ради проекта.

### Детектор (YOLO26) — `forte_object_detection_benchmark/`

```bash
cd forte_object_detection_benchmark
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 1) аудит меток/сплитов (находит утечки кадров одного видео между сплитами)
python run.py inspect --data /path/to/det_dataset --output runs/audit

# 2) быстрый прогон 3 экспериментов
CUDA_VISIBLE_DEVICES=0 python run.py sweep \
  --data /path/to/det_dataset --config experiments.json \
  --output runs/detection --limit 3 --epochs 40 --batch 4 --device 0

# 3) полный грид (8 экспериментов), ранжирование по val mAP50-95
CUDA_VISIBLE_DEVICES=0 python run.py sweep \
  --data /path/to/det_dataset --output runs/detection_full --epochs 70 --batch 4 --device 0

# 4) оценка победителя ОДИН раз на held-out test
python run.py test --data /path/to/det_dataset \
  --checkpoint runs/detection_full/<winner>/weights/best.pt \
  --output runs/final_test --imgsz 640 --batch 4 --device 0
```

Победный чекпойнт: `runs/detection_full/<winner>/weights/best.pt`. Подробности и заметки по
банковским классам — в [`forte_object_detection_benchmark/README.md`](forte_object_detection_benchmark/README.md).

### Action-модель — `forte_action_benchmark/`

```bash
cd forte_action_benchmark
pip install -r requirements.txt            # torch/torchvision — отдельно под свою CUDA

# 1) подготовка и сплит по ВИДЕО (не по кадрам), 4 тестовых ролика
python train.py prepare --data /path/to/action_dataset --manifest runs/data --test-count 4 --seed 42
#   при дубликатах между сплитами Roboflow: добавь --resplit-by-hash

# 2) один baseline
python train.py train --manifest runs/data --model mobilenet_gru \
  --loss ce --aug light --epochs 20 --batch 4 --frames 16 --size 128 --out runs/baseline

# 3) полный грид (11 конфигов: 5 бэкбонов × лоссы × аугментации × freeze/fine-tune)
python sweep.py --config experiments.json --manifest runs/data --output runs/full_grid
#   ранжирование по VAL macro-F1; победитель: runs/full_grid/<winner>/best.pt

# 4) held-out test ОДИН раз после выбора модели
python train.py test --manifest runs/data \
  --checkpoint runs/full_grid/<winner>/best.pt --out runs/final_test
```

Бэкбоны: `mobilenet_gru`, `resnet18_gru`, `r3d18`, `mc3_18`, `r2plus1d18`. Быстрый health-check
без весов/видео: `python sanity_check.py`. Детали — в
[`forte_action_benchmark/README.md`](forte_action_benchmark/README.md).

> Важно: если сегмент покрывает >80% видео — это **слабая разметка** (метка на всё видео),
> модель даёт приблизительные окна, а не точную локализацию событий.

### CV-бейзлайны — `forte_cv_experiments/`

```bash
cd forte_cv_experiments
pip install -r requirements-core.txt       # без весов (E0/E1)
pip install -r requirements-vision.txt     # для pretrained (E2/E3)
python run.py prepare --zip videos.zip --labels open_labels.xlsx   # 4 test-видео
python run.py extract --features-dir runs/features_motion --encoder none --detector none \
  --window-sec 4 --stride-sec 2 --frames-per-window 8
```

Полное описание E0–E5 — в [`forte_cv_experiments/README.md`](forte_cv_experiments/README.md).

### VLM-baseline (Qwen) — корень репозитория

```bash
# положи videos.zip в корень; скрипт сам распакует нужный ролик в ./videos/
python run_fortebank_qwen.py --video X-1 --frames 16
python run_fortebank_qwen.py --all --frames 16
```
Результаты складываются в `qwen_results/`. Промпт и строгая JSON-схема — `fortebank_qwen_prompt.md`.

---

## Экспорт моделей под CPU

После тренировки конвертируй победные `best.pt` в ONNX/OpenVINO для CPU-инференса в сервисе:

```bash
cd forte_railway_github
pip install -r requirements-worker-cpu.txt    # + CPU torch/torchvision (см. шапку файла)

python -m worker.vision.export \
  --detector /path/to/detection/best.pt \
  --action   /path/to/action/best.pt \
  --out models --format onnx --imgsz 640       # или --format openvino / both
```

Детектор → `models/best.onnx` или `models/best_openvino_model/`; action → `models/action.onnx`
(+ `action.meta.json`) или `models/action.xml`. Бэкенд в рантайме выбирается по расширению
(`.onnx`→ONNX Runtime, `.xml`/`_openvino_model`→OpenVINO).

---

## Сборка и деплой сервиса

Полная инструкция — [`forte_railway_github/README.md`](forte_railway_github/README.md).
Кратко, вариант «всё в одном контейнере на CPU» (Railway):

1. Экспортируй модели в `forte_railway_github/models/` (см. выше) и закоммить их:
   ```bash
   git add -f models/best_openvino_model models/action.onnx models/action.meta.json
   git commit -m "Ship CPU models" && git push
   ```
2. Railway → Deploy from GitHub → собирается `Dockerfile` (ставит CPU torch + ultralytics +
   onnxruntime/openvino, копирует `worker/` и `models/`, запускает `start-allinone.sh`:
   uvicorn + worker в одном контейнере).
3. Переменные окружения (см. ниже), привяжи Volume на `/data`, сгенерируй домен.
4. Проверь:
   ```bash
   curl https://<домен>/health
   curl -X POST https://<домен>/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video.mov"
   curl https://<домен>/api/v1/jobs/<job_id>   -H "X-API-Key: <ключ>"
   ```
   Интерактивная Swagger-документация: `https://<домен>/docs`.

Локальный запуск контракта API (mock-инференс, без CUDA): `pytest forte_railway_github/tests`.

---

## Переменные окружения

| Переменная | Назначение |
|---|---|
| `FORTE_API_KEY` | **Единственный обязательный** ключ. Это `X-API-Key`, который отдаёшь жюри. ≥16 символов. |
| `FORTE_WORKER_KEY` | Необязательный. Нужен, только если worker — отдельный недоверенный сервис; иначе `/internal` переиспользует `FORTE_API_KEY`. |
| `FORTE_DETECTOR_WEIGHTS` | Путь к детектору: `/app/models/best_openvino_model` или `/app/models/best.onnx`. |
| `FORTE_ACTION_CHECKPOINT` | Путь к action-модели: `/app/models/action.onnx` или `action.xml`. |
| `FORTE_DETECTOR_IMGSZ` | Разрешение инференса детектора; должно совпадать с `--imgsz` при экспорте (напр. 640). |
| `FORTE_DEVICE` | `cpu` или номер GPU (в all-in-one выставляется автоматически). |
| `FORTE_STORAGE_DIR` | Каталог данных/очереди (Railway Volume, напр. `/data`). |
| `FORTE_CPU_THREADS` | (опц.) ограничение числа потоков ONNX Runtime / OpenVINO под план Railway. |

Сгенерировать ключ: `python -c "import secrets; print(secrets.token_urlsafe(24))"`.
```
