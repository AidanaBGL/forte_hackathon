# ForteBank — локальный CV + action baseline (v1)

Готовый **экспериментальный** пайплайн на `videos.zip` + `open_labels.xlsx` для поиска подозрительных событий в кассовых видео. Не является готовой сертифицированной системой контроля кассовой дисциплины. Все инференсы локальные после первоначальной загрузки весов. Не используем распознавание лиц.

## Что уже есть в коде

| Эксперимент | Признаки | Что обучается | Можно прямо сейчас? |
|---|---|---|---|
| **E0 motion → video classifier** | optical flow, frame differences, HSV, edges | LogisticRegression, классификация всего видео | Да, по текущему Excel |
| **E1 motion → point-supervised temporal** | те же признаки, 4-секундные окна | LogisticRegression на положительных окнах вокруг GT-секунды и отрицательных видео | Да, но это слабая локализация, не класс нарушения |
| **E2 ResNet18 + YOLO person + temporal** | предобученные эмбеддинги кадров, motion, YOLO person | тот же temporal classifier; ResNet/YOLO пока frozen | Да, нужен GPU для скорости, веса загружаются при первом запуске |
| **E3 R3D-18 + YOLO-World + temporal** | Kinetics-pretrained video-эмбеддинги, open-vocabulary detections | temporal classifier, video backbone frozen | Да, но open-vocab деньги/сейф могут не детектироваться точно |
| **E4 action recognition (supervised)** | любые признаки E2/E3 и размеченные интервалы | классификатор банковских действий (cash_counting, safe_opening...) | После ручной разметки `action_segments.csv` |
| **E5 bank objects detection (supervised)** | изображения и bounding boxes | YOLO detector `banknotes`, `safe_door_open`, `safe_door_closed` и т. п. | После CVAT/Label Studio разметки объектов |

**Важно:** COCO-YOLO НЕ умеет по умолчанию детектировать банкноты или сейфы как отдельные классы. YOLO-World с текстовыми подсказками — zero-shot **гипотеза**, не готовая банковская money-модель. MediaPipe Hand Landmarker распознаёт landmarks, а не семантическое действие `count_cash`.

## 0. Файлы

Скачай проект и распакуй. В папку проекта положи:

```
forte_cv_experiments/
    run.py
    src/
    tools/
    configs/
    videos.zip            # из чата
    open_labels.xlsx      # из чата
```

Python 3.10+, желательно отдельное окружение. Проверенный офлайн smoke test: Python 3.11, OpenCV 4.13.0; GPU-ветки нужно испытать в твоём CUDA окружении.

Установка **без моделей**:

```bash
pip install -r requirements-core.txt
```

Для pretrained моделей:

```bash
pip install -r requirements-vision.txt
```

С учётом твоего PyTorch/CUDA окружения **не переустанавливай torch просто ради этого проекта** — если он установлен, ставь необходимые пакеты выборочно.

## 1. Создать split — ровно четыре test видео

```bash
python run.py prepare --zip videos.zip --labels open_labels.xlsx
```

Скрипт:

- нормализует кириллическую `Х-` и латинскую `X-` в названиях;
- читает Excel с бинарным GT и секундами;
- извлекает `.mov` в `data/videos/`;
- откладывает **X-15 (violation), X-16 (normal), X-22 (violation), X-23 (normal)** в test (их метки не участвуют в обучении);
- выявляет точные копии **X-4=X-47** и **X-5=X-48** по SHA-256; дубликаты не дублируют train;
- печатает конфликт таймкодов **X-4: 57 сек, X-47: 68 сек** (не пытаемся угадывать правильный);
- сохраняет `data/manifest.csv`.

**Эти четыре test видео нельзя использовать для настройки ROI, выбора модели, подбора порогов и аннотации обучающего датасета.** Если нужно честно сравнить 10 моделей после многократного просмотра метрик на этих четырёх роликах, понадобится другой/новый test: любой выбор модели по test приводит к leakage. Можно также до финального обучения переставить четыре holdout ID параметром `--test-ids` (две метки каждого класса), но не после просмотра метрик.

Все метрики по четырём видео имеют огромную дисперсию. Основные решения принимай по cross-validation на train, а test используй **один раз** для финального сравнения заранее выбранных кандидатов.

## 2. E0 и E1: работающие baseline без pretrained весов

Сначала сохраним окна и motion features:

```bash
python run.py extract \
  --features-dir runs/features_motion \
  --encoder none --detector none \
  --window-sec 4 --stride-sec 2 --frames-per-window 8
```

Для быстрой проверки синтаксиса на 2 роликах: добавь `--limit 2 --max-windows 3` и **используй отдельную папку features** — не обучай на урезанных фичах.

Обучение + 4 holdout predictions:

```bash
python run.py train --features-dir runs/features_motion --method video --out runs/e0_video
python run.py train --features-dir runs/features_motion --method temporal --out runs/e1_temporal
```

Что создаётся в каждом `runs/e*`:

- `metrics.json`: train out-of-fold + test summary;
- `test_predictions.csv`: все четыре held-out примера;
- `model.joblib`: sklearn classifier и TRAIN-only threshold;
- при `temporal`: `X-15_timeline.json`, ... и `evidence/X-15_candidate.jpg`, ...;
- при `video`: `candidate_time_sec` пустой по определению (video-level model не локализует).

**Не выдавай `candidate_time_sec` из E1 за точную секунду нарушения.** Это центр 4-секундного окна, выбранного weak-supervised классификатором.

## 3. E2: COCO человек + ResNet18 + temporal

```bash
python run.py extract \
  --features-dir runs/features_resnet_person \
  --encoder resnet18 --detector coco \
  --window-sec 4 --stride-sec 2 --frames-per-window 8 --max-side 960

python run.py train --features-dir runs/features_resnet_person \
  --method temporal --out runs/e2_resnet_person
```

COCO-person detector уже обучен и распознаёт человека, но **не отличает кассира от клиента по роли**. Роль можно вывести из нахождения в банковской кассовой зоне, отслеживания траектории и сценария, если камера и ракурс позволяют.

Для визуализации trajectories отдельных людей:

```bash
python tools/track_people.py data/videos/X-1.mov --out runs/person_tracks_X1
```

Track IDs — временные идентификаторы траекторий, без идентификации личности.

## 4. E3: Video backbone и open-vocabulary объекты

```bash
python run.py extract \
  --features-dir runs/features_r3d_world \
  --encoder r3d18 --detector world \
  --window-sec 4 --stride-sec 2 --frames-per-window 16 --max-side 960

python run.py train --features-dir runs/features_r3d_world \
  --method temporal --out runs/e3_r3d_world
```

`r3d18` = pretrained Kinetics video-ResNet18; YOLO-World загружает `yolov8s-world.pt` и запрашивает классы `person`, `banknotes`, `cash bundle`, `cash bag`, `safe door`, `laptop computer`, `money counting machine`, `identity document`. Для незнакомых банковских ракурсов точность может быть низкой. Детекторные классы — подсказки, не проверенная разметка.

Быстро осмотреть результаты YOLO-World:

```bash
python tools/detector_preview.py data/videos/X-1.mov --seconds 7 12 20 22
```

Если обнаружение мелких предметов слабое, сделай эксперимент с ROI кассового стола, например `--roi 0.1,0.1,0.95,0.95`. Но ROI выбирай **по train**, а не подгоняй под test. Для разных ракурсов координаты могут отличаться.

## 5. Как обучить модели на своём датасете — после разметки

### A. Action Recognition

Нужны реальные классы действий и временные границы. `open_labels.xlsx` этого не содержит.

Вручную размечай **только train** и заполни `configs/action_segments.csv`:

```csv
video_id,start_sec,end_sec,action
X-1,5.0,7.5,document_handling
X-1,9.0,11.5,computer_interaction
```

Эти две строки — **синтаксический пример, а не установленная истина о X-1**. Удали/исправь после просмотра. Рекомендуемые реальные классы по мере подтверждения: `cash_counting_manual`, `cash_counting_machine`, `cash_handover`, `cash_received`, `cash_retrieval`, `cash_storage`, `safe_opening`, `safe_closing`, `bag_opening`, `document_handling`, `computer_interaction`, `cashier_leaving`, `idle`.

```bash
python run.py train-actions \
  --features-dir runs/features_r3d_world \
  --segments configs/action_segments.csv --out runs/action_model.joblib

python run.py predict-actions \
  --features-dir runs/features_r3d_world \
  --model runs/action_model.joblib --out runs/actions_test --split test
```

Это linear action classifier поверх **frozen video embeddings** — разумная первая проба при 36 уникальных train. Для fine-tuning R3D-18 end-to-end потребуются больше action-labelled clips и контроль переобучения.

После action classification можно создать State Machine по процедурам НБ РК. **В этом v1 автоматическое юридическое заключение не реализовано**, потому что отдельные предположения о порядке закрытия сейфа пока не подтверждены нормативом.

### B. Object Detection (деньги, дверца сейфа, пакеты)

Выгрузи изображения **только из train**:

```bash
python tools/export_annotation_frames.py --per-video 12
```

Разметь их в CVAT/Label Studio bounding boxes для банкнот, пакета, открытой/закрытой дверцы, счётной машины и документов. Обязательно добавь много пустых/трудных кадров. При таком количестве данных начни с 200–500 **разнообразных** изображений, разделённых по исходным видео, не по кадрам.

Структура YOLO-датасета:

```
datasets/objects/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
```

`configs/objects.yaml` содержит список классов. Обучение:

```bash
python tools/train_money_detector.py \
 --data configs/objects.yaml --weights yolo11n.pt --epochs 40 --imgsz 640
```

Подключение обученных весов к E2:

```bash
python run.py extract \
 --features-dir runs/features_custom \
 --encoder resnet18 --detector custom \
 --custom-weights runs/bank_objects/weights/best.pt
```

## 6. Опциональный hand tracking

Для Mediapipe Tasks нужен отдельный model asset `hand_landmarker.task` (нужно один раз скачать из официального репозитория Google AI Edge). Затем:

```bash
python run.py extract \
  --features-dir runs/features_hands \
  --encoder resnet18 --detector coco \
  --hands mediapipe --hand-model /path/to/hand_landmarker.task
```

MediaPipe отслеживает landmark-положения **только если кисти различимы**. На потолочной камере и малом разрешении часто не видит руки. Даже работающий pose hand detector **не отличает пересчёт денег от работы с документами** без классификатора временных движений.

## 7. Сравнение и время

```bash
python tools/compare.py runs/e0_video runs/e1_temporal runs/e2_resnet_person runs/e3_r3d_world
```

Оценивай recall нарушений, false positive rate на корректных видео, точность timecode только у temporal модели, inference latency из `extract` (время **на видео**). Замеряй `<=300 сек` с резервом 20–30 секунд. На коротком train dataset нельзя рассчитывать на устойчивое значение метрик по четырём примерам; пригодится групповой cross-validation по сценариям.

## 8. API жюри (только для технического smoke test)

После E1/E2/E3 обученного temporal baseline:

```bash
pip install fastapi uvicorn python-multipart
export FORTE_API_KEY=your_long_random_key
export FORTE_MODEL=runs/e2_resnet_person/model.joblib
uvicorn api:app --host 0.0.0.0 --port 8080 --workers 1
```

Роуты `POST /api/v1/analyze`, `GET /api/v1/jobs/{job_id}`; заголовок `X-API-Key`; ответ с `violation` / `no_violation`, кандидатным таймкодом, одним JPEG base64. **Это пока лишь технический контракт API**: тип нарушения, правовое обоснование, обнаружение всех событий и надёжность моделью v1 не гарантируются. Не отправляй этот baseline как финальное юридически обоснованное решение жюри.

## Лицензии

Предобученная библиотека Ultralytics имеет условия AGPL-3.0 / Enterprise; проверь разрешённость коммерческого применения и требования хакатона. Лицензии конкретных весов, datasets и библиотек проверяются отдельно. Для production можно заменить Ultralytics на допустимую модель с подходящей лицензией.

## Структура

```
run.py                      # CLI prepare/extract/train/action training
src/dataset.py              # XLSX + ZIP + SHA duplicates + exact test split
src/features.py             # motion + pretrained visual embeddings + detectors + hands
src/experiments.py          # video classifier, point-supervised temporal, eval
src/actions.py              # optional labelled action head
api.py                      # local FastAPI technical stub
configs/objects.yaml        # YOLO object class list
configs/action_segments.csv # template for human action labels
tools/export_annotation_frames.py
tools/track_people.py
tools/detector_preview.py
tools/train_money_detector.py
tools/compare.py
```
