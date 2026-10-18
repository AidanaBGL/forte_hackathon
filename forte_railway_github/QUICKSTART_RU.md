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

**Внимание:** перед загрузкой реальных видео ForteBank в сторонний Railway проверь условия соревнования о передаче и хранении файлов у третьих лиц. Для тестовых роликов это может быть запрещено.
