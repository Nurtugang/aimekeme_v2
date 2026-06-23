# Intelligent Surveillance API

Модульный монолит интеллектуальной системы видеонаблюдения. Один FastAPI-процесс,
несколько алгоритмов. Сейчас реализованы:

- **fight** — детекция драк по клипу из 16 кадров (X3D-M, torch);
- **face** — распознавание лиц по одному кадру (ArcFace, модель `buffalo_l` через
  insightface/onnxruntime). Мини-база с несколькими эталонными фото на человека.

Все модели грузятся **один раз при старте** (lifespan) и складываются в реестр;

## Структура

```
app/
├── main.py                 # FastAPI: lifespan грузит все модели в app.state, /health
├── config.py               # Settings (env / .env)
├── fight/                  # model.py · detector.py · schemas.py · router.py
└── face/                   # model.py · detector.py · schemas.py · router.py
known_faces/                # мини-база лиц: <id>_<k>.jpg + faces.db (см. known_faces/README.md)
scripts/
├── build_payload.py        # нарезает видео на окна по 16 кадров для тестов fight
└── encode_image.py         # фото -> JSON для теста /detect/face
```

Модели грузятся один раз при старте (lifespan) и кладутся в `app.state.detectors`;
роутеры берут их через `request.app.state`.

## Установка

Нужно: Python 3.10 и GPU NVIDIA с CUDA 12 (Blackwell / RTX 50xx). Без GPU всё
работает на CPU (медленнее).

Ставить **двумя шагами** — основной стек и отдельно распознавание лиц:

```bash
python3.10 -m venv venv && source venv/bin/activate

# 1) основной стек: torch (cu128) + FastAPI + fight. Индекс torch уже прописан в req.txt.
pip install -r req.txt

# 2) распознавание лиц (ArcFace). Ставить ИМЕННО в этом порядке:
pip install onnxruntime-gpu==1.23.2
pip install --no-deps insightface==1.0.1
pip install onnx scipy scikit-image
```

Почему face-стек отдельно: пакет `insightface` тянет за собой **cpu**-`onnxruntime`,
который конфликтует с `onnxruntime-gpu` и ломает работу на видеокарте. Поэтому
GPU-вариант ставим первым, а `insightface` — без его зависимостей (`--no-deps`).
Эти пакеты намеренно НЕ в `req.txt`, чтобы `pip install -r req.txt` их не сломал.

GPU-нюанс: onnxruntime берёт CUDA-библиотеки из torch, поэтому face-модуль
импортирует torch до insightface (уже сделано в коде, `app/face/model.py`).

Веса моделей качаются при первом старте: X3D — в `~/.cache/huggingface/`,
ArcFace `buffalo_l` — в `~/.insightface/models/`.

## Запуск

```bash
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

Интерактивные доки: http://localhost:8000/docs

## Эндпоинты

### `POST /detect/fight`
Запрос — ровно 16 base64-JPEG кадров по порядку:
```json
{ "frames": ["<base64_jpg>", "...", "<base64_jpg>"] }
```
Ответ:
```json
{ "label": "fight", "confidence": 0.87, "processing_ms": 4.2 }
```
Ошибки (HTTP 422): `Expected 16 frames, got 10` · `Invalid base64 in frame 3`.

### `POST /detect/face`
Запрос — один base64-JPEG кадр:
```json
{ "frame": "<base64_jpg>" }
```
Ответ (для каждого лица — рамка, уверенность детектора, имя/ID из базы или `unknown`):
```json
{
  "faces": [
    { "box": [x1, y1, x2, y2], "det_confidence": 0.99,
      "identity": "nurtugan", "identity_id": 1, "similarity": 0.71 }
  ],
  "count": 1,
  "processing_ms": 18.0
}
```
`identity` — имя или `"unknown"`; `identity_id` — стабильный ID человека или `null`.
Ошибки (HTTP 422): `Invalid base64 image`.

### База лиц (enrollment)

Эталоны лиц хранятся в этом сервисе (`known_faces/`). Управление — через API.
Несколько фото на человека повышают точность (заливайте разные ракурсы/условия).

- `POST   /faces` — записать человека. `multipart/form-data`: `name` + `images`
  (1..N файлов, на каждом одно чёткое лицо) → `{ id, name, created_at, photos }`.
- `POST   /faces/{id}/images` — догрузить ещё фото человеку.
- `GET    /faces` — список `[{ id, name, created_at, photos }]`.
- `GET    /faces/{id}/image` — первичное фото человека (image/jpeg).
- `DELETE /faces/{id}` — удалить человека.

Ошибки: нет годных лиц → `422`; имя занято → `409`; нет человека → `404`;
файл больше лимита → `413`. Подробнее — `known_faces/README.md`.

### `GET /health`
Готовность каждой модели и устройство:
```json
{ "status": "ok", "models": [{"name": "fight", "ready": true, "device": "cuda"}, ...], "version": "0.2.0" }
```

## Быстрая проверка (fight)

`build_payload.py` нарезает видео на окна по 16 кадров. Рядом с видео появляется
папка `<имя_видео>/`, в каждой подпапке `window_*` лежат `payload.json` (тело
запроса) и `clip.mp4`.

```bash
python scripts/build_payload.py test/file_000001.avi
curl -X POST http://localhost:8000/detect/fight \
     -H "Content-Type: application/json" \
     -d @test/file_000001/window_0000_f000000-000015/payload.json
```

## Конфигурация

Переопределяется через переменные окружения или `.env` (см. `.env.example`):

| Переменная             | По умолч.     | Описание                                       |
|------------------------|---------------|------------------------------------------------|
| `DEVICE`               | `auto`        | `auto` / `cuda` / `cpu` / `cuda:0` ...          |
| `FIGHT_THRESHOLD`      | `0.5`         | `P(fight) >= threshold` ⇒ метка `fight`        |
| `FACE_MATCH_THRESHOLD` | `0.42`        | косинусная близость (ArcFace) >= порог ⇒ узнан  |
| `FACE_DET_THRESH`      | `0.5`         | порог детектора лиц (insightface)               |
| `KNOWN_FACES_DIR`      | `known_faces` | папка с эталонами лиц                           |

Число кадров (`expected_frames = 16`) задано в `app/config.py` — это требование модели.

## References
1. Feichtenhofer, C. (2020). X3D: Expanding Architectures for Efficient Video Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 203-213).
2. M. Cheng, K. Cai, and M. Li, "RWF-2000: An Open Large Scale Video Database for Violence Detection," in 2020 25th International Conference on Pattern Recognition (ICPR), 2021, pp. 4183-4190. doi: 10.1109/ICPR48806.2021.9412502.
3. N. Nguyen, "School Violence Detection: A Comparative Study of 3D CNN Architectures," Graduation thesis, University of Information Technology (UIT), VNU-HCM, 2026
