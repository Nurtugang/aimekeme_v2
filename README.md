# Intelligent Surveillance API

Модульный монолит интеллектуальной системы видеонаблюдения. Один FastAPI-процесс,
несколько алгоритмов. Сейчас реализованы:

- **fight** — детекция драк по клипу из 16 кадров (X3D-M, torch);
- **face** — распознавание лиц по одному кадру (ArcFace, модель `buffalo_l` через
  insightface/onnxruntime). Мини-база с несколькими эталонными фото на человека.
- **fire** — детекция огня/дыма по одному кадру. Три сменные модели (выбор через
  `FIRE_MODEL`): `siglip2` — SigLIP2-классификатор кадра целиком (transformers);
  `yolo_dfire` — YOLOv8n, дообученный на D-Fire, даёт боксы вокруг очага
  (ultralytics, AGPL; по умолчанию). Классы `fire`/`smoke`/`normal`;
- **counting** — подсчёт людей по одному кадру. Две сменные модели (выбор через
  `COUNT_MODEL`): `frcnn` — torchvision Faster R-CNN, класс person (BSD, по умолчанию);
  `yolo_head` — YOLOv8-детектор голов на SCUT-HEAD, точнее в толпе (ultralytics, AGPL).
  `count` = число боксов выше порога.
- **attention** — вовлечённость на лекции. Единственный модуль, который смотрит
  не на кадр, а на человека во времени: по каждому находит скелет (YOLO11-pose),
  направление взгляда (MediaPipe FaceLandmarker), предметы в руках (YOLO11/COCO)
  и накапливает окно в 30 с. Отвечает, кто вовлечён, кто спит, кто пишет, а кто
  сидит в телефоне. **Stateful** — см. раздел про эндпоинт.

Все модели грузятся **один раз при старте** (lifespan) и складываются в реестр;

## Структура

```
app/
├── main.py                 # FastAPI: lifespan грузит все модели в app.state, /health
├── config.py               # Settings (env / .env)
├── fight/                  # model.py · detector.py · schemas.py · router.py
├── face/                   # model.py · detector.py · schemas.py · router.py
├── fire/                   # model_siglip2.py · model_yolo_dfire.py · detector.py · schemas.py · router.py
├── counting/               # model_frcnn.py · model_yolo_head.py · detector.py · schemas.py · router.py
└── attention/              # вовлечённость: 5 моделей + время
    ├── model_pose.py       # YOLO11-pose: скелеты COCO-17
    ├── model_face.py       # MediaPipe FaceLandmarker: взгляд, глаза, углы головы
    ├── model_objects.py    # YOLO11 COCO: телефон / книга / ноутбук
    ├── geometry.py         # чистая математика: углы, позы, скор (тестируется без GPU)
    ├── tracking.py         # трекинг по IoU: один человек — один id
    ├── temporal.py         # окно по треку: длительность взгляда, PERCLOS, вовлечённость
    └── detector.py · schemas.py · router.py
known_faces/                # мини-база лиц: <id>_<k>.jpg + faces.db (см. known_faces/README.md)
tests/                      # pytest: геометрия, трекинг, окна, сборка детектора (без GPU)
scripts/
├── fight_build_payload.py  # нарезает видео на окна по 16 кадров для тестов fight
├── fight_infer.py          # офлайн-инференс драк
├── fire_infer.py           # офлайн-инференс огня/дыма
├── counting_infer.py       # офлайн-инференс подсчёта людей
├── attention_calibrate.py  # ЗАМЕР углов под свою камеру -> готовые строки для .env
├── attention_infer.py      # офлайн-инференс вовлечённости (камера/видео/фото)
├── attention_request.py    # проверка живого POST /detect/attention по HTTP
├── extract_frames.py       # нарезка видео на кадры
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

# 1) основной стек: torch (cu128) + FastAPI + fight/fire/counting. Индекс torch уже в req.txt.
#    Сюда же входит ultralytics (для counting `yolo_head`) — тяжёлый и под AGPL-3.0.
#    Если yolo_head не нужен (используете только `frcnn`) — можно убрать блок
#    "counting: yolo_head backend" из req.txt перед установкой.
pip install -r req.txt

# 2) распознавание лиц (ArcFace) — нужен и модулю face, и модулю attention
#    (attention берёт из того же пака buffalo_l детектор лиц + 3D-landmarks).
#    Ставить ИМЕННО в этом порядке:
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

Веса моделей качаются при первом старте: X3D и SigLIP2 (fire) — в
`~/.cache/huggingface/`, ArcFace `buffalo_l` — в `~/.insightface/models/`,
Faster R-CNN (counting `frcnn`) — в `~/.cache/torch/hub/checkpoints/`,
YOLOv8 head (counting `yolo_head`) — в `~/.cache/aimekeme/head_detector/`.

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

### `POST /detect/fire`
Запрос — один base64-JPEG кадр:
```json
{ "frame": "<base64_jpg>" }
```
Ответ:
```json
{ "label": "fire", "confidence": 0.91, "processing_ms": 20.0 }
```
`label` ∈ `fire` / `smoke` / `normal`. Порог `FIRE_THRESHOLD`: если топ-класс —
`fire`/`smoke`, но уверенность ниже порога, отдаётся `normal` (режем ложные
срабатывания на лампы/экраны). Это фильтр на **один** кадр; подтверждение по N
кадрам подряд — на стороне брокера (контракт stateless). CV-детекция огня
дополняет дымовые датчики, а не заменяет их.
Ошибки (HTTP 422): `Invalid base64 image`.

### `POST /detect/counting`
Запрос — один base64-JPEG кадр:
```json
{ "frame": "<base64_jpg>" }
```
Ответ:
```json
{ "label": "person", "count": 12, "confidence": 0.94, "processing_ms": 74.0 }
```
`count` — число обнаруженных людей (боксы выше `COUNT_SCORE_THRESH`);
`confidence` — средний score детектора по ним (0.0, если никого). Ответ одинаков
для обеих моделей (`COUNT_MODEL` = `frcnn` | `yolo_head`) — контракт не меняется.
Ошибки (HTTP 422): `Invalid base64 image`.

### `POST /detect/attention`
Оценка вовлечённости на лекции: взгляд, его длительность, сон, письмо, телефон.

Запрос — кадр и **идентификатор камеры**:
```json
{ "frame": "<base64_jpg>", "camera_id": "aud_301" }
```
Ответ:
```json
{
  "people": [
    { "track_id": 7, "box": [x1, y1, x2, y2],
      "keypoints": [[x, y, conf], "... 17 точек COCO ..."],
      "state": "phone", "engagement": 0.14,
      "gaze_yaw": -31.2, "gaze_pitch": 24.8,
      "head_yaw": -22.0, "head_pitch": 19.5, "eye_closure": 0.08,
      "attention_score": 0.21, "looking_now": false,
      "gaze_hold_s": 0.0, "looking_ratio": 0.12, "perclos": 0.04,
      "eyes_closed_s": 0.0,
      "activity": "phone", "activity_share": 0.87,
      "held_objects": ["cell phone"],
      "window_s": 30.0, "warming_up": false }
  ],
  "count": 24, "engaged_count": 17,
  "engagement_rate": 0.7083, "mean_engagement": 0.6612,
  "states": { "engaged": 15, "writing": 2, "phone": 4, "distracted": 2, "sleeping": 1 },
  "processing_ms": 78.0
}
```

**Состояния** (`state`) в порядке приоритета: `sleeping` → `phone` → `writing` →
`engaged` → `distracted` → `unknown` (лица не видно и поза ни о чём не говорит).
Письмо считается вовлечённостью: человек работает, просто не смотрит на доску.

**Как определяется что:**

| Признак | Откуда |
|---|---|
| направление взгляда | углы головы (матрица MediaPipe) + смещение зрачков (`eyeLook*`) |
| длительность взгляда | `gaze_hold_s` — текущая непрерывная серия, не обрезается окном |
| сон | PERCLOS ≥ `SLEEP_PERCLOS` и непрерывная серия закрытых глаз ≥ `SLEEP_MIN_CLOSED_S`; **или** голова на парте, когда лица не видно вовсе |
| телефон | `cell phone` детектором COCO рядом с кистью; запасной вариант — поза |
| письмо | `book`/`laptop` у кисти или кисть лежит низко на парте ниже локтя |

**`camera_id` обязателен по смыслу.** Треки и временные окна ведутся отдельно на
каждую камеру. Если слать все потоки с одним id, люди с разных камер начнут
сопоставляться друг с другом; если слать каждый кадр с новым id, окно никогда
не наберётся и все останутся `warming_up`.

**Скелет не дублируется.** Первичный ключ — трек, а не детекция: NMS внутри YOLO
оставляет по одной рамке на человека, трекер сводит её к постоянному `track_id`,
и лицо считается на кропе головы ЭТОГО трека. Двум скелетам на одном человеке
взяться неоткуда — это свойство архитектуры, а не постфильтра, и оно закреплено
тестами `tests/test_detector.py::TestNoDuplicates`.

**Модуль stateful** — единственный в сервисе. Это осознанный отход от правила 3
`docs/CONVENTIONS.md`: длительность взгляда по одному кадру не вычислима.
Состояние ограничено — окно `ENGAGEMENT_WINDOW_S`, TTL трека `TRACK_MAX_AGE_S`,
предел `MAX_CAMERAS`; мгновенные значения (`looking_now`, `attention_score`,
`head_*`) в ответе есть, поэтому брокер при желании считает своё поверх них.

Первые секунды после появления человека `warming_up: true` — окно ещё набирается,
выводам верить рано.

Ошибки (HTTP 422): `Invalid base64 image`.

> **Калибровка обязательна.** Углы «на доску» зависят от того, где висит камера.
> `python scripts/attention_calibrate.py` замеряет их и печатает готовые строки
> для `.env`, заодно проверяя знаки. Без калибровки при камере сбоку вся
> аудитория будет `distracted`.

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

## Быстрая проверка (attention)

```bash
# 1. КАЛИБРОВКА. Смотрите на доску -> `c`; на левый край аудитории -> `l`;
#    на правый -> `r`; выход -> `q`. Скрипт напечатает строки для .env
#    и проверит, что знак угла не перевёрнут.
python scripts/attention_calibrate.py

# 2. ОФЛАЙН той же моделью, что в API. Камера — живое окно со скелетами.
python scripts/attention_infer.py 0 camera
python scripts/attention_infer.py lecture.mp4 video --pose-variant x --pose-imgsz 1600
python scripts/attention_infer.py row3.jpg photo --no-objects

# 3. ЖИВОЙ ЭНДПОИНТ (сервис поднят). Серия кадров на один camera_id — так же,
#    как это делал бы брокер.
python scripts/attention_request.py 0 --frames 60 --interval 0.5

# 4. ТЕСТЫ логики — без GPU и без весов, доли секунды.
python -m pytest tests/ -q
```

Оценка вовлечённости требует ВРЕМЕНИ: на одиночном фото все будут `warming_up`,
а состояния — предварительными. Фото годится проверить скелеты и углы, не выводы.

Если после калибровки аудитория всё равно `distracted` — поднимите
`ATTENTION_YAW_TOLERANCE`: в широком зале крайние ряды смотрят на доску под
заметным углом.

## Конфигурация

Переопределяется через переменные окружения или `.env` (см. `.env.example`):

| Переменная             | По умолч.     | Описание                                       |
|------------------------|---------------|------------------------------------------------|
| `DEVICE`               | `auto`        | `auto` / `cuda` / `cpu` / `cuda:0` ...          |
| `FIGHT_THRESHOLD`      | `0.5`         | `P(fight) >= threshold` ⇒ метка `fight`        |
| `FIRE_MODEL`           | `yolo_dfire`  | модель огня: `siglip2` (классификатор) / `yolo_dfire` (боксы, AGPL) |
| `FIRE_THRESHOLD`       | `0.5`         | `fire`/`smoke` ниже порога уверенности ⇒ `normal` |
| `USE_TILING`           | `true`        | нарезка кадра 2x2 + кадр целиком для мелких очагов |
| `COUNT_MODEL`          | `frcnn`       | модель подсчёта: `frcnn` (person, BSD) / `yolo_head` (головы, AGPL) |
| `COUNT_SCORE_THRESH`   | `0.5`         | боксы ниже score/conf не идут в подсчёт         |
| `COUNT_HEAD_VARIANT`   | `medium`      | для `yolo_head`: `medium` (точнее) / `nano` (быстрее) |
| `FACE_MATCH_THRESHOLD` | `0.42`        | косинусная близость (ArcFace) >= порог ⇒ узнан  |
| `FACE_DET_THRESH`      | `0.5`         | порог детектора лиц (insightface)               |
| `KNOWN_FACES_DIR`      | `known_faces` | папка с эталонами лиц                           |
| `ATTENTION_YAW_CENTER` / `_PITCH_CENTER` | `0.0` | направление «на доску» (калибровка) |
| `ATTENTION_YAW_TOLERANCE` / `_PITCH_TOLERANCE` | `25` / `20` | допуск по взгляду, градусы |
| `ATTENTION_THRESHOLD`  | `0.5`  | `score >= threshold` ⇒ смотрит (0.5 = граница допуска) |
| `GAZE_EYE_GAIN`        | `25.0` | вклад смещения зрачков во взгляд, градусов      |
| `POSE_VARIANT`         | `x`    | размер YOLO11-pose: `n`…`x`                     |
| `POSE_IMGSZ`           | `1280` | вход детектора поз (для 4К осмысленно `1600`)   |
| `OBJECT_ENABLED`       | `true` | искать телефон/книгу/ноутбук отдельным детектором |
| `OBJECT_IMGSZ`         | `1600` | вход детектора предметов — телефон мелкий        |
| `OBJECT_HAND_RADIUS`   | `0.9`  | радиус привязки предмета к кисти, в ширинах плеч |
| `TRACK_IOU_THRESHOLD`  | `0.3`  | порог сопоставления треков                      |
| `TRACK_MAX_AGE_S`      | `3.0`  | сколько трек живёт без обновлений, с            |
| `MAX_CAMERAS`          | `32`   | предел одновременных камер (память состояния)   |
| `ENGAGEMENT_WINDOW_S`  | `30.0` | окно, за которое считается вовлечённость, с     |
| `ENGAGEMENT_LOOKING_RATIO` | `0.5` | доля окна со взглядом ⇒ `engaged`            |
| `ENGAGEMENT_HOLD_TARGET_S` | `5.0` | непрерывный взгляд, дающий полный бонус      |
| `ENGAGEMENT_WRITING_FLOOR` | `0.6` | письмо — тоже вовлечённость, не ниже этого   |
| `ENGAGEMENT_PHONE_FACTOR`  | `0.2` | множитель-штраф за телефон                   |
| `SLEEP_EYE_CLOSURE`    | `0.55` | `eyeBlink` выше ⇒ глаз считается закрытым       |
| `SLEEP_PERCLOS`        | `0.7`  | доля закрытых глаз за окно ⇒ сон                |
| `SLEEP_MIN_CLOSED_S`   | `3.0`  | непрерывная серия, чтобы не считать моргания    |
| `POSTURE_*`            | —      | пороги поз в ширинах плеч, **эвристики** (см. `.env.example`) |

Число кадров (`expected_frames = 16`) задано в `app/config.py` — это требование модели.

## References
1. Feichtenhofer, C. (2020). X3D: Expanding Architectures for Efficient Video Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 203-213).
2. M. Cheng, K. Cai, and M. Li, "RWF-2000: An Open Large Scale Video Database for Violence Detection," in 2020 25th International Conference on Pattern Recognition (ICPR), 2021, pp. 4183-4190. doi: 10.1109/ICPR48806.2021.9412502.
3. N. Nguyen, "School Violence Detection: A Comparative Study of 3D CNN Architectures," Graduation thesis, University of Information Technology (UIT), VNU-HCM, 2026
4. Zhai, X., et al. (2023). Sigmoid Loss for Language Image Pre-Training (SigLIP). In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 11975-11986). (fire-классификатор `prithivMLmods/Fire-Detection-Siglip2` на базе SigLIP2, Apache-2.0)
5. Li, Y., et al. (2021). Benchmarking Detection Transfer Learning with Vision Transformers. arXiv:2111.11429. (torchvision `fasterrcnn_resnet50_fpn_v2`, веса COCO)
6. Peng, D., et al. (2018). Detecting Heads using Feature Refine Net and Cascaded Multi-scale Architecture. arXiv:1803.09256. (датасет SCUT-HEAD)
7. Jocher, G., et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics (counting `yolo_head` — веса `Abcfsa/YOLOv8_head_detector`, обучены на SCUT-HEAD; AGPL-3.0)
