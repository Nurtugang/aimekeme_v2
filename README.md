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
- **persons** — детекция людей + цвет одежды (верх/низ) по пачке кадров (YOLOv8n-pose,
  ultralytics). Бокс делится на торс/ноги по keypoints скелета (fallback — фиксированные
  доли высоты). Опциональный `query` фильтрует людей по названию цвета.

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
└── persons/                # model.py · detector.py · schemas.py · router.py
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

# 1) основной стек: torch (cu128) + FastAPI + fight/fire/counting. Индекс torch уже в req.txt.
#    Сюда же входит ultralytics (для counting `yolo_head`) — тяжёлый и под AGPL-3.0.
#    Если yolo_head не нужен (используете только `frcnn`) — можно убрать блок
#    "counting: yolo_head backend" из req.txt перед установкой.
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

### `POST /detect/persons`
Запрос — 1..N base64-JPEG кадров и необязательный `query`:
```json
{
  "frames": ["<base64_jpg>", "<base64_jpg>"],
  "query": { "top": "blue", "bottom": "black" }
}
```
Ответ — по одному результату на кадр, в том же порядке:
```json
{
  "results": [
    { "persons": [
        { "box": [0.31, 0.22, 0.44, 0.78], "confidence": 0.91,
          "top_color": "blue", "top_hsv": [210.0, 0.62, 0.48],
          "bottom_color": "black", "bottom_hsv": [0.0, 0.05, 0.12] }
      ],
      "count": 1 },
    { "persons": [], "count": 0 }
  ],
  "processing_ms": 31.4
}
```
`box` — доли кадра `[x1, y1, x2, y2]` (0..1), не пиксели. `top_color`/`bottom_color` —
ближайшее название из фиксированной палитры; `top_hsv`/`bottom_hsv` — сырой HSV
`[hue 0-360, saturation 0-1, value 0-1]`. Бокс человека делится на торс/ноги по
keypoints скелета (YOLOv8n-pose); если keypoints не распознались уверенно —
фиксированные доли высоты. Кожа (руки/ноги) исключается из оценки цвета перед
кластеризацией.
Если передан `query`, в ответе остаются только люди, у которых совпали заданные
поля (`top` и/или `bottom`) — остальные отбрасываются, `count` — по оставшимся.
Ошибки (HTTP 422): `Invalid base64 in frame 3`.

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
| `FIRE_MODEL`           | `yolo_dfire`  | модель огня: `siglip2` (классификатор) / `yolo_dfire` (боксы, AGPL) |
| `FIRE_THRESHOLD`       | `0.5`         | `fire`/`smoke` ниже порога уверенности ⇒ `normal` |
| `USE_TILING`           | `true`        | нарезка кадра 2x2 + кадр целиком для мелких очагов |
| `COUNT_MODEL`          | `frcnn`       | модель подсчёта: `frcnn` (person, BSD) / `yolo_head` (головы, AGPL) |
| `COUNT_SCORE_THRESH`   | `0.5`         | боксы ниже score/conf не идут в подсчёт         |
| `COUNT_HEAD_VARIANT`   | `medium`      | для `yolo_head`: `medium` (точнее) / `nano` (быстрее) |
| `FACE_MATCH_THRESHOLD` | `0.42`        | косинусная близость (ArcFace) >= порог ⇒ узнан  |
| `FACE_DET_THRESH`      | `0.5`         | порог детектора лиц (insightface)               |
| `KNOWN_FACES_DIR`      | `known_faces` | папка с эталонами лиц                           |
| `PERSONS_CONF_THRESH`  | `0.4`         | боксы людей ниже уверенности детектора не идут в ответ |

Число кадров (`expected_frames = 16`) задано в `app/config.py` — это требование модели.

## References
1. Feichtenhofer, C. (2020). X3D: Expanding Architectures for Efficient Video Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 203-213).
2. M. Cheng, K. Cai, and M. Li, "RWF-2000: An Open Large Scale Video Database for Violence Detection," in 2020 25th International Conference on Pattern Recognition (ICPR), 2021, pp. 4183-4190. doi: 10.1109/ICPR48806.2021.9412502.
3. N. Nguyen, "School Violence Detection: A Comparative Study of 3D CNN Architectures," Graduation thesis, University of Information Technology (UIT), VNU-HCM, 2026
4. Zhai, X., et al. (2023). Sigmoid Loss for Language Image Pre-Training (SigLIP). In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 11975-11986). (fire-классификатор `prithivMLmods/Fire-Detection-Siglip2` на базе SigLIP2, Apache-2.0)
5. Li, Y., et al. (2021). Benchmarking Detection Transfer Learning with Vision Transformers. arXiv:2111.11429. (torchvision `fasterrcnn_resnet50_fpn_v2`, веса COCO)
6. Peng, D., et al. (2018). Detecting Heads using Feature Refine Net and Cascaded Multi-scale Architecture. arXiv:1803.09256. (датасет SCUT-HEAD)
7. Jocher, G., et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics (counting `yolo_head` — веса `Abcfsa/YOLOv8_head_detector`, обучены на SCUT-HEAD; AGPL-3.0)
