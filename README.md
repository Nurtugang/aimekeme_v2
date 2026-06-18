# Violence Detection API

Сервис детекции насилия — первый кирпич интеллектуальной системы видеонаблюдения.
Один эндпоинт классифицирует клип из 16 кадров как `fight` или `normal` (модель X3D-M).

## Структура

```
app/
├── main.py       # FastAPI: маршруты, загрузка модели один раз при старте (lifespan)
├── detector.py   # ViolenceDetector: decode → preprocess → инференс на GPU
├── model.py      # X3D-M: load_model + preprocess_frames (источник правды)
├── schemas.py    # Pydantic-модели запроса/ответа
└── config.py     # Settings (переопределяются через env / .env)
scripts/
└── build_payload.py   # нарезает видео на окна по 16 кадров для тестов
```

HTTP-слой тонкий; вся ML-логика в `detector.py` и `model.py`. Модель грузится
**один раз при старте**, а не на каждый запрос.

## Установка и запуск

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r req.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Интерактивные доки: http://localhost:8000/docs

## Эндпоинты

### `POST /detect_violence`

Запрос — ровно 16 base64-JPEG кадров по порядку:
```json
{ "frames": ["<base64_jpg>", "...", "<base64_jpg>"] }
```
Ответ:
```json
{ "label": "fight", "confidence": 0.87, "processing_ms": 4.2 }
```
Ошибки (HTTP 422):
```json
{ "detail": "Expected 16 frames, got 10" }
{ "detail": "Invalid base64 in frame 3" }
```

### `GET /health`
Готовность сервиса: загружена ли модель и на каком устройстве.

## Быстрая проверка

`build_payload.py` нарезает видео на окна по 16 кадров. Рядом с видео появляется
папка `<имя_видео>/`, в каждой подпапке `window_*` лежат `payload.json` (тело
запроса) и `clip.mp4` (эти 16 кадров как видео).

```bash
python scripts/build_payload.py test/file_000001.avi
curl -X POST http://localhost:8000/detect_violence \
     -H "Content-Type: application/json" \
     -d @test/file_000001/window_0000_f000000-000015/payload.json
```

## Конфигурация

Переопределяется через переменные окружения или `.env` (см. `.env.example`):

| Переменная        | По умолч. | Описание                                 |
|-------------------|-----------|------------------------------------------|
| `FIGHT_THRESHOLD` | `0.5`     | `P(fight) >= threshold` ⇒ метка `fight`  |
| `DEVICE`          | `auto`    | `auto` / `cuda` / `cpu` / `cuda:0` ...    |

Число кадров (`expected_frames = 16`) задано в `app/config.py` — это требование модели.

## References
1. Feichtenhofer, C. (2020). X3D: Expanding Architectures for Efficient Video Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 203-213).
2. M. Cheng, K. Cai, and M. Li, "RWF-2000: An Open Large Scale Video Database for Violence Detection," in 2020 25th International Conference on Pattern Recognition (ICPR), 2021, pp. 4183-4190. doi: 10.1109/ICPR48806.2021.9412502.
3. N. Nguyen, "School Violence Detection: A Comparative Study of 3D CNN Architectures," Graduation thesis, University of Information Technology (UIT), VNU-HCM, 2026
