"""Настройки приложения."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_version: str = "0.2.0"
    device: str = "auto"          # auto | cuda | cpu | cuda:0  (общий для всех моделей)

    # fight
    expected_frames: int = 16     # Модель для классификации драки X3D-M требует 16 кадров
    fight_threshold: float = 0.4

    # fire: выбор модели
    # siglip2 (классификатор кадра) | yolo_dfire (боксы, AGPL)
    fire_model: str = "siglip2"
    fire_threshold: float = 0.4   # fire/smoke ниже порога уверенности => normal
    use_tiling: bool = True       # Включить/выключить нарезку 2x2 для поиска мелких очагов

    # counting: выбор модели
    count_model: str = "yolo_head"        # frcnn (torchvision, класс person) | yolo_head (YOLOv8 SCUT-HEAD)
    count_score_thresh: float = 0.4   # порог score(frcnn)/conf(yolo_head) — боксы ниже в подсчёт не идут
    count_head_variant: str = "medium"  # только для yolo_head: medium (точнее) | nano (быстрее)

    # --- face (ArcFace / buffalo_l) ---
    known_faces_dir: str = "known_faces"
    face_match_threshold: float = 0.42
    face_det_thresh: float = 0.5
    face_det_size: int = 640
    face_max_upload_bytes: int = 5 * 1024 * 1024
    face_max_image_side: int = 1600

    # --- attention: вовлечённость на лекции ---
    # Направление «на доску» в углах ВЗГЛЯДА относительно камеры. Зависит от
    # того, где висит камера — замеряется scripts/attention_calibrate.py.
    attention_yaw_center: float = 0.0
    attention_pitch_center: float = 0.0
    attention_yaw_tolerance: float = 25.0    # допуск влево/вправо, градусы
    attention_pitch_tolerance: float = 20.0  # допуск вверх/вниз, градусы
    attention_threshold: float = 0.5         # score >= порога => смотрит (0.5 = граница допуска)
    gaze_eye_gain: float = 25.0              # градусов на единицу смещения зрачка

    # attention: детектор поз (скелеты). Для 4К берите x.
    pose_variant: str = "x"                  # n | s | m | l | x
    pose_conf: float = 0.35                  # порог уверенности на человека
    pose_nms_iou: float = 0.6                # NMS: чем ниже, тем агрессивнее режет дубли
    pose_imgsz: int = 1280                   # вход детектора поз (4К -> 1280/1600)
    pose_kpt_conf: float = 0.35              # ниже — точка скелета считается невидимой

    # attention: детектор предметов (телефон/книга/ноутбук) — надёжнее эвристик по позе.
    object_enabled: bool = True
    object_variant: str = "x"
    object_conf: float = 0.30
    object_imgsz: int = 1600                 # телефон мелкий, нужен вход больше
    object_hand_radius: float = 0.9          # радиус привязки к кисти, в ширинах плеч

    # attention: трекинг (один человек — один скелет и один стабильный id)
    track_iou_threshold: float = 0.3
    track_max_age_s: float = 3.0             # столько трек живёт без обновлений
    track_min_hits: int = 3                  # столько кадров до подтверждения трека
    max_cameras: int = 32                    # предел на число одновременных камер

    # attention: временное окно — вовлечённость по одному кадру не измерима
    engagement_window_s: float = 30.0
    engagement_min_span_s: float = 5.0       # пока меньше — warming_up
    engagement_min_samples: int = 5
    engagement_looking_ratio: float = 0.5    # доля окна со взглядом => engaged
    engagement_hold_target_s: float = 5.0    # непрерывный взгляд, дающий полный бонус
    engagement_look_weight: float = 0.7
    engagement_hold_weight: float = 0.3
    engagement_writing_floor: float = 0.6    # письмо — тоже вовлечённость
    engagement_phone_factor: float = 0.2     # штраф за телефон

    # attention: сон. PERCLOS — доля времени с закрытыми глазами за окно.
    sleep_eye_closure: float = 0.55          # blendshape eyeBlink выше => глаз закрыт
    sleep_perclos: float = 0.7
    sleep_min_closed_s: float = 3.0          # непрерывная серия, чтобы не считать моргания

    # attention: пороги поз (в ширинах плеч). ЭВРИСТИКИ — подстройте под аудиторию.
    posture_slump_drop: float = -0.15        # нос почти на линии плеч => голова упала
    posture_head_on_hands: float = 0.55      # кисть у самого лица => лежит на руках
    posture_head_down_deg: float = 18.0      # наклон головы, считающийся «вниз»
    posture_phone_min_lift: float = -0.85    # кисть поднята к груди/лицу
    posture_phone_to_face: float = 1.05
    posture_phone_lateral: float = 0.85
    posture_write_max_lift: float = -0.55    # кисть лежит низко (на парте)
    posture_write_min_to_face: float = 1.0
    posture_vote_share: float = 0.4          # доля голосов за активность в окне

    # attention: лицо
    face_min_confidence: float = 0.4

settings = Settings()
