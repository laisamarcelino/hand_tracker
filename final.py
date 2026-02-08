import time
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp

# -----------------------------
# Geometria
# -----------------------------
def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Ângulo em graus no ponto b, formado por a-b-c.
    """
    ba = a - b
    bc = c - b
    eps = 1e-6
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + eps)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def get_world_pt(world_landmarks, idx: int) -> np.ndarray:
    lm = world_landmarks[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def get_img_pt(norm_landmarks, idx: int, w: int, h: int) -> tuple[int, int]:
    lm = norm_landmarks[idx]
    return int(lm.x * w), int(lm.y * h)

def count_extended_fingers(world_landmarks, angle_threshold: float = 160.0) -> int:
    """
    Conta dedos estendidos usando ângulo ~180° (dedo “reto”) nos world landmarks (3D).
    Índices padrão do MediaPipe Hands/HandLandmarker (21 pontos).
    """
    # Thumb: 2-3-4 (MCP-IP-TIP)
    thumb = angle_deg(
        get_world_pt(world_landmarks, 2),
        get_world_pt(world_landmarks, 3),
        get_world_pt(world_landmarks, 4),
    ) > angle_threshold

    # Fingers: MCP-PIP-TIP (pulando DIP para simplificar)
    index_ = angle_deg(get_world_pt(world_landmarks, 5),  get_world_pt(world_landmarks, 6),  get_world_pt(world_landmarks, 8))  > angle_threshold
    middle = angle_deg(get_world_pt(world_landmarks, 9),  get_world_pt(world_landmarks, 10), get_world_pt(world_landmarks, 12)) > angle_threshold
    ring_  = angle_deg(get_world_pt(world_landmarks, 13), get_world_pt(world_landmarks, 14), get_world_pt(world_landmarks, 16)) > angle_threshold
    pinky  = angle_deg(get_world_pt(world_landmarks, 17), get_world_pt(world_landmarks, 18), get_world_pt(world_landmarks, 20)) > angle_threshold

    return int(thumb) + int(index_) + int(middle) + int(ring_) + int(pinky)

# -----------------------------
# Desenho simples (sem drawing_utils)
# -----------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),          # Indicador
    (5, 9), (9, 10), (10, 11), (11, 12),     # Médio
    (9, 13), (13, 14), (14, 15), (15, 16),   # Anelar
    (13, 17), (17, 18), (18, 19), (19, 20),  # Mindinho
    (0, 17)                                   # Base da palma
]

def draw_hand(frame, norm_landmarks, handedness_label: str):
    h, w = frame.shape[:2]

    pts = [get_img_pt(norm_landmarks, i, w, h) for i in range(21)]

    # Conexões
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

    # Pontos
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    # “Palma”: hull com wrist + MCPs
    palm_idx = [0, 5, 9, 13, 17]
    palm_pts = np.array([pts[i] for i in palm_idx], dtype=np.int32)
    hull = cv2.convexHull(palm_pts)
    cv2.polylines(frame, [hull], True, (255, 255, 0), 2)

    # Centro aproximado da palma
    cx, cy = palm_pts.mean(axis=0).astype(int)
    cv2.circle(frame, (cx, cy), 6, (255, 255, 0), -1)

    cv2.putText(frame, handedness_label, (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

# -----------------------------
# Main
# -----------------------------
def main():
    model_path = "models/hand_landmarker.task"  # ajuste para seu caminho

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    # Latência menor:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    history = deque(maxlen=10)  # filtro temporal (estabilidade)
    start = time.monotonic()

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # (Opcional) espelhar para experiência “tipo selfie”
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            timestamp_ms = int((time.monotonic() - start) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            total_fingers = 0

            if result.hand_landmarks and result.hand_world_landmarks:
                for i in range(len(result.hand_landmarks)):
                    norm_lms = result.hand_landmarks[i]
                    world_lms = result.hand_world_landmarks[i]

                    # handedness vem como lista de categorias
                    label = "Hand"
                    try:
                        label = result.handedness[i][0].category_name
                    except Exception:
                        pass

                    draw_hand(frame, norm_lms, label)

                    total_fingers += count_extended_fingers(world_lms)

            # Mapeamento: 0–10 (você pediu 1–10; 0 é bônus)
            total_fingers = max(0, min(10, total_fingers))
            history.append(total_fingers)

            stable = Counter(history).most_common(1)[0][0]

            cv2.putText(frame, f"Fingers: {stable}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

            cv2.imshow("Hand Digit (MediaPipe Tasks)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
