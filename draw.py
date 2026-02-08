import cv2
import mediapipe as mp

# Conexões das landmarks da mão (Tasks)
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

class HandDrawer:
    def __init__(self, margin=10, font_scale=0.8, thickness=2):
        self.margin = margin
        self.font_scale = font_scale
        self.thickness = thickness

        # OpenCV desenha em BGR
        self.color_points = (0, 0, 255)     # vermelho
        self.color_lines  = (0, 255, 0)     # verde
        self.color_text   = (88, 205, 54)   # verde vibrante

    @staticmethod
    # Limitador de valores (clamp) para garantir que os pontos fiquem dentro da imagem
    def _clamp(value, low, high):
        return low if value < low else high if value > high else value

    # Desenha as landmarks e conexões no frame BGR
    def draw(self, frame_bgr, result):
        # result é HandLandmarkerResult (Tasks)
        if not result or not result.hand_landmarks:
            return frame_bgr

        # Dimensões do frame
        h, w = frame_bgr.shape[:2]

        # Para cada mão detectada
        for idx, hand_lms in enumerate(result.hand_landmarks):
            # 1) Normalized landmarks -> pixels (com clamp por segurança)
            pts = []
            for lm in hand_lms:
                x = int(lm.x * w)
                y = int(lm.y * h)
                x = self._clamp(x, 0, w - 1)
                y = self._clamp(y, 0, h - 1)
                pts.append((x, y))

            # 2) Linhas (connections)
            # Em Tasks, cada item é Connection(start=?, end=?)
            for c in HAND_CONNECTIONS:
                a = c.start if hasattr(c, "start") else c[0]
                b = c.end   if hasattr(c, "end") else c[1]
                cv2.line(frame_bgr, pts[a], pts[b], self.color_lines, 2)

            # 3) Pontos
            for (x, y) in pts:
                cv2.circle(frame_bgr, (x, y), 4, self.color_points, -1)

            # 4) Label (Left/Right)
            label = "Hand"
            if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                label = result.handedness[idx][0].category_name  # "Left"/"Right"

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            text_x = min(xs)
            text_y = max(0, min(ys) - self.margin)

            cv2.putText(
                frame_bgr, label, (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX, self.font_scale,
                self.color_text, self.thickness, cv2.LINE_AA
            )

        return frame_bgr
