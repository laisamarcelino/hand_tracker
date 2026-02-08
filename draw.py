import cv2
import mediapipe as mp
import numpy as np

# Conexões das landmarks da mão (Tasks)
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

# Índices-chave da palma: wrist + MCPs (base dos dedos)
PALM_IDX = [0, 5, 9, 13, 17]

class HandDrawer:
    def __init__(self, margin=10, font_scale=0.8, thickness=2):
        self.margin = margin
        self.font_scale = font_scale
        self.thickness = thickness

        # OpenCV desenha em BGR
        self.color_points = (0, 0, 255)     # vermelho
        self.color_lines  = (0, 255, 0)     # verde
        self.color_text   = (88, 205, 54)   # verde vibrante
        self.color_palm   = (255, 255, 0)   # amarelo (palma)

    @staticmethod
    # Garante que os pontos fiquem dentro dos limites do frame
    def _clamp(value, low, high):
        return low if value < low else high if value > high else value

    # Desenha o esqueleto da mão, label e mapa da palma no frame BGR
    def draw(self, frame_bgr, result, draw_palm=True, draw_palm_roi=False):
        if not result or not result.hand_landmarks:
            return frame_bgr

        # Dimensões do frame para converter coordenadas normalizadas em pixels
        h, w = frame_bgr.shape[:2]

        # Para cada mão detectada, desenha conexões, pontos e label
        for idx, hand_lms in enumerate(result.hand_landmarks):
            # 1) Normalized -> pixels
            pts = []
            for lm in hand_lms:
                # MediaPipe retorna coordenadas normalizadas (0-1), converte para pixels
                x = self._clamp(int(lm.x * w), 0, w - 1)
                y = self._clamp(int(lm.y * h), 0, h - 1)
                pts.append((x, y))

            # 2) Linhas (connections)
            # HAND_CONNECTIONS é uma lista de pares de índices que formam as conexões entre as landmarks da mão
            for c in HAND_CONNECTIONS:
                a = c.start if hasattr(c, "start") else c[0]
                b = c.end   if hasattr(c, "end") else c[1]
                cv2.line(frame_bgr, pts[a], pts[b], self.color_lines, 2)

            # 3) Pontos
            for (x, y) in pts:
                cv2.circle(frame_bgr, (x, y), 4, self.color_points, -1)

            # 4) Label (Left/Right)
            # O resultado pode conter a classificação de "handedness" (mão esquerda/direita) para cada mão detectada.
            label = "Hand"
            if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                label = result.handedness[idx][0].category_name  # "Left"/"Right"

            # 5) Posição do texto: acima da mão, alinhado à esquerda
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            text_x = min(xs)
            text_y = max(0, min(ys) - self.margin)

            cv2.putText(
                frame_bgr, label, (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX, self.font_scale,
                self.color_text, self.thickness, cv2.LINE_AA
            )

            # Mapa da palma
            if draw_palm:
                # PALM_IDX define os pontos-chave da palma: wrist (0) + MCPs (5,9,13,17)
                palm_pts = np.array([pts[i] for i in PALM_IDX], dtype=np.int32)  # (5,2)
                hull = cv2.convexHull(palm_pts.reshape(-1, 1, 2))                # (N,1,2)
                cv2.polylines(frame_bgr, [hull], True, self.color_palm, 2)

                # Centroide da palma
                cx, cy = palm_pts.mean(axis=0).astype(int)
                cv2.circle(frame_bgr, (cx, cy), 6, self.color_palm, -1)

                # ROI opcional (para debug / futura classificação)
                if draw_palm_roi:
                    # escala ~ distância wrist (0) -> middle_mcp (9)
                    wx, wy = pts[0]
                    mx, my = pts[9]
                    scale = int(np.hypot(mx - wx, my - wy))
                    size = max(80, int(scale * 1.2))
                    
                    # Define a janela ROI centrada na palma, garantindo que fique dentro dos limites do frame
                    x1 = self._clamp(cx - size // 2, 0, w - 1)
                    y1 = self._clamp(cy - size // 2, 0, h - 1)
                    x2 = self._clamp(cx + size // 2, 0, w - 1)
                    y2 = self._clamp(cy + size // 2, 0, h - 1)

                    # retângulo no frame + janela ROI
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), self.color_palm, 2)
                    roi = frame_bgr[y1:y2, x1:x2]
                    if roi.size > 0:
                        cv2.imshow("Palm ROI", roi)

        return frame_bgr
