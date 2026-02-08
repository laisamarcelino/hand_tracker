import cv2
import time 
import numpy as np
from hand_tracker import HandTracker
from draw import HandDrawer

MODEL_PATH = "./models/hand_landmarker.task"

## Captura vídeo da webcam
cap = cv2.VideoCapture(0)

# Latência menor:
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Não foi possível abrir a webcam (index 0).")

# Inicializa o rastreador de mãos
tracker = HandTracker(MODEL_PATH, num_hands=2)

# Inicializa o desenhador de mãos
drawer = HandDrawer()

# Tempo inicial
start = time.monotonic()

while True:
    sucess, img_bgr = cap.read()
    if not sucess or img_bgr is None:
        print("Falha ao capturar o frame da webcam.")
        break

    img_bgr = cv2.flip(img_bgr, 1) # Espelha a imagem para efeito "selfie"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Converte BGR para RGB necessário para mediaPipe
    img_rgb = np.ascontiguousarray(img_rgb)  # cria uma cópia contígua da img na memória
    
    # Calcula o timestamp em milissegundos / necessario para saber a ordem e o espaço dos frames
    timestamp_ms = int((time.monotonic() - start) * 1000)

    # Detecta as mãos
    result = tracker.detect(img_rgb, timestamp_ms)

    # DEBUG
    n_land = len(result.hand_landmarks) if result.hand_landmarks else 0
    n_world = len(result.hand_world_landmarks) if result.hand_world_landmarks else 0
    n_hand  = len(result.handedness) if result.handedness else 0
    print(f"hands={n_land} world={n_world} handedness={n_hand}", end="\r")

    # desenha em cima do frame BGR (o que o OpenCV mostra)
    img_bgr = drawer.draw(img_bgr, result)

    # Exibe a webcam
    cv2.imshow("Webcam", img_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fecha as janelas e libera o vídeo
cap.release()
cv2.destroyAllWindows()