import mediapipe as mp

class HandTracker:
    def __init__(self, model_path: str, num_hands: int = 2):

        # Importa as classes necessárias do MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Define as opções para o landmarker de mãos
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            # Configurações de confiança (valor padrão)
            min_hand_detection_confidence=0.5, 
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Cria o landmarker de mãos com as opções definidas
        self._landmarker = HandLandmarker.create_from_options(options)

    # Detecta mãos em um frame RGB dado o timestamp em milissegundos
    def detect(self, rgb_frame, timestamp_ms: int):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        self._landmarker.close()
