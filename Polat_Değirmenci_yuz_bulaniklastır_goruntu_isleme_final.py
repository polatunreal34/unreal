from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Yüzü bulanıklaştıran fonksiyon
def sisli_surati_uygula(goruntu, tespit_sonucu):
    surat_noktalar = tespit_sonucu.face_landmarks
    kopya_resim = np.copy(goruntu)

    for surat in surat_noktalar:
        yukseklik, genislik, _ = kopya_resim.shape

        x_degerleri = [int(nokta.x * genislik) for nokta in surat]
        y_degerleri = [int(nokta.y * yukseklik) for nokta in surat]

        x_min = max(min(x_degerleri) - 20, 0)
        y_min = max(min(y_degerleri) - 20, 0)
        x_max = min(max(x_degerleri) + 20, genislik)
        y_max = min(max(y_degerleri) + 20, yukseklik)

        surat_alani = kopya_resim[y_min:y_max, x_min:x_max]
        sisli_surat = cv2.GaussianBlur(surat_alani, (99, 99), 30)
        kopya_resim[y_min:y_max, x_min:x_max] = sisli_surat

    return kopya_resim

# Yüz tanıma modeli ayarı
temel_secenekler = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
secenekler = vision.FaceLandmarkerOptions(
    base_options=temel_secenekler,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
surat_bulucu = vision.FaceLandmarker.create_from_options(secenekler)

# Kamerayı başlat
kamera = cv2.VideoCapture(0)
while kamera.isOpened():
    basarili_mi, kare = kamera.read()
    if not basarili_mi:
        break

    rgb_kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
    mp_goruntu = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_kare)

    tespit = surat_bulucu.detect(mp_goruntu)

    cikti_resim = sisli_surati_uygula(mp_goruntu.numpy_view(), tespit)
    cv2.imshow("Bulanık Yüz", cv2.cvtColor(cikti_resim, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
