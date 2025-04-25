import cv2
import numpy as np
import random
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe el modeli
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Kamera başlat
cam = cv2.VideoCapture(0)

# Fare resmi yükleniyor
mouse_image = cv2.imread("mouse.png", cv2.IMREAD_UNCHANGED)  # Alfa destekli PNG

# Fare bilgisi
MOUSE_SIZE = 60
mice = []
score = 0  # Skor
start_time = time.time()  # Oyun başladığında zaman başlar
game_duration = 60  # 1 dakika süre

# Fare pozisyonu üretme
def generate_mice(w, h):
    new_mice = []
    num = random.randint(1, 5)  # Fare sayısı maks 5 olacak
    for _ in range(num):
        new_mice.append({
            "pos": [random.randint(MOUSE_SIZE//2, w - MOUSE_SIZE//2),
                    random.randint(MOUSE_SIZE//2, h - MOUSE_SIZE//2)],
            "vel": [random.uniform(-3, 3), random.uniform(-3, 3)],  # Hareket hızı
            "visible": True
        })
    return new_mice

# Fareyi ekrandan dışarı çıkmaması için hareket ettirme
def move_mouse(mouse, w, h):
    x, y = mouse["pos"]
    vel_x, vel_y = mouse["vel"]
    
    # Fareyi hareket ettir
    x += vel_x
    y += vel_y

    # Ekranın dışına çıkmasın diye kontrol et
    if x <= MOUSE_SIZE//2 or x >= w - MOUSE_SIZE//2:
        vel_x = -vel_x  # Yatay hız yönünü değiştir
    if y <= MOUSE_SIZE//2 or y >= h - MOUSE_SIZE//2:
        vel_y = -vel_y  # Dikey hız yönünü değiştir

    mouse["pos"] = [x, y]
    mouse["vel"] = [vel_x, vel_y]

# Fare çizme fonksiyonu
def draw_mouse_image(frame, mouse_pos, mouse_img, size):
    mouse_img = cv2.resize(mouse_img, (size, size))
    x, y = int(mouse_pos[0]), int(mouse_pos[1])
    h, w, _ = mouse_img.shape
    x1, y1 = x - w // 2, y - h // 2
    x2, y2 = x1 + w, y1 + h

    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return

    roi = frame[y1:y2, x1:x2]

    if mouse_img.shape[2] == 4:
        alpha = mouse_img[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * mouse_img[:, :, c]
    else:
        roi[:] = mouse_img[:, :, :3]

    frame[y1:y2, x1:x2] = roi

# Skor güncelleme
def on_mouse_catch():
    global score
    score += 1  # Skor artışı

# Zamanlayıcıyı ve oyunu kontrol et
def get_remaining_time():
    elapsed_time = time.time() - start_time
    remaining_time = max(0, game_duration - int(elapsed_time))
    return remaining_time

# İlk karede ekran boyutunu öğrenelim
_, frame = cam.read()
frame = cv2.flip(frame, 1)
h, w, _ = frame.shape
mice = generate_mice(w, h)

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    current_time = time.time()

    # Oyun süresi bitip bitmediğini kontrol et
    remaining_time = get_remaining_time()
    if remaining_time == 0:
        cv2.putText(frame, "Oyun Bitti", (w//4, h//2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Skor: {score}", (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Catch the Mice!", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Fareleri hareket ettir
    for mouse in mice:
        if mouse["visible"]:
            move_mouse(mouse, w, h)  # Fareyi hareket ettir
            draw_mouse_image(frame, mouse["pos"], mouse_image, MOUSE_SIZE)  # Fareyi çiz

    # İşaret parmağıyla temas kontrolü
    for hand_landmarks in detection_result.hand_landmarks:
        index_tip = hand_landmarks[8]  # İşaret parmağı ucu
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)

        for mouse in mice:
            if mouse["visible"]:
                dist = np.linalg.norm(np.array([x, y]) - np.array(mouse["pos"]))
                if dist < MOUSE_SIZE // 2:
                    # Fareyi yakaladığında
                    mouse["visible"] = False  # Fareyi gizle
                    # Skoru güncelle
                    on_mouse_catch()

    # 1 saniye sonra yeni fareler
    if all(not m["visible"] for m in mice):
        mice = generate_mice(w, h)

    # Süreyi sol üstte yazdır
    cv2.putText(frame, f"Süre: {remaining_time}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Skoru yaz
    cv2.putText(frame, f"Skor: {score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Göster
    cv2.imshow("Catch the Mice!", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
