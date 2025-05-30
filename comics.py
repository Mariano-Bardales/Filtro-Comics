import cv2
import numpy as np

# Cargar imagen
img = cv2.imread("galgadot.jpg")
if img is None:
    print("No se pudo abrir la imagen")
    exit()

# Aumentar saturación para colores más vivos
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv[..., 1] = np.clip(hsv[..., 1] * 1.8, 0, 255)  # Más saturación
brillante = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Posterizar colores (reducción de colores)
Z = np.float32(brillante).reshape((-1, 3))
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, _, centros = cv2.kmeans(Z, 6, None, crit, 4, cv2.KMEANS_RANDOM_CENTERS)
reducido = centros[_].reshape(brillante.shape).astype(np.uint8)

# Detectar bordes bien definidos
gris = cv2.cvtColor(brillante, cv2.COLOR_BGR2GRAY)
bordes = cv2.adaptiveThreshold(cv2.medianBlur(gris, 5), 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 7, 5)

# Combinar bordes con imagen reducida
bordes_color = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)
comic_final = cv2.bitwise_and(reducido, bordes_color)

# Mostrar y guardar
cv2.imshow("Comic Estilo Marcado", comic_final)
cv2.imwrite("../imagenes/TomyComicMarcado.jpg", comic_final)
print("¡Filtro cómic aplicado con mayor intensidad!")

cv2.waitKey(0)
cv2.destroyAllWindows()