import cv2
img = cv2.imread("/Users/andreabenassi/Desktop/Uni/Magistrale/AICG/Progetto/image-gs/media/textures/RealWorldTexturedThings_chunk2/RWT202/castpol01.png", cv2.IMREAD_UNCHANGED)
print(f"Canali immagine: {img.shape[2]}")