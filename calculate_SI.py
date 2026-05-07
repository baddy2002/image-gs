import os
import cv2
import numpy as np
import torch
import glob
from PIL import Image
def get_mask_numpy(img_np):
    """
    Riproduce la logica della funzione _mask_empty_areas di model.py
    lavorando solo con Numpy e CPU.
    """
    h, w, _ = img_np.shape
    # creiamo una maschera inizialmente tutta bianca (255)
    # Lo scopo è colorare di nero (0) le zone esterne
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # definiamo i punti di partenza (i 4 angoli) e metà dei bordi
    seeds = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1), (h//2, 0), (h//2, w-1)]

    # Flood Fill: partiamo dagli angoli. 
    # Se il colore è simile (tolleranza 2,2,2), lo consideriamo "vuoto"
    for seed in seeds:
        cv2.floodFill(img_np, flood_mask, seed, (0, 0, 0), 
                    loDiff=(2, 2, 2), upDiff=(2, 2, 2), 
                    flags=4 | cv2.FLOODFILL_FIXED_RANGE)
        
    ## creiamo una mappa di quello che è MOLTO nero ma non è ancora stato riempito
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # solo pixel quasi neri (soglia bassa < 4) e che la flood_mask attuale segna come "non riempiti" (0)
    potential_holes = (gray < 8) & (flood_mask[1:-1, 1:-1] == 0)
    potential_holes = potential_holes.astype(np.uint8) * 255

    # troviamo i centri di queste "isole nere" per non lanciare 1000 floodfill inutili
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(potential_holes, connectivity=4)

    # Per ogni isola nera trovata, lanciamo un piccolo flood fill
    # Saltiamo la label 0 (che è il background della mappa potential_holes)
    for i in range(1, num_labels):
        # Se l'area nera è troppo piccola (es. meno di 2 pixel), forse è solo rumore/dettaglio scuro
        if stats[i, cv2.CC_STAT_AREA] < 2: 
            continue
        
        seed_internal = (int(centroids[i][0]), int(centroids[i][1]))
        cv2.floodFill(img_np, flood_mask, seed_internal, 255, 
                    loDiff=(2, 2, 2), upDiff=(2, 2, 2), 
                    flags=4 | cv2.FLOODFILL_FIXED_RANGE)

    # 3. Creazione maschera finale (senza il bordo di 1px per sicurezza, come da tuo codice)
    background_mask = flood_mask[1:-1, 1:-1]
    mask_np = np.ones((h, w), dtype=np.uint8) * 255
    mask_np[background_mask == 1] = 0
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_np, kernel, iterations=1)   
    border = cv2.subtract(dilated, mask_np)
    mask_final = cv2.subtract(mask_np, border)
    
    return mask_final # Ritorna 0-255

def compute_masked_si(image_path):
    """
    Calcola lo Spatial Information (SI) solo sui pixel della maschera.
    SI è definito come la deviazione standard del gradiente di Sobel.
    """
    # Usiamo PIL per il resize identico a quello del tuo model.py
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize((1024, 1024), Image.LANCZOS)
        img_rgb = np.array(img)
    
    mask = get_mask_numpy(img_rgb)
    
    # Passiamo in scala di grigi per il gradiente
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Calcolo gradienti Sobel
    sob_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Magnitudo del gradiente
    magnitude = cv2.magnitude(sob_x, sob_y)
    
    # Selezioniamo solo i pixel dentro la maschera (mask_final > 128)
    active_pixels = magnitude[mask > 128]
    
    if active_pixels.size == 0:
        return 0.0
    
    # SI = Deviazione Standard della magnitudo dei gradienti
    si_value = np.std(active_pixels)
    return si_value

def main():
    archive_path = "archive" # Assicurati che il percorso sia corretto
    output_file = "textures_complexity_si.csv"
    
    # Tipi di file immagine comuni
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(archive_path, ext)))
    
    print(f"Trovate {len(image_files)} immagini in '{archive_path}'. Inizio calcolo SI...")
    
    with open(output_file, "w") as f:
        f.write("texture,spatial_information\n")
        
        for img_path in sorted(image_files):
            name = os.path.basename(img_path)
            si = compute_masked_si(img_path)
            if si is not None:
                f.write(f"{name},{si:.4f}\n")
                print(f"Processed {name}: SI = {si:.4f}")

    print(f"Calcolo completato. Risultati salvati in {output_file}")

if __name__ == "__main__":
    main()