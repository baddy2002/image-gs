import os
import subprocess
import pandas as pd
from PIL import Image
import glob

# --- CONFIGURAZIONE ---
input_dir = "media/textures"
output_dir = "results"
csv_path = "experiment_results.csv"
images = glob.glob(f"{input_dir}/*.png")[:30] # Prendi le prime 30

densities = [1000, 2500, 5000, 10000, 20000]
betas = [1, 2, 4, 8, 10]

results = []

os.makedirs("media/textures_small", exist_ok=True)

for img_path in images:
    img_name = os.path.basename(img_path)
    small_path = f"media/textures_small/{img_name}"
    
    # 1. Ridimensionamento (se non esiste già)
    if not os.path.exists(small_path):
        with Image.open(img_path) as img:
            img = img.resize((1024, 1024), Image.LANCZOS)
            img.save(small_path)
            print(f"Ridimensionata: {img_name}")

    # 2. Ciclo Parametri
    for n in densities:
        for b in betas:
            exp_name = f"exp_{img_name.split('.')[0]}_n{n}_b{b}"
            print(f"\n>>> Running: {exp_name}")
            
            # Comando per lanciare il tuo main.py
            cmd = [
                "python", "main.py",
                f"--input_path={small_path}",
                f"--exp_name={exp_name}",
                f"--num_gaussians={n}",
                f"--beta_value={b}",
                "--quantize",
                "--use_mask"

            ]
            
            try:
                # Eseguiamo e catturiamo l'output per loggare i risultati finali
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                
                final_metrics = {}
                for line in process.stdout:
                    print(line, end="") # Vediamo i log in tempo reale su Kaggle
                    if "PSNR:" in line and "SSIM:" in line and "LPIPS:" in line:
                        # Esempio parsing: "PSNR: 30.45 | SSIM: 0.3100 | LPIPS: 0.1302"
                        parts = line.split("|")
                        final_metrics['psnr'] = float(parts[0].split(":")[1].strip())
                        final_metrics['ssim'] = float(parts[1].split(":")[1].strip())
                        final_metrics['lpips'] = float(parts[2].split(":")[1].strip())

                process.wait()

            except Exception as e:
                print(f"Errore durante l'esecuzione di {exp_name}: {e}")
                continue
            # Salvataggio dati
            if final_metrics:
                results.append({
                    "texture": img_name,
                    "gaussians": n,
                    "beta": b,
                    **final_metrics
                })
                
                # Salva il CSV ad ogni passo (così se Kaggle crasha non perdi tutto)
                pd.DataFrame(results).to_csv(csv_path, index=False)
print("\n--- ESPERIMENTI COMPLETATI ---")