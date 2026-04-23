import os
import subprocess
import pandas as pd
import glob
from PIL import Image
from kaggle_secrets import UserSecretsClient

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# --- CONFIGURAZIONE ---
csv_path = "experiment_results.csv"
input_dir = "media/textures"
small_dir = "media/textures_small"
os.makedirs(small_dir, exist_ok=True)

images = glob.glob(f"{input_dir}/*.png")[:30]
densities = [1000, 2500, 5000, 10000, 20000]
betas = [1, 2, 4, 8, 10]

# --- 1. LOGIN E RESUME DA WANDB ---
completed = set()
results = []
run = None

if HAS_WANDB:
    try:
        user_secrets = UserSecretsClient()
        wandb_key = user_secrets.get_secret("WANDB_API_KEY")
        wandb.login(key=wandb_key)
        
        # Iniziamo il run
        run = wandb.init(project="beta-splatting-texture", name="marathon_run_final", resume=True)
        
        # Tentativo di recupero CSV dal cloud
        try:
            artifact = run.use_artifact('experiment_results:latest')
            artifact_dir = artifact.download()
            df_existing = pd.read_csv(os.path.join(artifact_dir, csv_path))
            
            # Popoliamo i completati per saltarli
            completed = set(zip(df_existing['texture'], df_existing['gaussians'], df_existing['beta']))
            results = df_existing.to_dict('records')
            df_existing.to_csv(csv_path, index=False) # Copia locale di sicurezza
            print(f"Cloud Resume: {len(completed)} esperimenti saltati.")
        except:
            print("Nessun artifact trovato. Parto da zero o da file locale.")
            if os.path.exists(csv_path):
                df_local = pd.read_csv(csv_path)
                completed = set(zip(df_local['texture'], df_local['gaussians'], df_local['beta']))
                results = df_local.to_dict('records')

    except Exception as e:
        print(f"⚠️ Errore WandB: {e}")
        HAS_WANDB = False

# --- 2. LOOP ESPERIMENTI ---
for img_path in images:
    img_name = os.path.basename(img_path)
    small_path = os.path.join(small_dir, img_name)
    
    if not os.path.exists(small_path):
        with Image.open(img_path) as img:
            img = img.resize((1024, 1024), Image.LANCZOS)
            img.save(small_path)

    for n in densities:
        for b in betas:
            if (img_name, n, b) in completed:
                continue
            
            exp_name = f"exp_{img_name.split('.')[0]}_n{n}_b{b}"
            print(f"\n>>> Running: {exp_name}")
            input_path_for_cmd = f"textures_small/{img_name}"

            # NB: Assicurati che main.py accetti questo percorso relativo
            cmd = ["python", "main.py", 
                   f"--input_path={input_path_for_cmd}", 
                   f"--exp_name={exp_name}", 
                   f"--num_gaussians={str(n)}", 
                   f"--beta_value={str(b)}", 
                   "--quantize", "--use_mask"]
            
            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                
                final_metrics = {}
                for line in process.stdout:
                    print(line, end="")
                    if "PSNR:" in line and "LPIPS:" in line:
                        try:
                            psnr_match  = re.search(r'PSNR:\s*([\d.]+)', line)
                            ssim_match  = re.search(r'SSIM:\s*([\d.]+)', line)
                            lpips_match = re.search(r'LPIPS:\s*([\d.]+)', line)
                            
                            if psnr_match and ssim_match and lpips_match:
                                final_metrics = {
                                    "texture":   img_name,
                                    "gaussians": n,
                                    "beta":      b,
                                    "psnr":      float(psnr_match.group(1)),
                                    "ssim":      float(ssim_match.group(1)),
                                    "lpips":     float(lpips_match.group(1))
                                }
                        except Exception as parse_err:
                            print(f"Parse error: {parse_err} on line: {line}")
                            continue
                process.wait()

                if final_metrics:
                    results.append(final_metrics)
                    # Aggiorna CSV locale
                    pd.DataFrame(results).to_csv(csv_path, index=False)
                    
                    # Log su WandB (Dati finali)
                    if HAS_WANDB:
                        wandb.log(final_metrics)
                        # Ogni 10 esperimenti facciamo il backup del CSV sul cloud
                        if len(results) % 10 == 0:
                            new_art = wandb.Artifact('experiment_results', type='dataset')
                            new_art.add_file(csv_path)
                            run.log_artifact(new_art)
                            
            except Exception as e:
                print(f"Errore critico su {exp_name}: {e}")
                continue

if HAS_WANDB:
    # Caricamento finale del CSV completo
    final_art = wandb.Artifact('experiment_results', type='dataset')
    final_art.add_file(csv_path)
    run.log_artifact(final_art)
    wandb.finish()