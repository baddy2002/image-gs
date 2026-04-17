import re
import pandas as pd

def parse_logs(log_text):
    results = []
    # Cerchiamo il nome dell'esperimento
    # Esempio: >>> Running: exp_hermesyyy01_n1000_b1
    exp_pattern = re.compile(r">>> Running: exp_(.*)_n(\d+)_b(\d+)")
    # Cerchiamo le metriche finali
    metrics_pattern = re.compile(r"PSNR: (\d+\.\d+) \| SSIM: (\d+\.\d+) \| LPIPS: (\d+\.\d+)")

    current_exp = None
    
    lines = log_text.split('\n')
    for line in lines:
        exp_match = exp_pattern.search(line)
        if exp_match:
            current_exp = {
                "texture": exp_match.group(1) + ".png",
                "gaussians": int(exp_match.group(2)),
                "beta": int(exp_match.group(3))
            }
        
        metrics_match = metrics_pattern.search(line)
        if metrics_match and current_exp:
            current_exp["psnr"] = float(metrics_match.group(1))
            current_exp["ssim"] = float(metrics_match.group(2))
            current_exp["lpips"] = float(metrics_match.group(3))
            results.append(current_exp.copy())
            current_exp = None # Reset per il prossimo
            
    return pd.DataFrame(results)

# Incolla qui i tuoi log o caricali da file
with open('logs.txt', 'r') as f: logs = f.read()
df_recuperato = parse_logs(logs)
df_recuperato.to_csv("experiment_results.csv", index=False)
print(f"Recuperati {len(df_recuperato)} esperimenti!")