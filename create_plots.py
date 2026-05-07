import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Caricamento dati
# Se il file si chiama diversamente, cambia il nome qui
file_path = "experiment_results.csv"
if not os.path.exists(file_path):
    # Creo un file di esempio per test se il file non esiste
    from io import StringIO
    data = """texture,gaussians,beta,psnr,ssim,lpips
hermesyyy01.png,1000,0,18.0,0.21,0.60
hermesyyy01.png,1000,4,18.41,0.20,0.56
hermesyyy01.png,5000,0,22.0,0.27,0.32
hermesyyy01.png,5000,4,23.37,0.26,0.28
"""
    df = pd.read_csv(StringIO(data))
else:
    df = pd.read_csv(file_path)

# Pulizia: ordiniamo per densità per avere linee corrette
df = df.sort_values(by=['texture', 'beta', 'gaussians'])

# Configurazione stile
sns.set_theme(style="whitegrid")
textures = df['texture'].unique()
metrics = ['psnr', 'ssim', 'lpips']
num_try='third'
# Creazione cartella per i grafici
os.makedirs("plots", exist_ok=True)
os.makedirs("plots/"+num_try+"_try", exist_ok=True)
for tex in textures:
    tex_df = df[df['texture'] == tex]
    
    # Creiamo una figura con 3 subplot (uno per metrica)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Analisi Metriche: {tex}", fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # 2. Plot Baseline (Beta = 0 -> Gaussian)
        baseline = tex_df[tex_df['beta'] == 0]
        if not baseline.empty:
            ax.plot(baseline['gaussians'], baseline[metric], 
                    label="Gaussian (Baseline)", 
                    color='black', linestyle='--', linewidth=2.5, marker='o')
        
        # 3. Plot Vari Beta (Beta > 0)
        beta_values = sorted([b for b in tex_df['beta'].unique() if b > 0])
        colors = sns.color_palette("viridis", len(beta_values))
        
        for idx, b in enumerate(beta_values):
            b_df = tex_df[tex_df['beta'] == b]
            ax.plot(b_df['gaussians'], b_df[metric], 
                    label=f"Beta = {b}", 
                    color=colors[idx], linewidth=1.5, marker='s', alpha=0.8)
        
        ax.set_title(metric.upper(), fontsize=14)
        ax.set_xlabel("Numero Gaussiane")
        ax.set_ylabel("Valore")
        if i == 0: # Legenda solo sul primo grafico per pulizia
            ax.legend(title="Configurazione", bbox_to_anchor=(1.05, 1), loc='upper left')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Salvataggio
    save_path = f"plots/{num_try}_try/metrics_{tex.replace('.png', '')}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Grafico salvato: {save_path}")
    plt.show()