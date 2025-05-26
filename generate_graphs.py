# Copyright (c) Universidad Politecnica Madrid, 2025
# Authors: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>
# Dates:
#  Creation: May. 17, 2025
#  Modification: May 18, 2025
# Documented by: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>

# generate_graphs.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Crear carpeta si no existe
os.makedirs("figures", exist_ok=True)

# Cargar CSV
df = pd.read_csv("data/results/results.csv")
df.columns = df.columns.str.strip()

# Combinar 'image' y 'message' para un eje x detallado
df["input_pair"] = df["image"] + "\n" + df["message"]
df = df.sort_values(by=["image", "message", "method"])

# -------------------------
# Gráfica de SSIM (barras)
# -------------------------
plt.figure(figsize=(max(12, len(df["input_pair"].unique()) * 0.4), 6))
sns.barplot(
    data=df, 
    x="input_pair", 
    y="ssim", 
    hue="method"
)
plt.ylim(0.99, 1.0)  # rango estrecho para resaltar diferencias
plt.xticks(rotation=90)
plt.title("SSIM por imagen+mensaje y método (barras)")
plt.ylabel("SSIM")
plt.xlabel("Imagen + Mensaje")
plt.legend(title="Método", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("figures/ssim_barplot.png")
plt.close()

# -------------------------
# Gráfica de PSNR (barras)
# -------------------------
plt.figure(figsize=(max(12, len(df["input_pair"].unique()) * 0.4), 6))
sns.barplot(
    data=df, 
    x="input_pair", 
    y="psnr", 
    hue="method"
)

plt.xticks(rotation=90)
plt.title("PSNR por imagen+mensaje y método (barras)")
plt.ylabel("PSNR (dB)")
plt.xlabel("Imagen + Mensaje")
plt.legend(title="Método", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("figures/psnr_barplot.png")
plt.close()

# -------------------------
# Gráfica de píxeles modificados
# -------------------------
plt.figure(figsize=(max(12, len(df["input_pair"].unique()) * 0.4), 6))
sns.barplot(
    data=df,
    x="input_pair",
    y="pixels_modified",
    hue="method"
)
plt.xticks(rotation=90)
plt.title("Píxeles modificados por imagen+mensaje y método")
plt.ylabel("Píxeles modificados")
plt.xlabel("Imagen + Mensaje")
plt.legend(title="Método", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("figures/pixels_modified_barplot.png")
plt.close()

# -------------------------
# Gráfica de bits modificados
# -------------------------
plt.figure(figsize=(max(12, len(df["input_pair"].unique()) * 0.4), 6))
sns.barplot(
    data=df,
    x="input_pair",
    y="bits_modified",
    hue="method"
)
plt.xticks(rotation=90)
plt.title("Bits modificados por imagen+mensaje y método")
plt.ylabel("Bits modificados")
plt.xlabel("Imagen + Mensaje")
plt.legend(title="Método", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("figures/bits_modified_barplot.png")
plt.close()
