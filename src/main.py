from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from senales import generar_tono

# Carpeta results/ (funciona sin importar desde dónde se ejecute)
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

# ---------- Parámetros ----------
fs = 44100          # frecuencia de muestreo (Hz)
Ts = 1 / fs         # periodo de muestreo (s)
fm = 100            # frecuencia del mensaje (Hz)
duracion = 0.05     # duración (s)

# ---------- 1. Mensaje ----------
t, m = generar_tono(fm, duracion, fs)

print("=== Mensaje m(t) ===")
print(f"fs = {fs} Hz | Ts = {Ts*1e6:.2f} us")
print(f"Muestras N = {len(m)} | Duración = {len(m)/fs:.3f} s")
print(f"Min = {m.min():.3f} | Max = {m.max():.3f} | Media = {m.mean():.4f}")
print(f"Periodo T = {1/fm*1000:.1f} ms")

plt.figure(figsize=(10, 4))
plt.plot(t * 1000, m)
plt.title(f"Mensaje m(t): tono de {fm} Hz")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud (normalizada)")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS / "01_mensaje_tono.png", dpi=150)
plt.show()