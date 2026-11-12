from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

fs = 44100
fcorte = 4000

# ---------- Respuesta en frecuencia para distintos órdenes ----------
plt.figure(figsize=(10, 5))
for orden in [2, 4, 8]:
    b, a = butter(orden, fcorte, btype="low", fs=fs)
    f, H = freqz(b, a, worN=4096, fs=fs)
    H_efectiva = np.abs(H) ** 2          # filtfilt aplica el filtro dos veces -> |H|^2
    plt.plot(f / 1000, 20 * np.log10(H_efectiva + 1e-12), label=f"Butterworth orden {orden}")

plt.axvline(4, color="k", linestyle="--", label="Frecuencia de corte (4 kHz)")
plt.axvline(4.1, color="r", linestyle=":", label="Componente de aliasing (4,1 kHz)")
plt.axvline(16, color="g", linestyle=":", label="Copia en 2fc empieza en 16 kHz")
plt.title("Respuesta en frecuencia efectiva del pasabajos (con filtfilt)")
plt.xlabel("Frecuencia (kHz)")
plt.ylabel("|H(f)|² (dB)")
plt.ylim(-120, 5)
plt.xlim(0, fs / 2000)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "15_respuesta_filtro.png", dpi=150)

# ---------- Atenuación del filtro usado (orden 4) en frecuencias clave ----------
b, a = butter(4, fcorte, btype="low", fs=fs)
print("=== Filtro usado: Butterworth orden 4, fcorte = 4 kHz, con filtfilt ===")
for f0, nombre in [(1000, "voz"), (3000, "voz"), (4000, "corte"),
                   (4100, "aliasing"), (16000, "inicio copia 2fc"), (20000, "2fc")]:
    _, h = freqz(b, a, worN=[f0], fs=fs)
    print(f"{f0:6d} Hz ({nombre:17s}): {20 * np.log10(np.abs(h[0]) ** 2):7.1f} dB")

plt.show()