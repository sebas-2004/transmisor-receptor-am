from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from senales import (cargar_audio, filtro_pasabajos, generar_portadora, modular_am,
                     agregar_ruido, detector_envolvente, receptor_coherente)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# ---------- Parámetros (los mismos de la entrega 2) ----------
B = 4000
fc = 10000
Ac = 1.0
mu = 0.8
semilla = 42

fs, x = cargar_audio(DATA / "voz.wav")
m = filtro_pasabajos(x, B, fs)
m = m / np.max(np.abs(m))
t = np.arange(len(m)) / fs
c = generar_portadora(fc, t, Ac)
s = modular_am(m, c, mu)


def correlacion(a, b):
    return np.corrcoef(a, b)[0, 1]


# =============== 1. Calidad vs SNR para los dos receptores ===============
snrs = np.arange(0, 55, 5)
corr_env, corr_coh = [], []

print("=== Correlación vs SNR (μ = 0.8) ===")
for snr in snrs:
    r, _ = agregar_ruido(s, snr, semilla)
    ce = correlacion(m, detector_envolvente(r, fs, B, mu, Ac)[0])
    cc = correlacion(m, receptor_coherente(r, t, fc, fs, B, mu, Ac))
    corr_env.append(ce)
    corr_coh.append(cc)
    print(f"SNR = {snr:2d} dB | envolvente = {ce:.4f} | coherente = {cc:.4f}")

plt.figure(figsize=(9, 5))
plt.plot(snrs, corr_env, "o-", label="Detector de envolvente")
plt.plot(snrs, corr_coh, "s-", label="Receptor coherente")
plt.title("Calidad del mensaje recuperado vs SNR del canal (voz, μ = 0,8)")
plt.xlabel("SNR (dB)")
plt.ylabel("Correlación con el mensaje original")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "11_correlacion_vs_snr.png", dpi=150)


# =============== 2. Hipótesis: ¿el filtro del receptor limita la calidad? ===============
cortes = [2000, 3000, 4000, 5000, 6000, 8000]
corr_corte = []

print("\n=== Sin ruido: correlación vs frecuencia de corte del receptor ===")
for fcorte in cortes:
    ce = correlacion(m, detector_envolvente(s, fs, fcorte, mu, Ac)[0])
    corr_corte.append(ce)
    print(f"fcorte = {fcorte:5d} Hz | correlación = {ce:.4f}")

plt.figure(figsize=(9, 5))
plt.plot(np.array(cortes) / 1000, corr_corte, "o-")
plt.axvline(B / 1000, color="r", linestyle="--", label="Ancho de banda del mensaje (4 kHz)")
plt.title("Sin ruido: efecto de la frecuencia de corte del receptor")
plt.xlabel("Frecuencia de corte del pasabajos del receptor (kHz)")
plt.ylabel("Correlación")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "12_correlacion_vs_fcorte.png", dpi=150)


# =============== 3. Sobremodulación: envolvente vs coherente ===============
mu_sobre = 1.5
s_sobre = modular_am(m, c, mu_sobre)
m_env = detector_envolvente(s_sobre, fs, B, mu_sobre, Ac)[0]
m_coh = receptor_coherente(s_sobre, t, fc, fs, B, mu_sobre, Ac)

print(f"\n=== Sobremodulación (μ = {mu_sobre}, sin ruido) ===")
print(f"Envolvente: correlación = {correlacion(m, m_env):.4f}")
print(f"Coherente:  correlación = {correlacion(m, m_coh):.4f}")

# Ventana de 30 ms donde el mensaje es más negativo (ahí ocurre la sobremodulación)
centro = np.argmin(m)
mitad = int(0.015 * fs)
v = slice(max(centro - mitad, 0), centro + mitad)

plt.figure(figsize=(10, 5))
plt.plot(t[v] * 1000, m[v], "k--", linewidth=2, label="Original")
plt.plot(t[v] * 1000, m_env[v], label="Detector de envolvente")
plt.plot(t[v] * 1000, m_coh[v], label="Receptor coherente")
plt.title(f"Sobremodulación (μ = {mu_sobre}): comparación de receptores")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "13_sobremodulacion_receptores.png", dpi=150)

plt.show()