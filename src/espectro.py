from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from senales import (generar_tono, generar_portadora, modular_am, cargar_audio,
                     filtro_pasabajos, agregar_ruido, detector_envolvente, espectro)

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


def a_db(mag):
    """Convierte la amplitud a dB relativos al máximo (0 dB = componente más fuerte)."""
    return 20 * np.log10(mag / np.max(mag) + 1e-12)


# =============== A. Tono: comprobación de la teoría ===============
fs = 44100
fm = 100
fc_tono = 2000
mu = 0.8
Ac = 1.0
duracion = 0.1      # 0.1 s -> resolución de 10 Hz exactos

t, m = generar_tono(fm, duracion, fs)
s = modular_am(m, generar_portadora(fc_tono, t, Ac), mu)

f, M = espectro(m, fs)
_, S = espectro(s, fs)
df = fs / len(m)

print("=== Espectro con tono ===")
print(f"N = {len(m)} | Resolución Δf = fs/N = {df:.1f} Hz")
for nombre, freq, X, teorico in [("m(t) en fm", fm, M, 1.0),
                                  ("s(t) en fc - fm", fc_tono - fm, S, mu / 2),
                                  ("s(t) en fc", fc_tono, S, Ac),
                                  ("s(t) en fc + fm", fc_tono + fm, S, mu / 2)]:
    k = int(round(freq / df))
    print(f"{nombre:16s} ({freq:4d} Hz): medida = {X[k]:.3f} | teórica = {teorico:.3f}")

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(f, M)
ax[0].set_title(f"Espectro del mensaje m(t): tono de {fm} Hz")
ax[1].plot(f, S, color="tab:green")
ax[1].set_title(f"Espectro de la señal AM (fc = {fc_tono} Hz, μ = {mu}): portadora + 2 bandas laterales")
ax[1].set_xlim(0, 3000)
ax[1].set_xlabel("Frecuencia (Hz)")
for a in ax:
    a.set_ylabel("Amplitud")
    a.grid(True)
plt.tight_layout()
plt.savefig(RESULTS / "09_espectro_tono.png", dpi=150)


# =============== B. Voz: todo el sistema en frecuencia ===============
B = 4000
fc = 10000
snr_db = 20
semilla = 42

fs_v, x = cargar_audio(BASE / "data" / "voz.wav")
mv = filtro_pasabajos(x, B, fs_v)
mv = mv / np.max(np.abs(mv))
tv = np.arange(len(mv)) / fs_v
sv = modular_am(mv, generar_portadora(fc, tv, Ac), mu)
rv, _ = agregar_ruido(sv, snr_db, semilla)
mv_rec, rect, _ = detector_envolvente(rv, fs_v, B, mu, Ac)

print(f"\n=== Espectro con voz ===")
print(f"N = {len(mv)} | Resolución Δf = {fs_v/len(mv):.3f} Hz")

etapas = [("1) Audio original (sin prefiltro)", x),
          (f"2) Mensaje m(t) con prefiltro de {B/1000:.0f} kHz", mv),
          (f"3) Señal AM recibida (fc = {fc/1000:.0f} kHz, SNR = {snr_db} dB)", rv),
          ("4) Señal rectificada |r(t)| (antes del pasabajos)", rect),
          ("5) Mensaje recuperado (después del pasabajos)", mv_rec)]

fig, ax = plt.subplots(len(etapas), 1, figsize=(10, 12), sharex=True)
for i, (nombre, senal) in enumerate(etapas):
    fv, X = espectro(senal, fs_v)
    ax[i].plot(fv, a_db(X), linewidth=0.5)
    ax[i].set_title(nombre)
    ax[i].set_ylabel("Amplitud (dB)")
    ax[i].set_ylim(-100, 5)
    ax[i].grid(True)
ax[-1].set_xlabel("Frecuencia (Hz)")
ax[-1].set_xlim(0, fs_v / 2)
plt.tight_layout()
plt.savefig(RESULTS / "10_espectro_voz_sistema.png", dpi=150)

plt.show()
