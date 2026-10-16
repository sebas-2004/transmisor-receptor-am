from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from senales import (cargar_audio, guardar_audio, generar_portadora, modular_am,
                     envolvente_am, agregar_ruido, detector_envolvente,
                     filtro_pasabajos)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

# ---------- Parámetros ----------
B = 4000        # ancho de banda del mensaje (Hz)
fc = 10000      # frecuencia de la portadora (Hz)
Ac = 1.0        # amplitud de la portadora
mu = 0.8        # índice de modulación
snr_db = 20     # SNR del canal (dB)
semilla = 42    # semilla del ruido (reproducibilidad)

# ---------- 1. Cargar y preparar el mensaje ----------
fs, x = cargar_audio(DATA / "voz.wav")
Ts = 1 / fs

m = filtro_pasabajos(x, B, fs)      # prefiltro: limitar el mensaje a B
m = m / np.max(np.abs(m))           # volver a normalizar a [-1, 1]
t = np.arange(len(m)) / fs

print("=== Mensaje de audio ===")
print(f"fs = {fs} Hz | Ts = {Ts*1e6:.2f} us")
print(f"Muestras N = {len(m)} | Duración = {len(m)/fs:.2f} s")
print(f"Min = {m.min():.3f} | Max = {m.max():.3f} | Media = {m.mean():.5f} | "
      f"RMS = {np.sqrt(np.mean(m**2)):.3f}")
print(f"Nyquist: fs/2 = {fs/2:.0f} Hz > fc + B = {fc + B} Hz -> {fs/2 > fc + B}")

# ---------- 2. Transmisor, canal y receptor ----------
c = generar_portadora(fc, t, Ac)
s = modular_am(m, c, mu)
env = envolvente_am(m, mu, Ac)
r, _ = agregar_ruido(s, snr_db, semilla)
m_rec, _, _ = detector_envolvente(r, fs, B, mu, Ac)

ecm = np.mean((m - m_rec) ** 2)
corr = np.corrcoef(m, m_rec)[0, 1]
print(f"\n=== Recepción (SNR = {snr_db} dB) ===")
print(f"ECM = {ecm:.5f} | correlación = {corr:.4f}")

# Guardar audios para escucharlos
guardar_audio(RESULTS / "voz_mensaje.wav", m, fs)
guardar_audio(RESULTS / "voz_am_transmitida.wav", r, fs)
guardar_audio(RESULTS / "voz_recuperada.wav", m_rec, fs)

# ---------- 3. Selección de ventana ----------
# Ventana de 30 ms centrada donde la voz es más fuerte
centro = np.argmax(np.abs(m))
mitad = int(0.015 * fs)
v = slice(max(centro - mitad, 0), centro + mitad)

# ---------- 4. Gráficas ----------
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(t, m, linewidth=0.6)
ax[0].axvspan(t[v][0], t[v][-1], color="orange", alpha=0.3, label="Ventana analizada")
ax[0].set_title("Mensaje de voz m(t) completo (normalizado, limitado a 4 kHz)")
ax[0].set_xlabel("Tiempo (s)")
ax[0].legend(loc="upper right")
ax[1].plot(t[v] * 1000, m[v])
ax[1].set_title("Ventana de 30 ms del mensaje")
ax[1].set_xlabel("Tiempo (ms)")
for a in ax:
    a.set_ylabel("Amplitud")
    a.grid(True)
plt.tight_layout()
plt.savefig(RESULTS / "07_voz_mensaje.png", dpi=150)

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(t[v] * 1000, s[v], color="tab:green", linewidth=0.6, label="s(t)")
ax[0].plot(t[v] * 1000, env[v], "r--", label="Envolvente")
ax[0].set_title(f"Señal AM de la voz (fc = {fc/1000:.0f} kHz, μ = {mu}) - ventana de 30 ms")
ax[0].legend(loc="upper right")
ax[1].plot(t[v] * 1000, m[v], "k--", label="Original")
ax[1].plot(t[v] * 1000, m_rec[v], color="tab:blue", label="Recuperado")
ax[1].set_title(f"Mensaje recuperado con SNR = {snr_db} dB (correlación = {corr:.3f})")
ax[1].set_xlabel("Tiempo (ms)")
ax[1].legend(loc="upper right")
for a in ax:
    a.set_ylabel("Amplitud")
    a.grid(True)
plt.tight_layout()
plt.savefig(RESULTS / "08_voz_am_y_recepcion.png", dpi=150)

plt.show()