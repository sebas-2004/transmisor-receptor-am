from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from senales import generar_tono, generar_portadora, modular_am, envolvente_am

# Carpeta results/ (funciona sin importar desde dónde se ejecute)
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

# ---------- Parámetros ----------
fs = 44100          # frecuencia de muestreo (Hz)
Ts = 1 / fs         # periodo de muestreo (s)
fm = 100            # frecuencia del mensaje (Hz)
duracion = 0.05     # duración (s)
fc = 2000           # frecuencia de la portadora (Hz)
Ac = 1.0            # amplitud de la portadora

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

# ---------- 2. Portadora ----------
c = generar_portadora(fc, t, Ac)

print("\n=== Portadora c(t) ===")
print(f"fc = {fc} Hz | Ac = {Ac} | Periodo = {1/fc*1000:.2f} ms")
print(f"Muestras por ciclo de portadora = {fs/fc:.1f}")
print(f"Relación fc/fm = {fc/fm:.0f}")
print(f"Nyquist: fs/2 = {fs/2:.0f} Hz > fc + fm = {fc + fm} Hz -> {fs/2 > fc + fm}")

zoom = t <= 0.005   # primeros 5 ms, para ver los ciclos de la portadora

fig, ax = plt.subplots(3, 1, figsize=(10, 8))

ax[0].plot(t * 1000, m)
ax[0].set_title(f"Mensaje m(t) - {fm} Hz")

ax[1].plot(t * 1000, c, color="tab:orange")
ax[1].set_title(f"Portadora c(t) - {fc} Hz")

ax[2].plot(t[zoom] * 1000, c[zoom], color="tab:orange", label="Portadora c(t)")
ax[2].plot(t[zoom] * 1000, m[zoom], linewidth=2, label="Mensaje m(t)")
ax[2].set_title("Zoom: primeros 5 ms")
ax[2].legend(loc="upper right")

for a in ax:
    a.set_xlabel("Tiempo (ms)")
    a.set_ylabel("Amplitud")
    a.grid(True)

plt.tight_layout()
plt.savefig(RESULTS / "02_mensaje_y_portadora.png", dpi=150)
# ---------- 3. Modulación AM con distintos índices ----------
indices = [0.5, 1.0, 1.5]

fig, ax = plt.subplots(len(indices), 1, figsize=(10, 9), sharex=True)

print("\n=== Modulación AM ===")
for i, mu in enumerate(indices):
    s = modular_am(m, c, mu)
    env = envolvente_am(m, mu, Ac)

    # Fracción de la potencia que va en las bandas laterales (las que llevan el mensaje)
    eficiencia = mu**2 / (2 + mu**2) * 100

    if mu < 1:
        estado = "submodulación"
    elif mu == 1:
        estado = "modulación al 100 %"
    else:
        estado = "SOBREMODULACIÓN"

    print(f"mu = {mu}: max = {s.max():.2f} | min = {s.min():.2f} | "
          f"envolvente mín = {env.min():.2f} | eficiencia = {eficiencia:.1f} % -> {estado}")

    ax[i].plot(t * 1000, s, color="tab:green", linewidth=0.8, label="s(t)")
    ax[i].plot(t * 1000, env, "r--", label="Envolvente Ac[1+μm(t)]")
    ax[i].plot(t * 1000, -env, "r--")
    ax[i].set_title(f"Señal AM con μ = {mu} ({estado})")
    ax[i].set_ylabel("Amplitud")
    ax[i].grid(True)
    ax[i].legend(loc="upper right")

ax[-1].set_xlabel("Tiempo (ms)")
plt.tight_layout()
plt.savefig(RESULTS / "03_am_indices_modulacion.png", dpi=150)
plt.show()   # muestra todas las figuras; cerrarlas para terminar
