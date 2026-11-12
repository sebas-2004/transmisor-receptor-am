import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from senales import (filtro_pasabajos, generar_portadora, modular_am, agregar_ruido,
                     detector_envolvente, receptor_coherente, espectro)

# ---------- Parámetros del sistema ----------
fs = 44100        # frecuencia de muestreo (Hz)
duracion = 4      # segundos de grabación
B = 4000          # ancho de banda del mensaje (Hz)
fc = 10000        # portadora (Hz)
Ac = 1.0
mu = 0.8


def correlacion(a, b):
    return np.corrcoef(a, b)[0, 1]


def grabar():
    """Graba la voz por el micrófono y la deja lista como mensaje."""
    input(f"\nPresiona Enter y habla durante {duracion} s...")
    print("Grabando...")
    x = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype="float64")
    sd.wait()
    x = x[:, 0] - np.mean(x)            # mono y sin nivel DC
    x = filtro_pasabajos(x, B, fs)      # limitar el mensaje a B = 4 kHz
    print("Listo.")
    return x / np.max(np.abs(x))        # normalizar a [-1, 1]


def reproducir(senal, nombre, volumen=0.8):
    print(f"  Reproduciendo: {nombre}")
    sd.play(volumen * senal / np.max(np.abs(senal)), fs)
    sd.wait()


def transmitir(m):
    """Transmisor: mensaje -> señal AM."""
    t = np.arange(len(m)) / fs
    return t, modular_am(m, generar_portadora(fc, t, Ac), mu)


m = grabar()
t, s = transmitir(m)

while True:
    entrada = input("\nSNR del canal en dB (Enter = 20 | g = grabar otra vez | q = salir): ").strip().lower()
    if entrada == "q":
        break
    if entrada == "g":
        m = grabar()
        t, s = transmitir(m)
        continue
    snr = float(entrada) if entrada else 20.0

    # Canal y receptores (ruido nuevo en cada prueba)
    r, _ = agregar_ruido(s, snr)
    m_env = detector_envolvente(r, fs, B, mu, Ac, sobremuestreo=2)[0]
    m_coh = receptor_coherente(r, t, fc, fs, B, mu, Ac)

    print(f"\nSNR = {snr:.0f} dB | correlación envolvente = {correlacion(m, m_env):.4f} "
          f"| coherente = {correlacion(m, m_coh):.4f}")

    # Gráficas: ventana de 30 ms y espectro de la señal recibida
    plt.close("all")
    centro = np.argmax(np.abs(m))
    mitad = int(0.015 * fs)
    v = slice(max(centro - mitad, 0), centro + mitad)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    ax[0].plot(t[v] * 1000, m[v], "k--", linewidth=2, label="Original")
    ax[0].plot(t[v] * 1000, m_env[v], label="Envolvente (sobremuestreo x2)")
    ax[0].plot(t[v] * 1000, m_coh[v], label="Coherente")
    ax[0].set_title(f"Mensaje recuperado (SNR = {snr:.0f} dB) - ventana de 30 ms")
    ax[0].set_xlabel("Tiempo (ms)")
    ax[0].set_ylabel("Amplitud")
    ax[0].legend(loc="upper right")
    ax[0].grid(True)

    f, R = espectro(r, fs)
    ax[1].plot(f / 1000, 20 * np.log10(R / R.max() + 1e-12), linewidth=0.5)
    ax[1].set_title("Espectro de la señal AM recibida")
    ax[1].set_xlabel("Frecuencia (kHz)")
    ax[1].set_ylabel("Amplitud (dB)")
    ax[1].set_ylim(-100, 5)
    ax[1].grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    # Escuchar cada etapa
    reproducir(m, "mensaje original")
    reproducir(r, "señal AM en el canal (portadora de 10 kHz + ruido)", volumen=0.3)
    reproducir(m_env, "recuperado con detector de envolvente")
    reproducir(m_coh, "recuperado con receptor coherente")
    