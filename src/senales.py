import numpy as np


def vector_tiempo(duracion, fs):
    """Crea el eje de tiempo t (en segundos) con N = duracion * fs muestras."""
    N = int(duracion * fs)
    return np.arange(N) / fs


def generar_tono(f, duracion, fs, amplitud=1.0):
    """Genera un tono m(t) = A * cos(2*pi*f*t).
    Devuelve el vector de tiempo t y la señal m."""
    t = vector_tiempo(duracion, fs)
    m = amplitud * np.cos(2 * np.pi * f * t)
    return t, m
def generar_portadora(fc, t, Ac=1.0):
    """Genera la portadora c(t) = Ac * cos(2*pi*fc*t).
    Usa el MISMO vector de tiempo t del mensaje, para que ambas
    señales estén muestreadas en los mismos instantes."""
    return Ac * np.cos(2 * np.pi * fc * t)
def modular_am(m, c, mu):
    """Modulación AM de doble banda lateral con portadora (DSB-LC).
    s(t) = [1 + mu*m(t)] * c(t), con m(t) normalizado entre -1 y 1."""
    return (1 + mu * m) * c


def envolvente_am(m, mu, Ac=1.0):
    """Envolvente teórica de la señal AM: Ac * [1 + mu*m(t)]."""
    return Ac * (1 + mu * m)

def potencia(x):
    """Potencia promedio de una señal discreta: P = (1/N) * sum(x[n]^2)."""
    return np.mean(x ** 2)


def agregar_ruido(s, snr_db, semilla=None):
    """Agrega ruido blanco gaussiano (AWGN) a la señal s con una SNR dada en dB.
    Devuelve la señal ruidosa y el ruido generado."""
    rng = np.random.default_rng(semilla)
    Ps = potencia(s)
    Pn = Ps / (10 ** (snr_db / 10))                     # potencia de ruido necesaria
    ruido = np.sqrt(Pn) * rng.standard_normal(len(s))   # desviación estándar = sqrt(Pn)
    return s + ruido, ruido
    