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