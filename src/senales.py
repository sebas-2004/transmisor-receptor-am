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