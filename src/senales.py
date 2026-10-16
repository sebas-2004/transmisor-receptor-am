import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io import wavfile


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


def filtro_pasabajos(x, fcorte, fs, orden=4):
    """Filtro pasabajos Butterworth. filtfilt lo aplica hacia adelante y hacia
    atrás para no desfasar (retrasar) la señal."""
    b, a = butter(orden, fcorte, btype="low", fs=fs)
    return filtfilt(b, a, x)


def detector_envolvente(r, fs, fcorte, mu, Ac=1.0):
    """Receptor AM por detección de envolvente.
    1) Rectificación de onda completa: |r(t)|
    2) Pasabajos para eliminar la portadora (x pi/2 porque el promedio de |cos| es 2/pi)
    3) Quitar el nivel DC y reescalar: m = (env/Ac - 1) / mu
    Devuelve el mensaje recuperado, la señal rectificada y la envolvente."""
    rectificada = np.abs(r)
    env = (np.pi / 2) * filtro_pasabajos(rectificada, fcorte, fs)
    m_rec = (env / Ac - 1) / mu
    return m_rec, rectificada, env


def cargar_audio(ruta):
    """Carga un .wav y lo deja listo como mensaje:
    convierte a decimal, pasa a mono, quita el DC y normaliza a [-1, 1].
    Devuelve fs y la señal."""
    fs, x = wavfile.read(ruta)
    x = x.astype(np.float64)
    if x.ndim == 2:              # estéreo -> mono (promedio de canales)
        x = x.mean(axis=1)
    x = x - np.mean(x)           # quitar el nivel DC
    x = x / np.max(np.abs(x))    # normalizar a [-1, 1]
    return fs, x


def guardar_audio(ruta, x, fs):
    """Guarda una señal como .wav de 16 bits (normalizada para no saturar)."""
    x = x / np.max(np.abs(x))
    wavfile.write(ruta, fs, (x * 32767).astype(np.int16))