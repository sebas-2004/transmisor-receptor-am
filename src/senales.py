import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
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


def detector_envolvente(r, fs, fcorte, mu, Ac=1.0, sobremuestreo=1):
    """Receptor AM por detección de envolvente.
    0) (Opcional) Sobremuestreo x L antes de rectificar: el valor absoluto es no lineal
       y genera armónicos de la portadora (2fc, 4fc, ...) que pueden superar fs/2 y
       producir aliasing dentro de la banda del mensaje. Con fs*L caben sin doblarse.
    1) Rectificación de onda completa: |r(t)|
    2) Pasabajos (x pi/2 porque el promedio de |cos| es 2/pi)
    3) Volver a la fs original, quitar el nivel DC y reescalar: m = (env/Ac - 1) / mu"""
    L = sobremuestreo
    if L > 1:
        r = resample_poly(r, L, 1)            # interpolar: fs -> fs*L
    rectificada = np.abs(r)
    env = (np.pi / 2) * filtro_pasabajos(rectificada, fcorte, fs * L)
    if L > 1:
        env = resample_poly(env, 1, L)        # volver a la fs original
        rectificada = rectificada[::L]        # solo para poder graficarla
    m_rec = (env / Ac - 1) / mu
    return m_rec, rectificada, env


def receptor_coherente(r, t, fc, fs, fcorte, mu, Ac=1.0):
    """Receptor AM coherente (síncrono).
    1) Multiplica por una copia local de la portadora: r(t) * 2cos(2*pi*fc*t)
       -> queda Ac[1 + mu*m(t)] en banda base + una copia en 2fc
    2) Pasabajos para eliminar la componente en 2fc
    3) Quitar el nivel DC y reescalar: m = (y/Ac - 1) / mu
    Supone sincronización perfecta (misma frecuencia y fase que el transmisor)."""
    mezcla = r * 2 * np.cos(2 * np.pi * fc * t)
    banda_base = filtro_pasabajos(mezcla, fcorte, fs)
    return (banda_base / Ac - 1) / mu


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


def espectro(x, fs):
    """Espectro de amplitud de un solo lado usando la FFT.
    Devuelve las frecuencias f (Hz, de 0 a fs/2) y la amplitud |X[k]|*2/N,
    escalada para que un cos de amplitud A produzca un pico de altura A."""
    N = len(x)
    X = np.fft.rfft(x)                     # FFT de una señal real (solo f >= 0)
    f = np.fft.rfftfreq(N, d=1 / fs)       # f_k = k * fs / N
    mag = np.abs(X) / N
    mag[1:] *= 2                           # sumar la parte negativa del espectro (simétrica)
    return f, mag