import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# =====================================================================
# 1. DEFINICIÓN DE LA SEÑAL DE ENTRADA Y PORTADORA
# =====================================================================
fs = 10000.0          # Frecuencia de muestreo (Hz) -> cumple Nyquist (>= 2 * f_max)
duracion = 0.05       # Duración de la señal en segundos (50 ms)
t = np.arange(0, duracion, 1.0 / fs)

fm = 100.0            # Frecuencia del mensaje (baja frecuencia, 100 Hz)
fc = 1500.0           # Frecuencia de la portadora (alta frecuencia, 1500 Hz)
Ac = 1.0              # Amplitud de la portadora
indice_mod = 0.8      # Índice de modulación (0 < mu <= 1 evita sobremodulación)

# Señal mensaje de baja frecuencia (combinación de tono principal y armónico suave)
mensaje = np.cos(2 * np.pi * fm * t) + 0.25 * np.sin(2 * np.pi * (2 * fm) * t)
mensaje_norm = mensaje / np.max(np.abs(mensaje))  # Normalizada en [-1, 1]

# Portadora de alta frecuencia
portadora = Ac * np.cos(2 * np.pi * fc * t)


# =====================================================================
# 2. IMPLEMENTACIÓN DE LA MODULACIÓN AM
# =====================================================================
def modular_am(msg_norm, carrier, mu):
    """Genera señal AM convencional: s(t) = (1 + mu * m_norm(t)) * c(t)."""
    return (1.0 + mu * msg_norm) * carrier


def calcular_espectro(senal, fs_hz):
    """Calcula frecuencias positivas y magnitud normalizada mediante FFT."""
    n = len(senal)
    ventana = np.hanning(n)
    espectro = np.abs(np.fft.rfft(senal * ventana)) / (np.sum(ventana) / 2.0)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    return freqs, espectro


senal_am = modular_am(mensaje_norm, portadora, indice_mod)


# =====================================================================
# 3. ESCENARIOS DE CANAL: RUIDO, ATENUACIÓN Y DISTORSIÓN
# =====================================================================
rng = np.random.default_rng(seed=42)

# Escenario A: Ruido blanco gaussiano aditivo (AWGN)
sigma_ruido = 0.25
ruido = rng.normal(0.0, sigma_ruido, size=len(t))
senal_ruido = senal_am + ruido

# Escenario B: Atenuación del canal (pérdida de amplitud a lo largo del trayecto)
factor_atenuacion = 0.35  # Reduce la amplitud al 35%
senal_atenuada = factor_atenuacion * senal_am

# Escenario C: Distorsión no lineal (clipping / saturación del amplificador)
umbral_clip = 0.85
senal_distorsion_clip = np.clip(senal_am, -umbral_clip, umbral_clip)

# Escenario D: Sobremulación (mu = 1.4 > 1) -> distorsión de envolvente
senal_sobremodulada = modular_am(mensaje_norm, portadora, mu=1.4)


# =====================================================================
# 4. VISUALIZACIÓN EN TIEMPO Y FRECUENCIA
# =====================================================================
# Figura 1: Mensaje, portadora, señal AM y espectro en frecuencia
fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

axs[0].plot(t * 1000, mensaje_norm, color="tab:blue", label="Mensaje normalizado m(t)")
axs[0].set_title("Señal de Entrada (Baja Frecuencia)")
axs[0].set_xlabel("Tiempo (ms)")
axs[0].set_ylabel("Amplitud")
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc="upper right")

envolvente = Ac * (1.0 + indice_mod * mensaje_norm)
axs[1].plot(t * 1000, senal_am, color="tab:orange", linewidth=1.0, label="Señal AM s(t)")
axs[1].plot(t * 1000, envolvente, "--", color="black", linewidth=1.2, label="Envolvente superior")
axs[1].plot(t * 1000, -envolvente, "--", color="black", linewidth=1.2)
axs[1].set_title(f"Señal Modulada en Amplitud (fc = {fc} Hz, mu = {indice_mod})")
axs[1].set_xlabel("Tiempo (ms)")
axs[1].set_ylabel("Amplitud")
axs[1].grid(True, alpha=0.3)
axs[1].legend(loc="upper right")

freqs, mag_am = calcular_espectro(senal_am, fs)
axs[2].plot(freqs, mag_am, color="tab:green", label="Espectro AM")
axs[2].set_xlim(0, 3500)
axs[2].set_title("Análisis en Frecuencia (FFT) - Portadora y Bandas Laterales")
axs[2].set_xlabel("Frecuencia (Hz)")
axs[2].set_ylabel("Magnitud")
axs[2].grid(True, alpha=0.3)
axs[2].legend(loc="upper right")

plt.savefig("am_base_tiempo_frecuencia.png", dpi=150)

# Figura 2: Comparación de escenarios (Ruido, Atenuación, Distorsión)
fig2, axs2 = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

axs2[0, 0].plot(t * 1000, senal_ruido, color="tab:red", linewidth=0.9)
axs2[0, 0].set_title(f"Ruido AWGN (sigma = {sigma_ruido})")
axs2[0, 0].set_xlabel("Tiempo (ms)")
axs2[0, 0].set_ylabel("Amplitud")
axs2[0, 0].grid(True, alpha=0.3)

axs2[0, 1].plot(t * 1000, senal_atenuada, color="tab:purple", linewidth=1.0)
axs2[0, 1].set_ylim(-1.6, 1.6)
axs2[0, 1].set_title(f"Atenuación (factor = {factor_atenuacion})")
axs2[0, 1].set_xlabel("Tiempo (ms)")
axs2[0, 1].set_ylabel("Amplitud")
axs2[0, 1].grid(True, alpha=0.3)

axs2[1, 0].plot(t * 1000, senal_distorsion_clip, color="tab:brown", linewidth=1.0)
axs2[1, 0].set_title(f"Distorsión por Clipping (umbral = {umbral_clip})")
axs2[1, 0].set_xlabel("Tiempo (ms)")
axs2[1, 0].set_ylabel("Amplitud")
axs2[1, 0].grid(True, alpha=0.3)

axs2[1, 1].plot(t * 1000, senal_sobremodulada, color="tab:cyan", linewidth=1.0)
axs2[1, 1].set_title("Sobremulación (mu = 1.4 > 1)")
axs2[1, 1].set_xlabel("Tiempo (ms)")
axs2[1, 1].set_ylabel("Amplitud")
axs2[1, 1].grid(True, alpha=0.3)

plt.savefig("am_escenarios_canal.png", dpi=150)
plt.show()
