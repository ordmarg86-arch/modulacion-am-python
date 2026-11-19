# Implementación y análisis de un sistema de Modulación en Amplitud (AM)

Proyecto desarrollado en **Python 3.13** utilizando **NumPy** y **Matplotlib** para simular la modulación en amplitud (AM) convencional, analizar su comportamiento en los dominios del tiempo y la frecuencia (FFT) y evaluar el desempeño de la señal bajo distintos escenarios de canal.

## Objetivos
- Generar una señal mensaje de baja frecuencia y modularla sobre una portadora de alta frecuencia.
- Analizar la envolvente temporal y el espectro de frecuencias (portadora y bandas laterales).
- Evaluar el impacto del **ruido AWGN**, la **atenuación**, la **distorsión por *clipping*** y la **sobremodulación** ($\mu > 1$).

## Requisitos
- Python 3.13
- `numpy`
- `matplotlib`

Instalación de dependencias:
```bash
python -m pip install numpy matplotlib
```

## Ejecución
```bash
python python_modulacion_am.py
```

Al ejecutar el script se generan dos figuras en el directorio de trabajo:
1. `am_base_tiempo_frecuencia.png`: señal de entrada de baja frecuencia ($100\text{ Hz}$), señal modulada en AM ($f_c = 1500\text{ Hz}$, $\mu = 0.8$) con su envolvente y espectro de frecuencias mediante FFT.
2. `am_escenarios_canal.png`: comparación temporal ante ruido AWGN ($\sigma = 0.25$), atenuación ($\text{factor} = 0.35$), *clipping* ($\text{umbral} = 0.85$) y sobremodulación ($\mu = 1.4$).

## Estructura del análisis
- **Señal de entrada y portadora:** definición del vector de tiempo ($f_s = 10\text{ kHz}$, $50\text{ ms}$), mensaje normalizado y portadora.
- **Modulación AM:** aplicación de $s(t) = A_c [1 + \mu \, \tilde{m}(t)] \cos(2\pi f_c t)$.
- **Análisis en tiempo y frecuencia:** inspección de la envolvente y cálculo de espectro con ventana de Hanning y `rfft`.
- **Escenarios de comunicación:** contraste entre degradación de amplitud/SNR y distorsión no lineal o de envolvente.
