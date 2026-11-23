# Reducción de varianza en simulación Monte Carlo

Este proyecto contiene el código utilizado para comparar diferentes técnicas de reducción de varianza en la simulación Monte Carlo para opciones financieras.

Se trabajó con dos tipos de opciones: europea y asiática.

## Técnicas utilizadas

- Monte Carlo estándar.
- Variables antitéticas.
- Variable de control.
- Muestreo estratificado.
- Cuasi-Monte Carlo (Sobol).

## Cómo ejecutar el proyecto

Instalar las dependencias:

    pip install -r requirements.txt

Ejecutar el programa:

    python monte_carlo_varianza.py

Para una prueba rápida:

    python monte_carlo_varianza.py --rapido

Los resultados se guardan en la carpeta `resultados`.
