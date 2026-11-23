# Reducción de varianza en simulación Monte Carlo

Código del artículo *Análisis comparativo de técnicas de reducción de varianza en simulación Monte Carlo para la valoración de opciones financieras* (Universidad Católica de Santa María, Arequipa).

Compara cinco técnicas en una call europea (con precio exacto de Black-Scholes) y cuatro en una opción asiática aritmética de 12 fechas:

| Técnica | Europea | Asiática |
|---|---|---|
| Monte Carlo estándar | ✔ | ✔ |
| Variables antitéticas | ✔ | ✔ |
| Variable de control | ✔ | ✔ (asiática geométrica) |
| Muestreo estratificado | ✔ | – |
| Cuasi-Monte Carlo (Sobol aleatorizado) | ✔ | ✔ |

Parámetros: S0 = 100, K = 100, r = 5 %, σ = 20 %, T = 1 año. Cada configuración se repite 100 veces.

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python monte_carlo_varianza.py --rapido   # prueba rápida (menos de 1 minuto)
python monte_carlo_varianza.py            # experimento completo del artículo
```

Los resultados se guardan en `resultados/` (tablas `.csv` y tres gráficos `.png`). Las semillas son fijas, así que los errores (RMSE) coinciden con los del artículo; los tiempos dependen del equipo.

## Métricas

- **RMSE**: error cuadrático medio respecto al valor de referencia.
- **Varianza** del estimador entre las 100 repeticiones.
- **Eficiencia** = varianza × tiempo por estimación (menor es mejor).
- **Orden de convergencia**: pendiente de log(RMSE) frente a log(N).

## Autores

Maria Fernanda Linares Limache, Leonardo Victor Díaz Porras, Marcelo Rodrigo Moscoso Flores.
