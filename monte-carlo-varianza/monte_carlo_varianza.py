"""
Comparación de técnicas de reducción de varianza en simulación Monte Carlo
para la valoración de opciones (europea y asiática).
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

S0, K, R_LIBRE, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0
DESC = np.exp(-R_LIBRE * T)      # factor de descuento e^{-rT}
M = 12                           # fechas de observación de la opción asiática
DT = T / M

def black_scholes_call():
    """Precio exacto de la call europea (fórmula de Black-Scholes)."""
    d1 = (np.log(S0 / K) + (R_LIBRE + SIGMA**2 / 2) * T) / (SIGMA * np.sqrt(T))
    d2 = d1 - SIGMA * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * DESC * norm.cdf(d2)


def asiatica_geometrica_exacta():
    """Precio exacto de la asiática GEOMÉTRICA (se usa como variable de control)."""
    mu_g = np.log(S0) + (R_LIBRE - SIGMA**2 / 2) * DT * (M + 1) / 2
    var_g = SIGMA**2 * DT * (M + 1) * (2 * M + 1) / (6 * M)
    s_g = np.sqrt(var_g)
    d1 = (mu_g - np.log(K) + var_g) / s_g
    d2 = d1 - s_g
    return DESC * (np.exp(mu_g + var_g / 2) * norm.cdf(d1) - K * norm.cdf(d2))

def precio_terminal(z):
    """S_T a partir de variables normales estándar z."""
    return S0 * np.exp((R_LIBRE - SIGMA**2 / 2) * T + SIGMA * np.sqrt(T) * z)


def pago_call(s):
    """Pago descontado de la call: e^{-rT} * max(S_T - K, 0)."""
    return DESC * np.maximum(s - K, 0.0)


def mc_estandar(n, rng):
    """1) Monte Carlo simple: n normales independientes."""
    return pago_call(precio_terminal(rng.standard_normal(n))).mean()


def mc_antitetica(n, rng):
    """2) Antitéticas: n/2 pares (Z, -Z); se promedian los dos pagos de cada par."""
    z = rng.standard_normal(n // 2)
    return (0.5 * (pago_call(precio_terminal(z)) + pago_call(precio_terminal(-z)))).mean()


def mc_control(n, rng):
    """3) Variable de control: X = S_T descontado, con E[X] = S0 conocido.
    Estimador: media de (Y - beta*(X - E[X])), beta estimado con la muestra."""
    s = precio_terminal(rng.standard_normal(n))
    y = pago_call(s)
    x = DESC * s
    beta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    return (y - beta * (x - S0)).mean()


def mc_estratificado(n, rng):
    """4) Estratificado: (0,1) se divide en n estratos y se toma una uniforme
    en cada uno; luego se transforma con la inversa de la normal."""
    u = (np.arange(n) + rng.random(n)) / n
    return pago_call(precio_terminal(norm.ppf(u))).mean()


def mc_sobol(n, rng):
    """5) Cuasi-Monte Carlo: secuencia de Sobol aleatorizada (Owen scrambling).
    n debe ser potencia de 2."""
    m = int(np.log2(n))
    u = qmc.Sobol(1, scramble=True, seed=rng).random_base2(m)[:, 0]
    u = np.clip(u, 1e-12, 1 - 1e-12)  # evita ppf(0) o ppf(1) = infinito
    return pago_call(precio_terminal(norm.ppf(u))).mean()


TECNICAS_EUR = {
    "Estándar": mc_estandar,
    "Antitética": mc_antitetica,
    "Var. control": mc_control,
    "Estratificado": mc_estratificado,
    "Sobol (QMC)": mc_sobol,
}

def trayectorias(z):
    """Trayectorias de S en las M fechas a partir de normales z de forma (n, M)."""
    incrementos = (R_LIBRE - SIGMA**2 / 2) * DT + SIGMA * np.sqrt(DT) * z
    return S0 * np.exp(np.cumsum(incrementos, axis=1))


def pago_aritmetico(p):
    return DESC * np.maximum(p.mean(axis=1) - K, 0.0)


def pago_geometrico(p):
    return DESC * np.maximum(np.exp(np.log(p).mean(axis=1)) - K, 0.0)


def asi_estandar(n, rng):
    return pago_aritmetico(trayectorias(rng.standard_normal((n, M)))).mean()


def asi_antitetica(n, rng):
    z = rng.standard_normal((n // 2, M))
    return (0.5 * (pago_aritmetico(trayectorias(z)) + pago_aritmetico(trayectorias(-z)))).mean()


GEO_EXACTA = asiatica_geometrica_exacta()


def asi_control(n, rng):
    """Control: la asiática geométrica, muy correlacionada con la aritmética."""
    p = trayectorias(rng.standard_normal((n, M)))
    a, g = pago_aritmetico(p), pago_geometrico(p)
    beta = np.cov(a, g)[0, 1] / g.var(ddof=1)
    return (a - beta * (g - GEO_EXACTA)).mean()


def asi_sobol(n, rng):
    u = qmc.Sobol(M, scramble=True, seed=rng).random_base2(int(np.log2(n)))
    z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))
    return pago_aritmetico(trayectorias(z)).mean()


TECNICAS_ASI = {
    "Estándar": asi_estandar,
    "Antitética": asi_antitetica,
    "Var. control": asi_control,
    "Sobol (QMC)": asi_sobol,
}


def evaluar(tecnicas, ns, referencia, repeticiones, semilla):
    """Repite cada técnica `repeticiones` veces para cada N y calcula métricas."""
    filas = []
    for nombre, f in tecnicas.items():
        for n in ns:
            rng = np.random.default_rng(semilla)
            t0 = time.perf_counter()
            est = np.array([f(n, rng) for _ in range(repeticiones)])
            tiempo = (time.perf_counter() - t0) / repeticiones   # s por estimación
            err = est - referencia
            var = est.var(ddof=1)
            filas.append(dict(
                tecnica=nombre, N=n,
                rmse=np.sqrt((err**2).mean()),     # error cuadrático medio
                mae=np.abs(err).mean(),            # error absoluto medio
                sesgo=err.mean(),
                varianza=var,
                tiempo_s=tiempo,
                eficiencia=var * tiempo,           # menor = mejor
            ))
            print(f"  {nombre:14s} N={n:7d}  RMSE={filas[-1]['rmse']:.3e}", flush=True)
    df = pd.DataFrame(filas)
    base = df[df.tecnica == "Estándar"].set_index("N")
    df["reduccion_varianza"] = df.apply(lambda r: base.loc[r.N, "varianza"] / r.varianza, axis=1)
    df["ganancia_eficiencia"] = df.apply(lambda r: base.loc[r.N, "eficiencia"] / r.eficiencia, axis=1)
    return df


def grafico(df, columna, etiqueta_y, archivo, linea_ref=False):
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    marcas = ["o", "s", "^", "D", "v"]
    for marca, (nombre, g) in zip(marcas, df.groupby("tecnica", sort=False)):
        ax.loglog(g.N, g[columna], marker=marca, ms=4, lw=1, label=nombre)
    if linea_ref:  # recta de pendiente -1/2 que parte del punto inicial del estándar
        g = df[df.tecnica == "Estándar"]
        ax.loglog(g.N, g[columna].iloc[0] * (g.N.iloc[0] / g.N) ** 0.5, "k--", lw=0.8, label=r"$N^{-1/2}$")
    ax.set_xlabel("Número de simulaciones $N$")
    ax.set_ylabel(etiqueta_y)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(archivo, dpi=300)
    plt.close(fig)


def pendientes(df, nombre_tabla):
    """Orden de convergencia = pendiente de log(RMSE) vs log(N)."""
    print(f"\nPendientes de convergencia ({nombre_tabla}):")
    for nombre, g in df.groupby("tecnica", sort=False):
        p = np.polyfit(np.log(g.N), np.log(g.rmse), 1)[0]
        print(f"  {nombre:14s} {p:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="prueba rápida (menos N y repeticiones)")
    args = ap.parse_args()

    if args.rapido:
        ns_eur, ns_asi, reps, n_ref, reps_ref = [2**k for k in (7, 10, 13)], [2**k for k in (7, 10)], 20, 2**16, 4
    else:
        ns_eur = [2**k for k in (7, 10, 13, 16, 19)]   # 128 ... 524 288
        ns_asi = [2**k for k in (7, 10, 13, 16)]       # 128 ... 65 536
        reps, n_ref, reps_ref = 100, 2**20, 8

    os.makedirs("resultados", exist_ok=True)

    bs = black_scholes_call()
    print(f"Precio exacto Black-Scholes (call europea): {bs:.6f}")

    print("\n=== Opción europea ===")
    df_eur = evaluar(TECNICAS_EUR, ns_eur, bs, reps, semilla=12345)
    df_eur.to_csv("resultados/europea.csv", index=False)
    pendientes(df_eur, "europea")

    # Referencia de la asiática: no hay fórmula cerrada, así que se estima con
    # muchas simulaciones y variable de control (error estándar ~5e-5).
    rng = np.random.default_rng(999)
    refs = [asi_control(n_ref, rng) for _ in range(reps_ref)]
    ref_asi = float(np.mean(refs))
    print(f"\nReferencia asiática aritmética: {ref_asi:.5f} (error est. {np.std(refs)/np.sqrt(reps_ref):.1e})")

    print("\n=== Opción asiática ===")
    df_asi = evaluar(TECNICAS_ASI, ns_asi, ref_asi, reps, semilla=777)
    df_asi.to_csv("resultados/asiatica.csv", index=False)
    pendientes(df_asi, "asiática")

    grafico(df_eur, "rmse", "RMSE (USD)", "resultados/fig1_rmse_europea.png", linea_ref=True)
    grafico(df_eur, "eficiencia", "Eficiencia (varianza × tiempo)", "resultados/fig2_eficiencia_europea.png")
    grafico(df_asi, "rmse", "RMSE (USD)", "resultados/fig3_rmse_asiatica.png", linea_ref=True)
    print("\nListo. Revisa la carpeta resultados/")


if __name__ == "__main__":
    main()
