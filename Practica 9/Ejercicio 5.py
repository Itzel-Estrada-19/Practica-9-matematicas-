import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, exp, sin, cos, ln, lambdify
from scipy.integrate import solve_ivp

print("Ejercicios 5.5")
print("Ejercicio 1")

# Método de Runge-Kutta-Fehlberg personalizado
def rkf45(f, t0, y0, tf, tol, hmax, hmin):
    ts = [t0]
    ys = [y0]
    h = hmax
    while t0 < tf:
        if t0 + h > tf:
            h = tf - t0
        k1 = h * f(t0, y0)
        k2 = h * f(t0 + h/4, y0 + k1/4)
        k3 = h * f(t0 + 3*h/8, y0 + 3*k1/32 + 9*k2/32)
        k4 = h * f(t0 + 12*h/13, y0 + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
        k5 = h * f(t0 + h, y0 + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
        k6 = h * f(t0 + h/2, y0 - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)

        y_next = y0 + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
        z_next = y0 + 16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55

        err = abs(y_next - z_next)
        if err < tol:
            t0 += h
            y0 = y_next
            ts.append(t0)
            ys.append(y0)

        q = 0.84 * (tol / err)**0.25 if err != 0 else 2
        h = max(hmin, min(hmax, q*h))
    return np.array(ts), np.array(ys)

# Obtener ejercicios simbólicos con soluciones exactas
def obtener_ejercicios():
    t = symbols('t')
    ejercicios = {}

    # a)
    y_a = (1/5)*t*exp(3*t) - (1/25)*exp(3*t) + (1/25)*exp(-2*t)
    ejercicios['a'] = {
        "f": lambda t, y: y / t - (y / t)**2 if t != 0 else 0,
        "t0": 0.05, "tf": 1, "y0": 0,
        "sol_real": lambdify(t, y_a, modules=["numpy"])
    }

    # b)
    y_b = t + 1 / (1 - t)
    ejercicios['b'] = {
        "f": lambda t, y: 1 + (t - y)**2,
        "t0": 2, "tf": 3, "y0": 1,
        "sol_real": lambdify(t, y_b, modules=["numpy"])
    }

    # c)
    y_c = t * ln(t) + 2 * t
    ejercicios['c'] = {
        "f": lambda t, y: 1 + y / t if t != 0 else 0,
        "t0": 1, "tf": 2, "y0": 2,
        "sol_real": lambdify(t, y_c, modules=["numpy"])
    }

    # d)
    y_d = (1/2)*sin(2*t) - (1/3)*cos(3*t) + 4/3
    ejercicios['d'] = {
        "f": lambda t, y: np.cos(2*t) + np.sin(3*t),
        "t0": 0, "tf": 1, "y0": 1,
        "sol_real": lambdify(t, y_d, modules=["numpy"])
    }

    return ejercicios

# Parámetros generales
TOL = 1e-4
HMAX = 0.25
HMIN = 0.05

# Ejecutar ejercicios 5.5 (Problema 1)
ejercicios = obtener_ejercicios()
for inc, datos in ejercicios.items():
    print(f"\n--- Ejercicio 5.5 inciso {inc} ---")

    t_vals, y_aprox = rkf45(datos['f'], datos['t0'], datos['y0'], datos['tf'], TOL, HMAX, HMIN)
    y_exact = datos['sol_real'](t_vals)

    # Imprimir tabla
    print(f"{'t':>8} | {'y_aprox':>12} | {'y_exact':>12} | {'Error absoluto':>15}")
    print("-" * 55)
    for t_i, ya, ye in zip(t_vals, y_aprox, y_exact):
        error = abs(ya - ye)
        print(f"{t_i:8.4f} | {ya:12.8f} | {ye:12.8f} | {error:15.2e}")

    # Graficar
    plt.figure()
    plt.plot(t_vals, y_aprox, 'bo-', label='RKF45')
    plt.plot(t_vals, y_exact, 'r--', label='Sol. Exacta')
    plt.title(f'Ejercicio 5.5 inciso {inc}')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------
# PROBLEMAS 2 y 3 CON solve_ivp
# ------------------------

# Función auxiliar general para imprimir resultados
def resolver_inciso(nombre, f, t_span, y0, h_max, h_min, tol, sol_exacta=None):
    print(f"\nInciso {nombre}:")
    sol = solve_ivp(f, t_span, [y0], method='RK45', rtol=tol, atol=tol, max_step=h_max)
    for t, y in zip(sol.t, sol.y[0]):
        print(f"t = {t:.5f}, y ≈ {y:.5f}")
    if sol_exacta:
        t_sym = symbols('t')
        y_real_func = lambdify(t_sym, sol_exacta, 'numpy')
        y_real_vals = y_real_func(sol.t)
        errores = np.abs(sol.y[0] - y_real_vals)
        print("\nComparación con la solución exacta:")
        for t, y_aprox, y_real, err in zip(sol.t, sol.y[0], y_real_vals, errores):
            print(f"t = {t:.5f}, Aprox = {y_aprox:.5f}, Real = {y_real:.5f}, Error = {err:.2e}")

# Parámetros específicos
TOL2 = 1e-4
TOL3 = 1e-6

# --- PROBLEMA 2 ---
resolver_inciso("2a", lambda t, y: (y/t)**2 + y/t, (1, 1.2), 1, 0.005, 0.02, TOL2)

# --- PROBLEMA 3 ---
resolver_inciso("3a", lambda t, y: 1 + (t - y)**2, (2, 3), 1, 0.025, 0.01, TOL2)
resolver_inciso("3b", lambda t, y: 1 + y/t if t != 0 else 0, (1, 2), 2, 0.025, 0.01, TOL3)
