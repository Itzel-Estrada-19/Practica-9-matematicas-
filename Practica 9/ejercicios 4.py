import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def metodo_euler_modificado(f_sym, y_exact_expr, t0, y0, h, t_final, titulo):
    t, y = sp.symbols('t y')
    f_numeric = sp.lambdify((t, y), f_sym, modules='numpy')
    y_exact_func = sp.lambdify(t, y_exact_expr, modules='numpy')

    ts = np.arange(t0, t_final + h, h)
    ys = np.zeros(len(ts))
    ys[0] = y0

    for i in range(1, len(ts)):
        t_i = ts[i - 1]
        y_i = ys[i - 1]
        y_predictor = y_i + h * f_numeric(t_i, y_i)
        y_corrector = y_i + h / 2 * (f_numeric(t_i, y_i) + f_numeric(t_i + h, y_predictor))
        ys[i] = y_corrector

    ys_exact = y_exact_func(ts)

    print(f"===== {titulo} =====")
    print(f"{'t':>6} | {'y_aprox':>10} | {'y_real':>10} | {'error':>10}")
    print("-" * 42)
    for i in range(len(ts)):
        print(f"{ts[i]:6.2f} | {ys[i]:10.6f} | {ys_exact[i]:10.6f} | {abs(ys[i] - ys_exact[i]):10.2e}")

    # Gráfica
    plt.plot(ts, ys, 'o--', label='Euler Modificado')
    plt.plot(ts, ys_exact, 'r-', label='Solución Exacta')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.show()


# === Inciso a ===
t, y = sp.symbols('t y')
f_a = t * sp.exp(3 * t) - 2 * y
y_real_a = (1 / 5) * t * sp.exp(3 * t) - (1 / 25) * sp.exp(3 * t) + (1 / 25) * sp.exp(-2 * t)
metodo_euler_modificado(f_a, y_real_a, t0=0, y0=0, h=0.5, t_final=1.0, titulo="Ejercicio 5.4 - Inciso a")

# === Inciso b ===
f_b = 1 + (t - y) ** 2
y_real_b = t + 1 / (1 - t)
metodo_euler_modificado(f_b, y_real_b, t0=2, y0=1, h=0.5, t_final=3.0, titulo="Ejercicio 5.4 - Inciso b")

# === Inciso c ===
f_c = 1 + y / t
y_real_c = t * sp.ln(t) + 2 * t
metodo_euler_modificado(f_c, y_real_c, t0=1, y0=2, h=0.25, t_final=2.0, titulo="Ejercicio 5.4 - Inciso c")

# === Inciso d ===
f_d = sp.cos(2 * t) + sp.sin(3 * t)
y_real_d = (1 / 2) * sp.sin(2 * t) - (1 / 3) * sp.cos(3 * t) + (4 / 3)
metodo_euler_modificado(f_d, y_real_d, t0=0, y0=1, h=0.25, t_final=1.0, titulo="Ejercicio 5.4 - Inciso d")

# === Ejercicio 15 ===
print("\n===== Ejercicio 15 =====")

# Parámetros
t, x = sp.symbols('t x')
k = 6.22e-19
n1 = n2 = 2e3
n3 = 3e3

# EDO: dx/dt = f(t, x)
f15_expr = k * (n1 - x / 2) ** 2 * (n2 - x / 2) ** 2 * (n3 - 3 * x / 4) ** 3
f15_numeric = sp.lambdify((t, x), f15_expr, modules='numpy')

# Condiciones iniciales y parámetros numéricos
t0 = 0
x0 = 0
t_final = 0.2
h = 0.01

ts = np.arange(t0, t_final + h, h)
xs = np.zeros(len(ts))
xs[0] = x0

for i in range(1, len(ts)):
    t_i = ts[i - 1]
    x_i = xs[i - 1]
    x_predict = x_i + h * f15_numeric(t_i, x_i)
    x_correct = x_i + h / 2 * (f15_numeric(t_i, x_i) + f15_numeric(t_i + h, x_predict))
    xs[i] = x_correct

print(f"{'t':>6} | {'x(t) KOH formadas':>20}")
print("-" * 30)
for i in range(len(ts)):
    if abs(ts[i] - 0.2) < 1e-6 or ts[i] == 0.0:
        print(f"{ts[i]:6.2f} | {xs[i]:20.6f}")

# Gráfica
plt.plot(ts, xs, 'b--', label='x(t) - Unidades de KOH')
plt.xlabel('t (s)')
plt.ylabel('x(t) - Cantidad de KOH')
plt.title('Ejercicio 15 - Formación de KOH vs Tiempo')
plt.grid(True)
plt.legend()
plt.show()
