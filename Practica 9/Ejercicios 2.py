import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

print("Ejercicios 5.2")

print("\nEjercicio 1")
print("Inciso a")
x0 = 0.0
y0 = 0.0
xn = 1.0
h = 0.5
N = int((xn - x0) / h)
t = x0
y = y0
print("Valores de inicio:")
print(f"x = {t:.2f}, y = {y:.2f}")
for i in range(1, N + 1):
    y_a = t*sp.exp(3*t)-2*y
    y = y + h * y_a
    t = t + h
    print(f"Iteración {i}")
    print(f"x = {t:.4f}, y = {y:.4f}")

print("Inciso b")
x0 = 2.0
y0 = 1.0
xn = 3.0
h = 0.5
N = int((xn - x0) / h)
t = x0
y = y0
print("Valores de inicio:")
print(f"x = {t:.2f}, y = {y:.2f}")
for i in range(1, N + 1):
    y_b = 1 + (t-1)**2
    y = y + h * y_b
    t = t + h
    print(f"Iteración {i}")
    print(f"x = {t:.4f}, y = {y:.4f}")

print("Inciso c")
x0 = 1.0
y0 = 2.0
xn = 2.0
h = 0.25
N = int((xn - x0) / h)
t = x0
y = y0
print("Valores de inicio:")
print(f"x = {t:.2f}, y = {y:.2f}")
for i in range(1, N + 1):
    y_a = 1 + y/t
    y = y + h * y_a
    t = t + h
    print(f"Iteración {i}")
    print(f"x = {t:.4f}, y = {y:.4f}")

print("Inciso d")
x0 = 0.0
y0 = 1.0
xn = 1.0
h = 0.25
N = int((xn - x0) / h)
t = x0
y = y0
print("Valores de inicio:")
print(f"x = {t:.2f}, y = {y:.2f}")
for i in range(1, N + 1):
    y_a = sp.cos(2*t) + sp.sin(3*t)
    y = y + h * y_a
    t = t + h
    print(f"Iteración {i}")
    print(f"x = {t:.4f}, y = {y:.4f}")

print("\nEjercicio 11")
print("Inciso a")

# Variables simbólicas
t = sp.Symbol('t', real=True)
x = sp.Function('x')(t)
xn = sp.Function('xn')(t)
p = sp.Function('p')(t)
b, d, r = sp.symbols('b d r', positive=True)

# Ecuaciones dadas
dx_dt = sp.Eq(x.diff(t), (b - d)*x)
dxn_dt = sp.Eq(xn.diff(t), (b - d)*xn + r*b*(x - xn))
p_def = sp.Eq(p, xn / x)
dp_dt = sp.diff(p_def.rhs, t)
dp_dt_simplified = sp.simplify(dp_dt.subs([(x.diff(t), dx_dt.rhs), (xn.diff(t), dxn_dt.rhs)]))
dp_dt_final = sp.Eq(p.diff(t), dp_dt_simplified)

print("Ecuación diferencial simplificada para p(t):")
sp.pprint(dp_dt_final)

print("\nInciso b - Método de Euler")

# Parámetros dados
p0 = 0.01
b_val = 0.02
d_val = 0.015
r_val = 0.1
rb_val = r_val * b_val
T = 50
h = 1
n_steps = int(T / h)

# Método de Euler
t_values = np.arange(0, T + h, h)
p_values = np.zeros_like(t_values)
p_values[0] = p0

for i in range(n_steps):
    p_values[i+1] = p_values[i] + h * rb_val * (1 - p_values[i])

print(f"Aproximación de p(50) con Euler: {p_values[-1]:.6f}")

print("\nInciso c - Solución exacta")

# Solución directa: p(t) = 1 - C*exp(-rb*t)
C = sp.Symbol('C')
p_general = 1 - C * sp.exp(-r * b * t)
C_val = sp.solve(p_general.subs(t, 0) - p0, C)[0]
p_solution = p_general.subs(C, C_val)
p_50 = p_solution.subs({t: 50, b: b_val, r: r_val})
p_50_val = float(p_50.evalf())

print("Solución exacta p(t):")
sp.pprint(sp.Eq(p, p_solution))
print(f"\np(50) exacta: {p_50_val:.6f}")
print(f"Comparación con Euler: {p_values[-1]:.6f}")

# Gráfica
plt.plot(t_values, p_values, label="Euler (aproximado)")
t_plot = np.linspace(0, 50, 500)
p_exact_func = sp.lambdify(t, p_solution.subs({b: b_val, r: r_val}), modules="numpy")
plt.plot(t_plot, p_exact_func(t_plot), label="Exacta", linestyle='--')
plt.xlabel('Tiempo t (años)')
plt.ylabel('Proporción p(t)')
plt.title('Proporción de no conformistas en el tiempo')
plt.legend()
plt.grid(True)
plt.show()
