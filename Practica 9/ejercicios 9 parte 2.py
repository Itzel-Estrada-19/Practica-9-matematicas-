import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del modelo
k1 = 3.0      # Tasa de natalidad de presas
k2 = 0.002    # Tasa de mortalidad de presas por depredador
k3 = 0.0006   # Tasa de natalidad de depredadores por presa
k4 = 0.5      # Tasa de mortalidad de depredadores

# Condiciones iniciales
x1_0 = 1000   # Población inicial de presas
x2_0 = 500    # Población inicial de depredadores

# Sistema de ecuaciones diferenciales
def lotka_volterra(t, y):
    x1, x2 = y
    dx1dt = k1 * x1 - k2 * x1 * x2
    dx2dt = k3 * x1 * x2 - k4 * x2
    return [dx1dt, dx2dt]

# Intervalo de tiempo
t_span = (0, 4)
t_eval = np.linspace(0, 4, 500)

# Resolver el sistema
sol = solve_ivp(lotka_volterra, t_span, [x1_0, x2_0], t_eval=t_eval, method='RK45')

# Extraer resultados
t = sol.t
x1 = sol.y[0]
x2 = sol.y[1]

# Gráfica de las poblaciones en el tiempo
plt.figure(figsize=(12, 6))
plt.plot(t, x1, 'b-', label='Presa (x₁)')
plt.plot(t, x2, 'r-', label='Depredador (x₂)')
plt.title('Modelo Lotka-Volterra: Dinámica Poblacional', fontsize=14)
plt.xlabel('Tiempo (años)', fontsize=12)
plt.ylabel('Población', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Gráfica de fase (x1 vs x2)
plt.figure(figsize=(8, 8))
plt.plot(x1, x2, 'g-', linewidth=1)
plt.title('Diagrama de Fase: Presas vs Depredadores', fontsize=14)
plt.xlabel('Población de Presas (x₁)', fontsize=12)
plt.ylabel('Población de Depredadores (x₂)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Punto de equilibrio estable
x1_eq = k4 / k3
x2_eq = k1 / k2
plt.plot(x1_eq, x2_eq, 'ro', markersize=8, label=f'Equilibrio ({x1_eq:.0f}, {x2_eq:.0f})')
plt.legend(fontsize=12)
plt.show()

# Análisis de estabilidad
print("\nAnálisis de Estabilidad:")
print(f"Punto de equilibrio: x₁ = {x1_eq:.2f}, x₂ = {x2_eq:.2f}")
print("\nInterpretación física:")
print("1. Las poblaciones oscilan en ciclos alrededor del punto de equilibrio.")
print("2. Cuando hay muchas presas, los depredadores aumentan, lo que luego reduce las presas.")
print("3. Al disminuir las presas, los depredadores también disminuyen, permitiendo que las presas se recuperen.")
print("4. Este ciclo se repite indefinidamente en el modelo básico.")