import sympy as sp

print("Ejercicio 5.3 - Método de Taylor de segundo orden")

# Definir símbolos
t, y = sp.symbols('t y')

def metodo_taylor_orden_2(f_expr, y0, t0, h, n, inciso):
    f = sp.lambdify((t, y), f_expr, modules='numpy')
    df = sp.diff(f_expr, t) + sp.diff(f_expr, y) * f_expr
    df_func = sp.lambdify((t, y), df, modules='numpy')

    print(f"\nInciso {inciso}")
    print("Valores de inicio:")
    print(f"t = {t0:.2f}, y = {y0:.4f}")
    for i in range(1, n + 1):
        y1 = y0 + h * f(t0, y0) + (h**2 / 2) * df_func(t0, y0)
        t0 += h
        y0 = y1
        print(f"Iteración {i}: t = {t0:.4f}, y = {y0:.4f}")

# Inciso a: y' = t*e^{3t} - 2y, y(0)=0, h=0.5, 0 ≤ t ≤ 1
f_a = t * sp.exp(3*t) - 2*y
metodo_taylor_orden_2(f_a, y0=0, t0=0, h=0.5, n=2, inciso='a')

# Inciso b: y' = 1 + (t - y)^2, y(2)=1, h=0.5, 2 ≤ t ≤ 3
f_b = 1 + (t - y)**2
metodo_taylor_orden_2(f_b, y0=1, t0=2, h=0.5, n=2, inciso='b')

# Inciso c: y' = 1 + y/t, y(1)=2, h=0.25, 1 ≤ t ≤ 2
f_c = 1 + y/t
metodo_taylor_orden_2(f_c, y0=2, t0=1, h=0.25, n=4, inciso='c')

# Inciso d: y' = cos(2t) + sin(3t), y(0)=1, h=0.25, 0 ≤ t ≤ 1
f_d = sp.cos(2*t) + sp.sin(3*t)
metodo_taylor_orden_2(f_d, y0=1, t0=0, h=0.25, n=4, inciso='d')

# ========================
# Ejercicio 7 - Proyectil con resistencia del aire
# ========================
print("\nEjercicio 7 - Movimiento de un proyectil con resistencia del aire")

# Definir variables
t = sp.Symbol('t')
v = sp.Function('v')(t)

# Datos del problema
m = 0.11  # kg
g = 9.8   # m/s^2
k = 0.002 # kg/m
v0 = 8.0  # m/s

# Ecuación diferencial: m*v' = -mg - k*v*|v|
# Esto se transforma en: dv/dt = -g - (k/m)*v*|v|
def dvdt(v):
    return -g - (k/m)*v*abs(v)

# Método de Euler para calcular la velocidad
t0 = 0
v_t = v0
h = 0.1
n = 10  # de 0.1 a 1.0 s

print("\nInciso a) Velocidad desde t = 0.1 hasta t = 1.0:")
for i in range(1, n+1):
    v_t = v_t + h * dvdt(v_t)
    t0 += h
    print(f"t = {t0:.1f} s, v = {v_t:.4f} m/s")

# Para el inciso b), buscamos cuándo v ≈ 0 (altura máxima)
print("\nInciso b) Tiempo en que el proyectil alcanza su altura máxima:")

t0 = 0
v_t = v0
h = 0.01
while v_t > 0:
    v_t = v_t + h * dvdt(v_t)
    t0 += h

tiempo_max = round(t0, 1)
print(f"El proyectil alcanza su altura máxima aproximadamente a los {tiempo_max:.1f} segundos.")
