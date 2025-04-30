import  sympy as sp
#Declarar variables simbolicas
t = sp.symbols("t")
y = sp.Function("y")
print("Ejercicios 5.1\n Ejercicio 1")
#Inciso a
f_a = y(t)*sp.cos(t)
r_a = sp.dsolve(sp.Eq(sp.Derivative(y(t), t),f_a),y(t), ics={y(0): 1})
print("Resultado= ",r_a)
#Inciso b
f_b = (2/t) * y(t) + t*2 * sp.exp(t)
r_b = sp.dsolve(sp.Eq(sp.Derivative(y(t), t),f_b),y(t), ics={y(1): 0})
print("Resultado= ",r_b)
#Inciso c
f_c = -(2/t) * y(t) + t*2 * sp.exp(t)
r_c = sp.dsolve(sp.Eq(sp.Derivative(y(t), t),f_c),y(t), ics={y(1): sp.sqrt(2)*sp.exp(1)})
print("Resultado= ",r_c)
#Inciso d
f_d = (4*t*3*y(t))/(1+t*4)
r_d = sp.dsolve(sp.Eq(sp.Derivative(y(t), t),f_d),y(t), ics={y(0): 1})
print("Resultado= ",r_d)

#Veriificar continuidad y condiciones de Lipschitz
for label, expr in zip(['a', 'b', 'c', 'd'], [f_a, f_b, f_c, f_d]):
    df_dy = sp.diff(expr, y(t))
    print(f"Inciso ({label}): ∂f/∂y = {df_dy} (continua: Sí)")
t, tau = sp.symbols("t tau")
y = sp.Function("y")
print("\nEjercicio 7")
y0 = 1  # y0(t) es constante
f_tau = -y0 + tau + 1
y1 = 1 + sp.integrate(f_tau, (tau, 0, t))
print("y1(t) usando Picard:", y1)
#Ejercicio 7
# Declarar símbolos
t, tau = sp.symbols("t tau")
y = sp.Function("y")
# Inciso a
y0 = 1
f_tau = -y0 + tau + 1
y1 = 1 + sp.integrate(f_tau, (tau, 0, t))
print("Inciso a y1(t) usando Picard:", y1)
# Inciso b
yk = [sp.Integer(1)]  # y0(t) = 1

for i in range(3):  # Generar y1, y2, y3
    prev_y = yk[-1].subs(t, tau)  # y_{k-1}(tau)
    f_tau = -prev_y + tau + 1
    y_next = 1 + sp.integrate(f_tau, (tau, 0, t))
    yk.append(sp.simplify(y_next))
# Mostrar resultados
print("Resultados del inciso b:")
for i, yi in enumerate(yk):
    print(f"y{i}(t) = {yi}")