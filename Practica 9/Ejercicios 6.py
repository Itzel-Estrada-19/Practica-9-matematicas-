import numpy as np
import matplotlib.pyplot as plt


class AdamsBashforthSolver:
    def __init__(self, f, y0, t0, tf, h, exact_solution=None, problem_name=""):
        self.f = f
        self.y0 = y0
        self.t0 = t0
        self.tf = tf
        self.h = h
        self.exact_solution = exact_solution
        self.problem_name = problem_name
        self.t_values = np.arange(t0, tf + h / 2, h)
        self.n = len(self.t_values)

    def rk4_start(self):
        y = np.zeros(self.n)
        y[0] = self.y0
        for i in range(min(3, self.n - 1)):
            t = self.t_values[i]
            k1 = self.h * self.f(t, y[i])
            k2 = self.h * self.f(t + self.h / 2, y[i] + k1 / 2)
            k3 = self.h * self.f(t + self.h / 2, y[i] + k2 / 2)
            k4 = self.h * self.f(t + self.h, y[i] + k3)
            y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return y

    def solve(self):
        y = self.rk4_start()
        if self.n >= 4:
            for i in range(3, self.n - 1):
                t = self.t_values[i]
                f0 = self.f(t, y[i])
                f1 = self.f(t - self.h, y[i - 1])
                f2 = self.f(t - 2 * self.h, y[i - 2])
                f3 = self.f(t - 3 * self.h, y[i - 3])
                y[i + 1] = y[i] + self.h * (55 * f0 - 59 * f1 + 37 * f2 - 9 * f3) / 24
        return y

    def exact_values(self):
        if self.exact_solution is None:
            return None
        return np.array([self.exact_solution(t) for t in self.t_values])

    def get_results(self):
        y_approx = self.solve()
        y_exact = self.exact_values() if self.exact_solution else None
        errors = np.abs(y_approx - y_exact) if y_exact is not None else None
        return {
            'problem': self.problem_name,
            't_values': self.t_values,
            'approx': y_approx,
            'exact': y_exact,
            'errors': errors
        }


def print_result(res):
    print(f"\n>>> PROBLEMA {res['problem'].upper()} <<<")
    print(f"{'t':<8}{'Aproximación':<20}{'Exacto':<20}{'Error absoluto':<20}")
    print("-" * 68)
    for i in range(len(res['t_values'])):
        t = res['t_values'][i]
        approx = res['approx'][i]
        exact = res['exact'][i] if res['exact'] is not None else np.nan
        error = res['errors'][i] if res['errors'] is not None else np.nan
        print(f"{t:<8.2f}{approx:<20.8f}{exact:<20.8f}{error:<20.8f}")


def plot_result(res):
    plt.figure(figsize=(8, 6))
    plt.plot(res['t_values'], res['approx'], 'bo-', label='Aproximación')
    if res['exact'] is not None:
        plt.plot(res['t_values'], res['exact'], 'r--', label='Exacto')
    plt.title(f'Solución del problema {res["problem"]}')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()


def solve_all_problems():
    f_a = lambda t, y: t * np.exp(3 * t) - 2 * y
    exact_a = lambda t: (1 / 3) * t * np.exp(3 * t) - (1 / 24) * np.exp(3 * t) + (1 / 25) * np.exp(-2 * t)
    solver_a = AdamsBashforthSolver(f_a, y0=0, t0=0, tf=1, h=0.2,
                                    exact_solution=exact_a, problem_name="a")

    f_b = lambda t, y: 1 + (t - y) ** 2
    exact_b = lambda t: t + 1 / (1 - t)
    solver_b = AdamsBashforthSolver(f_b, y0=1, t0=2, tf=3, h=0.2,
                                    exact_solution=exact_b, problem_name="b")

    f_c = lambda t, y: 1 + y / t
    exact_c = lambda t: t * np.log(t) + 2 * t
    solver_c = AdamsBashforthSolver(f_c, y0=2, t0=1, tf=2, h=0.2,
                                    exact_solution=exact_c, problem_name="c")

    f_d = lambda t, y: np.cos(2 * t) + np.sin(3 * t)
    exact_d = lambda t: (1 / 2) * np.sin(2 * t) - (1 / 3) * np.cos(3 * t) + 4 / 3
    solver_d = AdamsBashforthSolver(f_d, y0=1, t0=0, tf=1, h=0.2,
                                    exact_solution=exact_d, problem_name="d")

    return [
        solver_a.get_results(),
        solver_b.get_results(),
        solver_c.get_results(),
        solver_d.get_results()
    ]


if __name__ == "__main__":
    all_results = solve_all_problems()

    for result in all_results:
        print_result(result)
        plot_result(result)
