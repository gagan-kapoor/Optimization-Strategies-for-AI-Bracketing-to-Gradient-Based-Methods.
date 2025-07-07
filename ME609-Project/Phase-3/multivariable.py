import numpy as np
from single_var_optimizer import *
import json

class Multi_variable_optimizer:
    def __init__(self, f, m, low, high, d, eps = 10**-6):
        self.f = f
        self.m = m
        self.low = low
        self.high = high
        self.d = d
        self.k = 0
        self.eps = eps
        self.fun_evals = 0
        self.history = []

    def surface_plot(self, history):

        if len(history[1]) != 2:
            print("surface plot is not possible!")
            return None

        x_history = np.array([item[1] for item in history])  # Extract all x vectors
        x1_vals = x_history[:, 0]  # First component of x (x1)
        x2_vals = x_history[:, 1]  # Second component of x (x2)

        # Compute function values at each point in x_history
        Z_history = np.array(
            [self.f(x) for x in x_history]
        )  # Compute f(x) for each x in the history

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        x = y = np.arange(self.low, self.high, 0.1)
        X, Y = np.meshgrid(x, y)
        zs = np.array([self.f(l) for l in np.column_stack((np.ravel(X), np.ravel(Y)))])
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z, alpha=0.6, edgecolor="none", cmap="viridis")
        ax.contour(
            X,
            Y,
            Z,
            offset=np.min(Z) - 100,
            levels=30,
            cmap="viridis",
            linestyles="solid",
        )
        ax.scatter(x1_vals, x2_vals, Z_history, s=50, c="black", marker="x")
        ax.plot(x1_vals, x2_vals, Z_history, color="black", lw=2, alpha=0.5)
        ax.scatter(x1_vals, x2_vals, np.min(Z) - 100, s=50, c="black", marker="x")
        ax.plot(x1_vals, x2_vals, np.min(Z) - 100, color="black", lw=2)

        plt.show()

    def plot_progress(self, history):
        title = "Optimization Progress"
        xlabel = "Iteration"
        ylabel = "Objective Function Value"
        history = np.array(history, dtype=object)

        # Extract iteration numbers and x vectors
        iterations = [item[0] for item in history]  # Iteration numbers
        x_vectors = np.array(
            [item[1] for item in history]
        )  # Extract the x vectors as a NumPy array

        # Compute function values for each x_vector using the objective function
        function_values = np.array([self.f(x) for x in x_vectors])

        # Plotting the progress
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, function_values, marker="o", linestyle="-", color="b")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    def partial_derivative(self, i, x):
        eps = 10**-2
        original_value = x[i]

        x[i] = original_value + eps
        right = self.f(x)

        x[i] = original_value - eps
        left = self.f(x)

        x[i] = original_value

        return (right - left) / (2.0 * eps)

    def gradient(self, x):
        self.fun_evals += 2 * len(x)
        grad = np.array([self.partial_derivative(i, x) for i in range(len(x))])
        return grad

    def alpha_range(self, x, d):
        L = np.array(self.low)
        x = np.array(x)
        d = np.array(d)
        H = np.array(self.high)

        lower_bounds = float("-inf")
        upper_bounds = float("inf")

        for i in range(len(d)):
            if d[i] > 0:
                lower_bounds = max(lower_bounds, (L[i] - x[i]) / d[i])
                upper_bounds = min(upper_bounds, (H[i] - x[i]) / d[i])
            elif d[i] < 0:
                lower_bounds = max(lower_bounds, (H[i] - x[i]) / d[i])
                upper_bounds = min(upper_bounds, (L[i] - x[i]) / d[i])

        return lower_bounds, upper_bounds

    def unidirectional_search(self, x, d, eps1):

        def multi_to_single(alpha, X=x, D=d):
            return self.f(X + alpha * D)

        a, b = self.alpha_range(x, d)

        optimize = single_variable_optimizer(multi_to_single, a, b, eps1)
        l, r, _, _, f_eval = optimize.fit()
        self.fun_evals += f_eval
        return x + l * d

    def angle_between_vectors(self, a, b):

        dot_product = np.dot(a, b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        cos_theta = dot_product / (norm_a * norm_b)
        cos_theta = np.clip(cos_theta, -1, 1)

        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def Conjugate_Gradient_Method(self, x):

        eps1 = eps2 = 10**-6

        grad_x = self.gradient(x)
        s = -1 * grad_x
        s = s / np.linalg.norm(s)

        x_next = self.unidirectional_search(x, s, eps1)
        grad_x_next = self.gradient(x_next)

        self.history.append([self.k, x])
        self.history.append([self.k, x_next])

        if (
            np.linalg.norm(x_next - x) / np.linalg.norm(x) <= eps2
            or np.linalg.norm(grad_x_next) <= self.eps
        ):
            return x_next

        while self.k <= self.m:
            s_next = (
                -1 * grad_x_next
                + ((np.linalg.norm(grad_x_next) ** 2) / (np.linalg.norm(grad_x) ** 2))
                * s
            )
            s_next = s_next / np.linalg.norm(s_next)

            x = x_next
            x_next = self.unidirectional_search(x_next, s_next, eps1)

            self.history.append([self.k, x_next])

            grad_x = grad_x_next
            grad_x_next = self.gradient(x_next)

            if (
                np.linalg.norm(x_next - x) / np.linalg.norm(x) <= eps2
                or np.linalg.norm(grad_x_next) <= self.eps
            ):
                return x_next

            if np.array_equal(s, s_next) or self.angle_between_vectors(s, s_next) < 1:
                return self.Conjugate_Gradient_Method(x_next)

            s = s_next
            self.k += 1

        return x_next

def function(value):
    sum_squares = lambda x: np.sum((np.arange(1, len(x) + 1) * np.array(x) ** 2))

    rosenbrock = lambda x: np.sum(
        100 * (np.array(x[1:]) - np.array(x[:-1]) ** 2) ** 2
        + (np.array(x[:-1]) - 1) ** 2
    )

    dixon_price = lambda x: (x[0] - 1) ** 2 + np.sum(
        np.arange(2, len(x) + 1) * (2 * np.array(x[1:]) ** 2 - np.array(x[:-1])) ** 2
    )

    trid = lambda x: np.sum((np.array(x) - 1) ** 2) - np.sum(
        np.array(x[1:]) * np.array(x[:-1])
    )

    zakharov = (
        lambda x: np.sum(np.array(x) ** 2)
        + np.sum(0.5 * np.arange(1, len(x) + 1) * np.array(x)) ** 2
        + np.sum(0.5 * np.arange(1, len(x) + 1) * np.array(x)) ** 4
    )

    match value:
        case 1:
            return sum_squares
        case 2:
            return rosenbrock
        case 3:
            return dixon_price
        case 4:
            return trid
        case 5:
            return zakharov

def main():
    function_id = input("enter function id: ")

    data = None
    with open("input.json", "r") as file:
        data = json.load(file)

    f = function(int(function_id))
    l, h = data[function_id]["range"]
    d = data[function_id]["d"]

    optimizer = Multi_variable_optimizer(f, 5000, l, h, d)

    x = np.random.uniform(l, h, d)

    print("initial random guess from uniform distribution:", x)

    optimal_x = optimizer.Conjugate_Gradient_Method(x)
    history = optimizer.history
    iterations = optimizer.k
    print("Optimal X is", " ".join([f"{val:.3f}" for val in optimal_x]))
    print(f"Objective Function Value at Optimal X is {f(optimal_x)}")
    print("Total number of Iterations", iterations)
    print("Total number of function evaluations", optimizer.fun_evals)

    optimizer.plot_progress(history)

    if d == 2:
        optimizer.surface_plot(history)


# 1. 0 0 0 0 0
# 2. 1 1 1
# 3. 1.000 0.707 0.595 +-0.545
# 4. 6.000 10.000 12.000 12.000 10.000 6.000
# 5. 0 0


if __name__ == "__main__":
    main()