import math, random, matplotlib.pyplot as plt


class single_variable_optimizer:

    def __init__(self, f, a, b, eps):
        self.f = f
        self.a = a
        self.b = b
        self.eps = eps
        self.fun_evals = 0

    def plot(self, plotting, algo):
        # Unpack the list of tuples into two lists
        x_values, y_values = zip(*plotting)

        # Create a plot
        if algo == "Bounding Phase":
            plt.plot(x_values, y_values, marker="o", linestyle="-")
        else:
            plt.scatter(x_values, y_values)
        plt.xlabel("Iterations")
        plt.ylabel("function values")
        plt.title(algo)
        plt.grid(True)

        # Show the plot
        plt.show()

    def fit(self):
        a_new, b_new, history_bf = self.bounding_phase(self.a, self.b)
        a_final, b_final, history_ih = self.interval_halving_method(
            a_new, b_new, self.eps
        )
        return a_final, b_final, history_bf, history_ih, self.fun_evals

    # Function to evaluate the objective function
    # It negates the result if maximization is required, effectively turning it into a minimization problem

    # Function to perform the bounding phase method
    # This method helps in finding a smaller interval [a_new, b_new] where the optimal solution is likely to be
    def bounding_phase(self, a, b):

        iterations_guess = 0
        fun_eval_guess = 0

        iterations = 0
        fun_eval = 0

        delta = 0.5  # Initial step size
        x = 0  # Starting point
        fun = 0
        f1 = 0
        f3 = 0

        while True:

            fun_eval_guess += 3
            iterations_guess += 1

            x = random.uniform(
                a, b
            )  # Randomly select a starting point within the range

            # Calculating the Function Values at neighbouring points
            f1 = self.f(x - abs(delta))
            fun = self.f(x)
            f3 = self.f(x + abs(delta))

            # Check the function values to decide the direction
            if f1 >= fun and fun >= f3:
                delta = abs(delta)
                break
            elif f1 <= fun and fun <= f3:
                delta = -1 * abs(delta)
                break

        k = 0  # Initialize iteration counter
        x_prev = 0  # Previous x value
        x_next = x + 2**k * delta  # Next x value

        plotting = []

        plotting.append((iterations, fun))

        if delta < 0:
            f_next = f1
        else:
            f_next = f3

        plotting.append((iterations, f_next))

        # Expand the search interval until the function value starts to increase
        while True:
            iterations += 1
            if f_next < fun:
                fun = f_next
                x_prev = x
                x = x_next
                if delta > 0:
                    x_next = min(
                        b, x + 2**k * delta
                    )  # Ensure x_next stays within the upper bound
                else:
                    x_next = max(
                        a, x + 2**k * delta
                    )  # Ensure x_next stays within the lower bound
                k += 1
                f_next = self.f(x_next)
                fun_eval += 1
            else:
                break

            plotting.append((iterations, f_next))

        self.fun_evals += fun_eval + fun_eval_guess

        # Ensure that x_prev is less than x_next
        if x_prev > x_next:
            x_prev, x_next = x_next, x_prev

        return x_prev, x_next, plotting

    # Function to perform the interval halving method
    # This method further refines the interval to pinpoint the optimal solution
    def interval_halving_method(self, a, b, eps):

        iterations = 0
        fun_eval = 1

        plotting = []

        xm = (a + b) / 2  # Midpoint of the interval
        f2 = self.f(xm)
        l = b - a  # Length of the interval

        # Continue halving the interval until it is smaller than the tolerance level
        while abs(l) >= eps:
            iterations += 1
            fun_eval += 2

            x1 = a + l / 4  # First quarter point
            x2 = b - l / 4  # Third quarter point

            f1 = self.f(x1)
            f3 = self.f(x2)

            plotting.append((iterations, f2))

            # Narrow down the interval based on the function values at x1, xm, and x2
            if f1 < f2:
                b = xm
                xm = x1
                f2 = f1
            elif f2 > f3:
                a = xm
                xm = x2
                f2 = f3
            else:
                a = x1
                b = x2

            l = b - a  # Update the length of the interval

        self.fun_evals += fun_eval
        return a, b, plotting


def function(value):

    eps = 10**-3

    f1 = lambda x: -1 * ((2 * x - 5) ** 4 - (x**2 - 1) ** 3)

    f2 = lambda x: -1 * (8 + x**3 - 2 * x - 2 * math.exp(x))

    f3 = lambda x: -1 * (4 * x * math.sin(x))

    f4 = lambda x: (2 * (x - 3) ** 2 + math.exp(0.5 * x**2))

    f5 = lambda x: x**2 - 10 * math.exp(0.1 * x)

    f6 = lambda x: -1 * (20 * math.sin(x) - 15 * x**2)

    match value:
        case 1:
            return f1, -10, 0, eps
        case 2:
            return f2, -2, 1, eps
        case 3:
            return f3, 0.5, 3.14, eps
        case 4:
            return f4, -2, 3, eps
        case 5:
            return f5, -6, 6, eps
        case 6:
            return f6, -4, 4, eps

def main():
    f, a, b, eps = function(6)
    optimizer = single_variable_optimizer(f, a, b, eps)
    l, h, history_bf, history_ih = optimizer.fit()
    print(f"optima lies in the interval {l:.3f}, {h:.3f}")
    optimizer.plot(history_bf, "Bounding_phase")
    optimizer.plot(history_ih, "Interval Halving")


if __name__ == "__main__":
    main()
