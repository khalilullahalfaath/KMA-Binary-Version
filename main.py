import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.kma_algorithm import KMA
from sklearn.datasets import load_breast_cancer



class KMADriver:
    def __init__(
        self,
        function_id: int,
        dimension: int,
        max_num_eva: int,
        pop_size: int,
        X: np.ndarray,
        y: np.ndarray,
        transfer_function_name,
    ):
        self.function_id = function_id
        self.dimension = dimension
        self.max_num_eva = max_num_eva
        self.pop_size = pop_size
        self.kma = KMA(
            function_id, dimension, max_num_eva, pop_size, X, y, transfer_function_name
        )

    def run(self):
        best_indiv, opt_val, num_eva, fopt, fmean, proc_time, evo_pop_size = (
            self.kma.run()
        )
        self.report_results(best_indiv, opt_val, num_eva, proc_time)
        self.visualize_convergence(fopt, fmean)
        self.visualize_log_convergence(fopt, fmean)
        self.visualize_population_size(evo_pop_size)

    def report_results(self, best_indiv, opt_val, num_eva, proc_time):
        print(f"Function              = F{self.function_id}")
        print(f"Dimension             = {self.dimension}")
        print(f"Number of evaluations = {num_eva}")
        print(f"Processing time (sec) = {proc_time:.10f}")
        print(f"Global optimum        = {self.kma.fthreshold_fx:.10f}")
        print(f"Actual solution       = {opt_val:.10f}")
        print(f"Best individual       = {best_indiv}")

    def visualize_convergence(self, fopt, fmean):
        plt.figure(figsize=(10, 6))
        plt.plot(fopt, "r-", label="Best fitness")
        plt.plot(fmean, "b--", label="Mean fitness")
        plt.title(f"Convergence curve of Function F{self.function_id}")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()

    def visualize_log_convergence(self, fopt, fmean):
        if fopt[-1] >= 0:
            plt.figure(figsize=(10, 6))
            plt.plot(np.log10(fopt), "r-", label="Best fitness")
            plt.plot(np.log10(fmean), "b--", label="Mean fitness")
            plt.title(f"Log convergence curve of Function F{self.function_id}")
            plt.xlabel("Generation")
            plt.ylabel("Log Fitness")
            plt.legend()
            plt.show()

    def visualize_population_size(self, evo_pop_size):
        plt.figure(figsize=(10, 6))
        plt.plot(evo_pop_size)
        plt.axis([1, len(evo_pop_size), 0, self.kma.max_ada_pop_size + 5])
        plt.title(
            f"Fixed and self-adaptive population size for Function F{self.function_id}"
        )
        plt.xlabel("Generation")
        plt.ylabel("Population size (Number of individuals)")
        plt.show()


if __name__ == "__main__":
    # function id = 0 -> feature selection
    function_id = 0

    # NOTE: Update dimension based on x_size
    dimension = 50

    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_shape = X.shape
    y_shape = y.shape

    max_num_eva = 25000
    pop_size = 5

    driver = KMADriver(
        function_id, dimension, max_num_eva, pop_size, X, y, "s_shaped_1"
    )
    driver.run()
