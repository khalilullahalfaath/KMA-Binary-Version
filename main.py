import numpy as np
import matplotlib.pyplot as plt
from src.algorithms.kma_algorithm import KMA
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


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

        if function_id == 0:
            self.X = X
            self.y = y

        self.kma = KMA(
            function_id, dimension, max_num_eva, pop_size, X, y, transfer_function_name
        )

    def run(self):
        best_indiv, opt_val, num_eva, fopt, fmean, proc_time, evo_pop_size = (
            self.kma.run()
        )

        if self.function_id == 0:
            # get the data
            # split data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            #  convert to binary
            best_indiv = np.array(best_indiv)  # Ensure x_input is a NumPy array
            binary_solution = (best_indiv >= np.random.rand(len(best_indiv))).astype(
                int
            )

            # selet the features
            selected_features_train = X_train[:, binary_solution == 1]
            selected_features_test = X_test[:, binary_solution == 1]

            # Initialize Random Forest Classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)

            # Fit the model on the training data
            clf.fit(selected_features_train, y_train)

            # Predict on the test data
            y_pred = clf.predict(selected_features_test)

            # Evaluate accuracy
            accuracy = accuracy_score(y_test, y_pred)

        self.report_results(
            best_indiv, opt_val, num_eva, proc_time, accuracy, binary_solution
        )
        self.visualize_convergence(fopt, fmean)
        self.visualize_log_convergence(fopt, fmean)
        self.visualize_population_size(evo_pop_size)

    def report_results(
        self, best_indiv, opt_val, num_eva, proc_time, accuracy=0, binary_solution=0
    ):
        print(f"Function              = F{self.function_id}")
        print(f"Dimension             = {self.dimension}")
        print(f"Number of evaluations = {num_eva}")
        print(f"Processing time (sec) = {proc_time:.10f}")
        print(f"Global optimum        = {self.kma.fthreshold_fx:.10f}")
        print(f"Actual solution       = {opt_val:.10f}")
        print(f"Best individual       = {best_indiv}")

        if function_id == 0:
            print(f"Binary solution   = {binary_solution}")
            print(f"Accuracy          = {accuracy}")

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

    dimension = 50

    max_num_eva = 25000
    pop_size = 5

    if function_id == 0:
        data = load_breast_cancer()
        X = data.data
        y = data.target

        X_shape = X.shape
        y_shape = y.shape

        # NOTE: Update dimension based on x_size
        dimension = X_shape[1]

        # TODO: Update max_num_eva is necessary
        max_num_eva = 2500

    driver = KMADriver(
        function_id, dimension, max_num_eva, pop_size, X, y, "time_varying"
    )
    driver.run()
