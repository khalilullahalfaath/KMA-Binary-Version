import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from src.algorithms.kma_algorithm import KMA
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import itertools
import random


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
        min_adaptive_size,
        max_adaptive_size,
        max_gen_exam1,
        max_gen_exam2,
    ):
        self.function_id = function_id
        self.dimension = dimension
        self.max_num_eva = max_num_eva
        self.pop_size = pop_size
        self.min_adaptive_size = min_adaptive_size
        self.max_adaptive_size = max_adaptive_size
        self.max_gen_exam1 = max_gen_exam1
        self.max_gen_exam2 = max_gen_exam2

        if function_id == 0:
            self.X = X
            self.y = y
            self.transfer_function = transfer_function_name

        self.kma = KMA(
            function_id,
            dimension,
            max_num_eva,
            pop_size,
            X,
            y,
            transfer_function_name,
            min_adaptive_size,
            max_adaptive_size,
            max_gen_exam1,
            max_gen_exam2,
        )

    def run(self, experiment_number=None):
        best_indiv, opt_val, num_eva, fopt, fmean, proc_time, evo_pop_size = (
            self.kma.run()
        )

        results = {
            "Experiment": experiment_number,
            "Function_ID": self.function_id,
            "Dimension": self.dimension,
            "Number_of_Evaluations": num_eva,
            "Processing_Time": proc_time,
            "Global_Optimum": self.kma.fthreshold_fx,
            "Actual_Solution": opt_val,
            "Fitness_Optimum": str(fopt.tolist()),
            "Fitness_Mean": str(fmean.tolist()),
            "pop_size": self.pop_size,
            "max_num_eva": self.max_num_eva,
            "min_adaptive_size": self.min_adaptive_size,
            "max_adaptive_size": self.max_adaptive_size,
            "max_gen_exam1": self.max_gen_exam1,
            "max_gen_exam2": self.max_gen_exam2,
        }

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

            # If no features are selected, create a randomized index
            if selected_features_train.shape[1] == 0:
                num_positions = np.random.choice([2, 3])
                n = X_train.shape[1]

                if selected_features_train.shape[1] == 0:
                    # If no features selected, randomly select features
                    positions_to_change = np.random.choice(
                        n, num_positions, replace=False
                    )
                    binary_solution[positions_to_change] = 1

                    # reselect features
                    selected_features_train = X_train[:, binary_solution == 1]
                    selected_features_test = X_test[:, binary_solution == 1]

            # Initialize Random Forest Classifier
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

            # Fit the model on the training data
            clf.fit(selected_features_train, y_train)

            # Predict on the test data
            y_pred = clf.predict(selected_features_test)

            # Evaluate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            number_selected = selected_features_train.shape[1]
            number_total = X_train.shape[1]

            alpha = 0.99

            fx = accuracy + alpha * (1 - (number_selected / number_total))

            fitness_value = -fx

            # Update results dictionary
            results.update(
                {
                    "Transfer_Function": self.transfer_function,
                    "Binary_Solution": str(binary_solution.tolist()),
                    "Number_Selected_Features": number_selected,
                    "Total_Features": number_total,
                    "Accuracy": accuracy,
                    "Fitness_Value": -fitness_value,
                }
            )

            print(results.keys())

        # Save results to CSV
        self.save_results_to_csv(results)

        self.report_results(
            best_indiv,
            opt_val,
            num_eva,
            proc_time,
            accuracy,
            binary_solution,
            fitness_value,
        )
        # self.visualize_convergence(fopt, fmean)
        # self.visualize_log_convergence(fopt, fmean)
        # self.visualize_population_size(evo_pop_size)

    def save_results_to_csv(self, results):
        """Save experiment results to a CSV file, appending if file exists."""
        csv_path = "kma_experiments_transfer_function.csv"

        # Check if file exists to determine whether to write header
        file_exists = os.path.exists(csv_path)

        # Convert results to DataFrame
        results_df = pd.DataFrame([results])

        # Append to CSV, write header only if file doesn't exist
        results_df.to_csv(csv_path, mode="a", header=not file_exists, index=False)

    def report_results(
        self,
        best_indiv,
        opt_val,
        num_eva,
        proc_time,
        accuracy=0,
        binary_solution=0,
        fitness_value=0,
    ):
        print(f"Function              = F{self.function_id}")
        print(f"Dimension             = {self.dimension}")
        print(f"Number of evaluations = {num_eva}")
        print(f"Processing time (sec) = {proc_time:.10f}")
        print(f"Global optimum        = {self.kma.fthreshold_fx:.10f}")
        print(f"Actual solution       = {opt_val:.10f}")
        print(f"Best individual       = {best_indiv}")

        if self.function_id == 0:
            print(f"Binary solution   = {binary_solution}")
            print(f"Accuracy          = {accuracy}")
            print(f"Fitness value     = {-fitness_value}")

    # Rest of the methods remain the same as in the original script
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


def random_search_parameters(
    pop_size_range=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    max_num_eva_range=[1000, 1500, 2000, 2500],
    min_adaptive_size_range=[1, 2, 3, 4, 5],
    max_adaptive_size_range=[6, 7, 8, 9, 10],
    max_gen_exam1_range=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    max_gen_exam2_range=[100, 200, 300, 400, 500],
    num_iterations=100,
):
    """
    Generate random parameter combinations for KMA algorithm.

    Args:
        *_range: Lists of possible values for each parameter
        num_iterations: Number of random combinations to generate

    Returns:
        list: A list of dictionaries with random parameter combinations
    """
    random_combinations = []

    for _ in range(num_iterations):
        random_combination = {
            "pop_size": random.choice(pop_size_range),
            "max_num_eva": random.choice(max_num_eva_range),
            "min_adaptive_size": random.choice(min_adaptive_size_range),
            "max_adaptive_size": random.choice(max_adaptive_size_range),
            "max_gen_exam1": random.choice(max_gen_exam1_range),
            "max_gen_exam2": random.choice(max_gen_exam2_range),
        }
        random_combinations.append(random_combination)

    return random_combinations


def main_multi_run_hyperparameters():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # function id = 0 -> feature selection
    function_id = 0

    # Load breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Get dataset dimensions
    X_shape = X.shape
    dimension = X_shape[1]

    # Define parameter ranges
    pop_size_range = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    max_num_eva_range = [1000, 1500, 2000, 2500]
    min_adaptive_size_range = [1, 2, 3, 4, 5]
    max_adaptive_size_range = [6, 7, 8, 9, 10]
    max_gen_exam1_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    max_gen_exam2_range = [100, 200, 300, 400, 500]

    # Number of random search iterations
    num_iterations = 100
    num_experiments_per_iteration = 10

    # Generate random parameter combinations
    random_parameter_combinations = random_search_parameters(
        pop_size_range,
        max_num_eva_range,
        min_adaptive_size_range,
        max_adaptive_size_range,
        max_gen_exam1_range,
        max_gen_exam2_range,
        num_iterations,
    )

    # Print total number of combinations
    print(
        f"Total number of random parameter combinations: {len(random_parameter_combinations)}"
    )

    # Run experiments for each random parameter combination
    for combo_index, param_combo in enumerate(random_parameter_combinations, 1):
        print(
            f"\n--- Random Parameter Combination {combo_index}/{len(random_parameter_combinations)} ---"
        )
        print("Parameters:", param_combo)

        # Run multiple experiments for each parameter combination
        for experiment in range(1, num_experiments_per_iteration + 1):
            print(f"\n--- Experiment {experiment} ---")

            driver = KMADriver(
                function_id,
                dimension,
                param_combo["max_num_eva"],
                param_combo["pop_size"],
                X,
                y,
                "time_varying",
                param_combo["min_adaptive_size"],
                param_combo["max_adaptive_size"],
                param_combo["max_gen_exam1"],
                param_combo["max_gen_exam2"],
            )
            driver.run(experiment_number=experiment)


def main_multi_run_transfer_functions():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Hyperparameter terbaik
    pop_size = 8
    max_num_eva = 1000
    min_adaptive_size = 5
    max_adaptive_size = 6
    max_gen_exam1 = 70
    max_gen_exam2 = 500

    # Daftar transfer functions
    transfer_functions = [
        "s_shaped_1",
        "s_shaped_2",
        "s_shaped_3",
        "s_shaped_4",
        "v_shaped_1",
        "v_shaped_2",
        "v_shaped_3",
        "v_shaped_4",
        "time_varying_v1",
        "time_varying_v2",
        "time_varying_v3",
        "time_varying_v4",
        "time_varying_s1",
        "time_varying_s2",
        "time_varying_s3",
        "time_varying_s4",
    ]

    # Load breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Jalankan eksperimen untuk setiap transfer function
    results = []
    num_experiments = 20

    for function_name in transfer_functions:
        accuracies = []
        print(f"\n--- Testing Transfer Function: {function_name} ---")
        for experiment in range(1, num_experiments + 1):
            driver = KMADriver(
                function_id=0,
                dimension=X.shape[1],
                max_num_eva=max_num_eva,
                pop_size=pop_size,
                X=X,
                y=y,
                transfer_function_name=function_name,
                min_adaptive_size=min_adaptive_size,
                max_adaptive_size=max_adaptive_size,
                max_gen_exam1=max_gen_exam1,
                max_gen_exam2=max_gen_exam2,
            )
            driver.run(experiment_number=experiment)


def main_single_run():
    # function id = 0 -> feature selection
    function_id = 0

    # Load breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Get dataset dimensions
    X_shape = X.shape
    dimension = X_shape[1]

    # Define hyperparameters
    pop_size = 8
    max_num_eva = 2500
    min_adaptive_size = 3
    max_adaptive_size = 8
    max_gen_exam1 = 40
    max_gen_exam2 = 200

    driver = KMADriver(
        function_id,
        dimension,
        max_num_eva,
        pop_size,
        X,
        y,
        "time_varying",
        min_adaptive_size,
        max_adaptive_size,
        max_gen_exam1,
        max_gen_exam2,
    )

    driver.run(experiment_number=experiment)


if __name__ == "__main__":
    main_multi_run_transfer_functions()
