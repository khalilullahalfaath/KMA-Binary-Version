import numpy as np
from typing import Tuple, Callable, List
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier

# from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import math


class FeatureSelection:
    @staticmethod
    def get_problem(x: np.ndarray, y: np.ndarray) -> tuple:
        return FeatureSelection.get_params(x, y)

    def get_params(x, y) -> tuple[int, float, float, float]:
        """
        Get params of a feature selection problem

        Args:
            dimension: the number of dimension default equals to dimension
            function_id: the id of a function

        Returns:
            Returns the number of variables (int), upper bound value (float),  lower_bound value (float), and function treshold of fx (float)

        Raises:
            ValueError: function_id not found
        """
        # check the shape of x
        if x.ndim != 2 and x.shape[0] != 1:
            x = x.reshape(1, -1)
            if x.ndim != 2 and x.shape[0] != 1:
                raise Exception("Shape of the array X is not (1,n_var)")

        if y.ndim != 2 and y.shape[0] != 1:
            y = y.reshape(1, -1)
            if y.ndim != 2 and y.shape[0] != 1:
                raise Exception("Shape of the array y is not (1,n_var)")

        # set n_var to x.shape[1]
        n_var = int(x.shape[1])

        # set ub and lb
        ub = 1
        lb = 0

        # assume that min_val is 0
        min_val = 0

        return n_var, ub, lb, min_val

    def apply_transfer_function(
        x: np.ndarray, function_name: str, num_eva: int = None, max_num_eva: int = None
    ):
        # Compute the transfer function based on the selected type
        match (function_name):
            case "s_shaped_1":
                """Standard sigmoid function"""
                x_transformed = 1 / (1 + np.exp(-2 * x))
            case "s_shaped_2":
                """Hyperbolic tangent function"""
                x_transformed = 1 / (1 + np.exp(-x))
            case "s_shaped_3":
                """Modified sigmoid function"""
                x_transformed = 1 / (1 + np.exp(-x / 2))
            case "v_shaped_1":
                x_transformed = abs(2 / np.pi * np.arctan(math.pi / 2 * x))
            case "v_shaped_2":
                """Modified absolute value function"""
                x_transformed = abs(np.tanh(x))
            case "v_shaped_3":
                """Scaled absolute value function"""
                x_transformed = np.abs(x)
            case "time_varying":
                if num_eva is None or max_num_eva is None:
                    raise ValueError(
                        "num_eva and max_num_eva cannot be empty when using time-varying method"
                    )
                # TODO
            case _:
                raise ValueError(
                    f"Transfer function with the name '{function_name}' is not recognized"
                )

        # x_binary = (x_transformed >= 0.5).astype(int)

        # # fix the shape of x_binary
        # if x_binary.ndim != 1:
        #     x_binary = x_binary.flatten()

        return x_transformed

    def evaluate(
        x_input: np.ndarray,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        alpha: 0.5,
    ) -> float:
        """
        Evaluate feature selection solution

        Args:
            x_input: Solution vector
            x_train: Training data
            x_test: Test data
            y_train: Training labels
            y_test: Test labels
            alpha: metric coefficient. Default: 0.5

        Returns:
            float: Evaluation score (lower is better)
        """
        # convert to binary: binarization standard

        # random value
        treshold = np.random.rand()
        binary_solution = (x_input >= 0.5).astype(int)

        # fix the shape of binary_solution
        if binary_solution.ndim != 1:
            binary_solution = binary_solution.flatten()

        # make sure that its binary
        if not np.array_equal(binary_solution, binary_solution.astype(bool)):
            raise ValueError("binary_solution must be a binary array (only 0s and 1s).")

        # Ensure proper shape for feature selection
        if x_train.shape[1] != binary_solution.shape[0]:
            raise ValueError(
                "The number of features in x_train must match the length of binary_solution."
            )

        # Select features based on binary_solution
        selected_features_train = x_train[:, binary_solution == 1]
        selected_features_test = x_test[:, binary_solution == 1]

        # If no features are selected or all the features are selected, return a high penalty score
        if (
            selected_features_train.shape[1] == 0
            or selected_features_train.shape[1] == x_train.shape[1]
        ):
            return 1.5  # Maximum penalty, as no features selected or all the features all selected

        # if selected_features_train.shape[1] == 0:
        #     return 1.0  # Maximum penalty, as no features selected

        # Initialize Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Fit the model on the training data
        clf.fit(selected_features_train, y_train)

        # Predict on the test data
        y_pred = clf.predict(selected_features_test)

        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        number_selected = selected_features_train.shape[1]
        number_total = x_train.shape[1]

        fx = accuracy + alpha * (1 - (number_selected / number_total))

        fitness_function = 1.5 - fx

        print(fitness_function)

        # Objective: Minimize 1 - accuracy
        return fitness_function
