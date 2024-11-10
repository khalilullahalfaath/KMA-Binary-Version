import numpy as np
from typing import Tuple, Callable, List
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
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

        x_binary = (x_transformed >= 0.5).astype(int)

        return x_binary

    def evaluate(
        binary_solution: np.ndarray, X_data: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Evaluate feature selection solution

        Args:
            binary_solution: Solution vector indicating selected features
            X_data
            y

        Returns:
            float: Evaluation score (lower is better)
        """
        # make sure that its binary
        if not np.array_equal(binary_solution, binary_solution.astype(bool)):
            raise ValueError("binary_solution must be a binary array (only 0s and 1s).")

        # Check if binary_solution is 1D, and reshape if necessary
        if binary_solution.ndim != 1:
            binary_solution = binary_solution.flatten()

        # Ensure proper shape for feature selection
        if X_data.shape[1] != binary_solution.shape[0]:
            raise ValueError(
                "The number of features in X_data must match the length of binary_solution."
            )

        # Ensure proper shape
        # Select features based on binary_solution
        selected_features = X_data[:, binary_solution == 1]

        # If no features are selected, return a high penalty score
        if selected_features.shape[1] == 0:
            return 1.0  # Maximum penalty, as no features selected

        # Initialize Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Evaluate using cross-validation to get accuracy
        accuracy = cross_val_score(
            clf, selected_features, y, cv=5, scoring="accuracy"
        ).mean()

        # Objective: Minimize 1 - accuracy
        return 100 - accuracy
