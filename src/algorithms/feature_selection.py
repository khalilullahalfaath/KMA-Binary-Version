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

        # set n_var to y[0].shape
        n_var = int(y[0].shape[0])

        # set ub and lb
        ub = 1
        lb = 0

        # assume that min_val is 0
        min_val = 0

        return n_var, ub, lb, min_val

    def apply_transfer_function(
        x: np.ndarray, function_name: str, num_eva: int = None, max_num_eva: int = None
    ):
        # trimr x based on the selected transfer function
        match (function_name):
            case "s_shaped_1":
                """Standard sigmoid function"""
                return 1 / (1 + np.exp(-2 * x))
            case "s_shaped_2":
                """Hyperbolic tangent function"""
                return 1 / (1 + np.exp(-x))
            case "s_shaped_3":
                """Modified sigmoid function"""
                return 1 / (1 + np.exp(-x / 2))
            case "v_shaped_1":
                return abs(2 / np.pi * np.arctan(math.pi / 2 * x))
            case "v_shaped_2":
                """Modified absolute value function"""
                return abs(np.tanh(x))
            case "v_shaped_3":
                """Scaled absolute value function"""
                return np.abs(x)
            case "time_varying":
                if num_eva == None or max_num_eva == None:
                    raise ValueError(
                        "num_eva and max_num_eva cannot be empty when using time-varying method"
                    )
                """Time-varying transfer function"""

                sigma_max = 1
                sigma_min = 0.1

                sigma = (sigma_max - sigma_min) * (
                    1 - (num_eva / max_num_eva)
                ) + sigma_min

                # Sigmoid functions
                def sigmoid_positive(vd, sigma):
                    return 1 / (1 + np.exp(-sigma * vd))

                def sigmoid_negative(vd, sigma):
                    return 1 / (1 + np.exp(sigma * vd))

                p_positive = sigmoid_positive(x, sigma)
                p_negative = sigmoid_negative(x, sigma)

                # Generate binary positions P_d and P_d' based on the probability thresholds
                Pd = np.where(np.random.rand(*p_positive.shape) < p_positive, 1, 0)

                Pd_prime = np.where(
                    np.random.rand(*p_negative.shape) > p_negative, 1, 0
                )

                # def objective_function(binary_position):
                #     # TODO: Replace with the actual function
                #     return np.sum(binary_position)

                # Evaluate each solution
                fitness_Pd = self.evaluate(Pd)
                fitness_Pd_prime = self.evaluate(Pd_prime)

                # Select the better solution
                if fitness_Pd >= fitness_Pd_prime:
                    return Pd, fitness_Pd
                else:
                    return Pd_prime, fitness_Pd_prime

            case _:
                raise ValueError(
                    f"Transfer function with the name of {function_name} cannot be implemented"
                )

    def evaluate(
        self, binary_solution: np.ndarray, X_data: np.ndarray, y: np.ndarray
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
        return 1 - accuracy
