# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse

import numpy as np
import openml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder

from BoostedAdaSSP import BoostedAdaSSP

parser = argparse.ArgumentParser(description="BoostedAdaSSP")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument(
    "--num_iterations", type=int, default=100, help="number of iterations"
)
parser.add_argument("--shrinkage", type=str, default="constant")

parser.add_argument("--epsilon", type=float, default=1)
parser.add_argument("--delta", type=float, default=1e-6)

parser.add_argument("--x_bound", type=float, default=1)
parser.add_argument("--y_bound", type=float, default=1)

parser.add_argument("--SUITE_ID", type=int, choices=[297, 299], default=297)

args = parser.parse_args()


def preprocessing_data(X, y, categorical_indicator):
    data = pd.concat((X, y), axis=1)
    data = data.dropna(axis=0, how="any")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    is_cat = np.asarray(categorical_indicator)

    cat_cols = X.columns.values[is_cat]
    num_cols = X.columns.values[~is_cat]

    cat_ohe_step = ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))

    cat_pipe = Pipeline([cat_ohe_step])
    num_pipe = Pipeline([("identity", FunctionTransformer())])
    transformers = [("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)]
    ct = ColumnTransformer(transformers=transformers)

    pipe = Pipeline(
        [
            ("ct", ct),
        ]
    )

    X = pipe.fit_transform(X)

    return X, y


benchmark_suite = openml.study.get_suite(args.SUITE_ID)  # obtain the benchmark suite
for task_id in benchmark_suite.tasks:  # iterate over all tasks
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    X, y = preprocessing_data(X, y, categorical_indicator)
    rng = np.random.RandomState(args.seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = BoostedAdaSSP(
        args.x_bound,
        args.y_bound,
        args.epsilon,
        args.delta,
        args.num_iterations,
        args.shrinkage,
        rng,
    )

    model.fit(X_train, y_train)

    y_score = model.predict(X_test)
    y_score_train = model.predict(X_train)

    print(
        task_id,
        args.seed,
        args.epsilon,
        args.delta,
        args.num_iterations,
        args.shrinkage,
        args.x_bound,
        args.y_bound,
        mean_squared_error(y_train, y_score_train),
        mean_squared_error(y_test, y_score),
    )
