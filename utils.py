import os
import pickle
import pandas as pd


def store_classification_results(algorithm_name: str, f1_train: float, f1_test: float, model_path: str,
                                 results_file_path="data/results.csv"):
    results_df = None

    try:
        results_df = pd.read_csv(results_file_path)
    except FileNotFoundError:
        pass

    # if there is no results_df, create one
    if results_df is None:
        results_df = pd.DataFrame(columns=['algorithm', 'f1_train', 'f1_test', 'model_path'])

    # check if there is already a result entry for this algorithm
    if algorithm_name in results_df['algorithm'].values:
        results_df.loc[results_df['algorithm'] == algorithm_name, ['f1_train', 'f1_test', 'model_path']] = [f1_train, f1_test, model_path]
    else:
        results_df = results_df.append({
            'algorithm': algorithm_name,
            'f1_train': f1_train,
            'f1_test': f1_test,
            'model_path': model_path
        }, ignore_index=True)

    results_df.to_csv(results_file_path)


def get_classification_results(results_file_path="data/results.csv"):
    try:
        return pd.read_csv(results_file_path)
    except FileNotFoundError:
        pass


def print_classification_results(results_file_path="data/results.csv"):
    results_df = get_classification_results(results_file_path)

    if results_df is not None:
        print(results_df.to_string())
