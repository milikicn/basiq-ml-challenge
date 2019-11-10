import os
import pickle
import pandas as pd


def store_classification_results(algorithm_name: str, f1_train: float, f1_test: float):
    # path to the pickled df with results
    results_file_path = "data/results.pickle"

    results_df = None

    # check if the file exists (first time it wont)
    if os.path.isfile(results_file_path):
        with open(results_file_path, "rb") as f:
            try:
                results_df = pickle.load(f)
            except Exception:
                pass

    # if there is no results_df, create one
    if results_df is None:
        results_df = pd.DataFrame(columns=['Algorithm', 'F1 on training', 'F1 on test'])

    # check if there is already a result entry for this algorithm
    if algorithm_name in results_df['Algorithm'].values:
        results_df.loc[results_df['Algorithm'] == algorithm_name] = [algorithm_name, f1_train, f1_test]
    else:
        results_df = results_df.append({
            'Algorithm': algorithm_name,
            'F1 on training': f1_train,
            'F1 on test': f1_test
        }, ignore_index=True)

    with open(results_file_path, "wb") as f:
        pickle.dump(results_df, f)


def print_classification_results():
    # path to the pickled df with results
    results_file_path = "data/results.pickle"

    # check if the file exists (first time it wont)
    if os.path.isfile(results_file_path):
        with open(results_file_path, "rb") as f:
            try:
                results_df = pickle.load(f)
                print(results_df.to_string())
            except Exception:
                pass
