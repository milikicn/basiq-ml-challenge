import pandas as pd


def store_classification_results(algorithm_name: str, f1_train: float, f1_test: float, model_path: str,
                                 results_file_path="data/results.csv"):
    """Store the results of a classification process into a CSV file."""

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

    results_df.to_csv(results_file_path, index=False)


def get_classification_results(results_file_path="data/results.csv"):
    """Retrieve the file with the classification results."""

    try:
        return pd.read_csv(results_file_path)
    except FileNotFoundError:
        pass


def print_classification_results(results_file_path="data/results.csv"):
    """Print the classification results from the file containing this information."""

    results_df = get_classification_results(results_file_path)

    if results_df is not None:
        print(results_df.to_string())
