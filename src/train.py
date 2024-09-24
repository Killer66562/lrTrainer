from typing import Any
from xgboost.callback import TrainingCallback
from datetime import datetime, timezone, timedelta

import argparse
import logging
import sklearn.metrics
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
import xgboost
import sklearn
import joblib
import pandas as pd


class CustomCallback(TrainingCallback):
    def __init__(self, iters_per_log: int, max_iters: int) -> None:
        super().__init__()
        self._iters_per_log = iters_per_log
        self._max_iters = max_iters

    def after_iteration(self, model: Any, epoch: int, evals_log: xgboost.callback.Dict[str, xgboost.callback.Dict[str, xgboost.callback.List[float] | xgboost.callback.List[xgboost.callback.Tuple[float]]]]) -> bool:
        if epoch % self._iters_per_log == 0:
            logging.info(f"epoch: {epoch} / {self._max_iters}")
        return False


def main():
    '''
    Default: read datasets from /mnt/datasets.
    These files should exist.
    + /mnt/datasets/x_train.csv
    + /mnt/datasets/y_train.csv
    + /mnt/datasets/x_test.csv
    + /mnt/datasets/y_test.csv
    '''
    # Training settings
    parser = argparse.ArgumentParser(description="KNN")
    parser.add_argument("--rs", type=int, default=1, metavar="RS",
                        help="random state (default: 1)")
    parser.add_argument("--it", type=int, default=10000, metavar="IT",
                        help="max_iterator (default: 10000)")
    parser.add_argument("--booster", type=str, choices=["gbtree", "gblinear", "dart"], default="gbtree", 
                        help="Choose the booster", metavar="B")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "cuda"], default="cuda", 
                        help="Choose the device", metavar="DEV")
    parser.add_argument("--x_train_path", type=str, default="datasets/x_train.csv",
                        help="Assign the path of x_train_csv", metavar="x-train-path")
    parser.add_argument("--x_test_path", type=str, default="datasets/x_test.csv",
                        help="Assign the path of x_train_csv", metavar="x-test-path")
    parser.add_argument("--y_train_path", type=str, default="datasets/y_train.csv",
                        help="Assign the path of x_train_csv", metavar="y-train-path")
    parser.add_argument("--y_test_path", type=str, default="datasets/y_test.csv",
                        help="Assign the path of x_train_csv", metavar="y-test-path")
    parser.add_argument("--save_model", type=bool, default=False, 
                        help="Save model or not", metavar="save-model")
    parser.add_argument("--model_folder_path", type=str, default="models", 
                        help="The folder to save the model", metavar="model-folder-path")

    args = parser.parse_args()

    # Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics.
    # If log_path is empty print log to StdOut, otherwise print log to the file.
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.DEBUG
    )
    
    x_train_df = pd.read_csv(args.x_train_path)
    y_train_df = pd.read_csv(args.y_train_path)
    x_test_df = pd.read_csv(args.x_test_path)
    y_test_df = pd.read_csv(args.y_test_path)

    model = LogisticRegression(
        random_state = args.rs,
        max_iter=args.it
    )
    model.fit(x_train_df.values, y_train_df.values)
    
    logging.info("Done!")

    y_pred = model.predict(x_test_df.values)
    accuracy = sklearn.metrics.accuracy_score(y_test_df.values, y_pred)

    logging.info(f"accuracy={accuracy}")

    if args.save_model is True:
        rs_str = "rs-" + str(args.rs)
        it_str = "it-" + str(args.it)
        booster_str = "booster-" + str(args.booster)
        current_time_str = datetime.now(tz=timezone(offset=timedelta(hours=8))).strftime("%Y-%m-%d-%H-%M-%S")
        str_list = [rs_str, it_str, booster_str, current_time_str]
        model_name = "_".join(str_list) + ".model"
        model_folder_path_processed = str(args.model_folder_path)
        while model_folder_path_processed.endswith("/"):
            model_folder_path_processed = model_folder_path_processed.removesuffix("/")
        model_path = f"{model_folder_path_processed}/{model_name}"

        joblib.dump(model, model_path)

        logging.info(f"model_path={model_path}")

if __name__ == '__main__':
    main()