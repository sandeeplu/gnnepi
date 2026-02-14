
from jarvis.db.jsonutils import loadjson

import json



def extract_targets_preds(file_path):

    """Extract target and predicted values from a JSON file."""

    data = loadjson(file_path)

    x, y = [], []

    

    for entry in data:

        target = entry.get('target_out', [])

        pred = entry.get('pred_out', [])



        # Handle cases where pred_out or target_out might be a scalar or list

        if isinstance(target, list) and isinstance(pred, list):

            for t, p in zip(target, pred):

                x.append(t)

                y.append(p)

                print(f"{file_path} → target_out: {t}, pred_out: {p}")

        elif isinstance(target, list):

            # pred_out is scalar

            for t in target:

                x.append(t)

                y.append(pred)

                print(f"{file_path} → target_out: {t}, pred_out: {pred}")

        else:

            # both are scalars

            x.append(target)

            y.append(pred)

            print(f"{file_path} → target_out: {target}, pred_out: {pred}")



    print(f"\nTotal entries in x ({file_path}):", len(x))

    print(f"Total entries in y ({file_path}):", len(y))

    print("-" * 60)

    return x, y





# Load and process Train, Val, and Test JSONs

train_x, train_y = extract_targets_preds("Train_results.json")

val_x, val_y = extract_targets_preds("Val_results.json")

test_x, test_y = extract_targets_preds("Test_results.json")


