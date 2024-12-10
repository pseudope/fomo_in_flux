from pathlib import Path
import numpy as np
import wandb
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--xid', type=str, default='y9ggpoma', help='W&B xid to compute results for')
    parser.add_argument('--project', type=str, default='continualfomo_finetune', help='W&B project where run is stored')
    parser.add_argument('--entity', type=str, default='codis', help='W&B entity where project is stored')
    parser.add_argument('--print-last-only', type=bool, default=True, help='Print only final KA and ZS values')
    args = parser.parse_args()

    # Login to your W&B account
    wandb.login()

    # Initialize W&B API
    api = wandb.Api()

    # Define the project and the run name you are looking for
    project = args.project
    entity = args.entity

    NUM_RET_EVAL = 2
    NUM_CLS_EVAL = 20

    NUM_RET_ADAPT = 1
    NUM_CLS_ADAPT = 40

    run = api.run(f"{entity}/{project}/{args.xid}")
    filtered_runs = [run]

    zero_shot_point = {"KA": -1, "ZS": -1, "GM": -1}
    method_point = {"KA": [], "ZS": [], "GM": []}

    if not filtered_runs:
        print("No matching runs found.")
    else:
        for index, run in enumerate(filtered_runs):
            # Print run details
            print(f"Run ID: {run.id}")
            print(f"Run Name: {run.name}")
            print(f"Run Status: {run.state}")

            metrics = run.history(
                keys=[
                    "adaptation.alldata-evalonly.total.accuracy",
                    "adaptation.alldata-trainonly.total.accuracy",
                    "adaptation.alldata-evalonly.total.recall@5",
                    "adaptation.alldata-trainonly.total.recall@5",
                ]
            )

            # Convert metrics to list of dictionaries
            metrics_list = metrics.to_dict(orient="records")

            # Get zero-shot point (first entry in metrics)
            if metrics_list:
                zero_shot_point["KA"] = NUM_CLS_ADAPT * metrics_list[0].get(
                    "adaptation.alldata-trainonly.total.accuracy", -1
                ) + NUM_RET_ADAPT * metrics_list[0].get(
                    "adaptation.alldata-trainonly.total.recall@5", -1
                )
                zero_shot_point["KA"] = zero_shot_point["KA"] / (
                    NUM_CLS_ADAPT + NUM_RET_ADAPT
                )

                zero_shot_point["ZS"] = NUM_CLS_EVAL * metrics_list[0].get(
                    "adaptation.alldata-evalonly.total.accuracy", -1
                ) + NUM_RET_EVAL * metrics_list[0].get(
                    "adaptation.alldata-evalonly.total.recall@5", -1
                )
                zero_shot_point["ZS"] = zero_shot_point["ZS"] / (
                    NUM_CLS_EVAL + NUM_RET_EVAL
                )

                zero_shot_point["GM"] = np.sqrt( zero_shot_point["ZS"] * zero_shot_point["KA"] )

            # Populate adaptation and evaluation points for each stream
            for metric in metrics_list:
                adapt_acc = NUM_CLS_ADAPT * metric.get(
                    "adaptation.alldata-trainonly.total.accuracy", -1
                ) + NUM_RET_ADAPT * metric.get(
                    "adaptation.alldata-trainonly.total.recall@5", -1
                )
                adapt_acc = adapt_acc / (NUM_CLS_ADAPT + NUM_RET_ADAPT)
                eval_acc = NUM_CLS_EVAL * metric.get(
                    "adaptation.alldata-evalonly.total.accuracy", -1
                ) + NUM_RET_EVAL * metric.get(
                    "adaptation.alldata-evalonly.total.recall@5", -1
                )
                eval_acc = eval_acc / (NUM_CLS_EVAL + NUM_RET_EVAL)
                method_point["KA"].append(adapt_acc)
                method_point["ZS"].append(eval_acc)
                method_point["GM"].append(np.sqrt(adapt_acc * eval_acc))

    # Print the collected data points
    print("Zero-shot points:", zero_shot_point)
    if args.print_last_only:
        print("Method points:", {"KA": method_point["KA"][-1], "ZS": method_point["ZS"][-1], "GM": method_point["GM"][-1]})
