import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
import numpy as np


def score_step(average_metrics, weights=None):
    if weights is None:
        weights = {key: 1 for key in average_metrics}

    if average_metrics.keys() != weights.keys():
        raise ValueError("The keys of metrics and weights must be the same.")

    score = sum(average_metrics[key] * weights[key] for key in average_metrics)
    return score


def map_metrics_to_template(metrics):
    # Mapping of metric names to the desired keys
    metric_mapping = {
        "val/fake_np": "n mol",
        "val/fake_qed": "QED",
        "val/fake_logp": "Solub.",
        "val/fake_sas": "SA",
        "val/mean_of_prods": "KL",
    }

    # Create the new dictionary with the mapped values
    mapped_metrics = {}
    for metric_name, metric_value in metrics.items():
        if metric_name in metric_mapping:
            mapped_metrics[metric_mapping[metric_name]] = metric_value

    return mapped_metrics


def update_data_template(data_template, average_metrics):
    # Mapping of average metric names to data template keys
    metric_mapping = {
        "val/fake_np": "n mol",
        "val/fake_qed": "QED",
        "val/fake_logp": "Solub.",
        "val/fake_sas": "SA",
        # "val/mean_of_prods": "KL",
    }

    for base in ["As-moles", "qAs-moles"]:

        # Iterate over the z values (2, 3, 4)
        for z in [2, 3, 4]:
            # Construct the key for the current z value
            key = f"{base}-{z}"

            # Get the average metrics for the current key
            metrics = average_metrics.get(key, {})

            # Update the data template with the average metrics
            for metric_name, metric_value in metrics.items():
                if metric_name in metric_mapping:
                    data_template[z]["new"]["As-moles"][
                        metric_mapping[metric_name]
                    ] = metric_value
                    data_template[z]["new"]["Qas-Moles"][
                        metric_mapping[metric_name]
                    ] = metric_value

    return data_template


def generate_latex_table(data):
    rows = ["n mol", "QED", "Solub.", "SA", "KL"]  # Row names
    z_values = [2, 3, 4]  # Big columns
    old_headers = ["QmolGan", "MolGan", "p-value"]
    new_headers = ["As-moles", "Qas-Moles", "p-value"]

    # Start building the LaTeX table
    latex_table = r"""
    \begin{table*}[]
    \resizebox{\columnwidth}{!}{
       \begin{tabular}{ccccccc|cccccc|cccccc}
        & \multicolumn{6}{c|}{z = 2}                                                       & \multicolumn{6}{c|}{z = 3}                                                       & \multicolumn{6}{c}{z = 4}                                                        \\ \cline{2-19} 
        & QMG & MG & \multicolumn{1}{c|}{p-value} & ASM & QSM & p-value & QMG & MG & \multicolumn{1}{c|}{p-value} & ASM & QSM & p-value & QMG & MG & \multicolumn{1}{c|}{p-value} & ASM & QSM & p-value \\
    """

    # Fill rows with data
    for row_name in rows:
        latex_table += f"{row_name} & "  # Row label
        for z in z_values:
            for part in ["old", "new"]:
                headers = old_headers if part == "old" else new_headers
                for header in headers:
                    # Fetch the data or leave blank if missing
                    value = (
                        data.get(z, {}).get(part, {}).get(header, {}).get(row_name, "")
                    )

                    if isinstance(value, float):
                        value = f"{value:.3f}"

                    latex_table += f"\({value}\) & "
        latex_table = latex_table.rstrip("& ") + r" \\" + "\n"  # End the row

    # Close the LaTeX table
    latex_table += r"""
    \end{tabular}}
    \end{table*}
    
    """
    return latex_table


def compute_average_metrics_of_a_step(client, run, step_number):
    """
    Computes the average of specified metrics for a given experiment ID, run ID, and step number.

    Returns:
        dict: A dictionary where keys are metric names and values are their averages for the given step.
    """
    # mlflow.set_tracking_uri(mlflow_data_path)

    # client = MlflowClient()

    # # Validate the run belongs to the experiment
    # run = client.get_run(run_id)
    # if run.info.experiment_id != experiment_id:
    #     raise ValueError(
    #         "The specified run ID does not belong to the given experiment ID."
    #     )

    # Fetch all logged metrics for the run
    averages = {}

    for metric_name in run.data.metrics.keys():
        if metric_name == "epoch":
            continue

        try:
            # Get all metric history for the given metric name
            metric_history = client.get_metric_history(run.info.run_id, metric_name)
            # Filter metrics by the specified step (step number)
            step_metrics = [
                metric.value for metric in metric_history if metric.step == step_number
            ]

            if len(step_metrics) == 0:
                # print(
                #    f"E: No values found for metric '{metric_name}' at step {step_number}"
                # )
                averages[metric_name] = (
                    -1  # No values logged for this metric at the step
                )
            elif len(step_metrics) > 1:
                print(
                    f"E: Multiple values found. {len(step_metrics)} values found for metric '{metric_name}' at step {step_number}"
                )
                averages[metric_name] = step_metrics[
                    0
                ]  # Just return the first value for now
            else:
                averages[metric_name] = step_metrics[0]  # Just return the value for now

        except Exception as e:
            print(f"E: Error computing average for metric '{metric_name}': {e}")

    return averages

    # # Get all metric history for the given metric name
    # metric_history = client.get_metric_history(run_id, metric_name)
    # # Filter metrics by the specified step (step number)
    # step_metrics = [
    #     metric.value for metric in metric_history if metric.step == step_number
    # ]

    # if len(step_metrics) == 0:
    #     print(
    #         f"E: No values found for metric '{metric_name}' at step {step_number}"
    #     )
    # if len(step_metrics) > 1:
    #     print(
    #         f"E: Multiple values found. {len(step_metrics)} values found for metric '{metric_name}' at step {step_number}"
    #     )

    # try:
    #     if step_metrics:
    #         averages[metric_name] = step_metrics[0]  # Just return the value for now
    #     else:
    #         averages[metric_name] = (
    #             None  # No values logged for this metric at the step
    #         )
    # except Exception as e:
    #     print(f"E: Error computing average for metric '{metric_name}': {e}")
