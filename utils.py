import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient


def update_data_template(data_template, average_metrics):
    # Mapping of average metric names to data template keys
    metric_mapping = {
        'Validation_fake_data_nmol': 'n mol',
        'Validation_fake_data_logp': 'Solub.',
        'Validation_fake_data_sas': 'SA',
        'Validation_fake_data_qed': 'QED'
    }

    # Iterate over the z values (2, 3, 4)
    for z in [2, 3, 4]:
        # Get the average metrics for the current z value
        metrics = average_metrics.get(z, {})

        # Update the data template with the average metrics
        for metric_name, metric_value in metrics.items():
            if metric_name in metric_mapping:
                data_template[z]['new']['As-moles'][metric_mapping[metric_name]] = metric_value
                data_template[z]['new']['Qas-Moles'][metric_mapping[metric_name]] = metric_value

    return data_template

def generate_latex_table(data):
    rows = ["n mol", "QED", "Solub.", "SA", "KL"]  # Row names
    z_values = [2, 3, 4]  # Big columns
    old_headers = ["QmolGan", "MolGan", "p-value"]
    new_headers = ["As-moles", "Qas-Moles", "p-value"]

    # Start building the LaTeX table
    latex_table = r"""
    \begin{table}[]
    \begin{tabular}{ccccccc|cccccc|cccccc}
        & \multicolumn{6}{c|}{z = 2}                                                       & \multicolumn{6}{c|}{z = 3}                                                       & \multicolumn{6}{c}{z = 4}                                                        \\ \cline{2-19} 
        & QmolGan & MolGan & \multicolumn{1}{c|}{p-value} & As-moles & Qas-Moles & p-value & QmolGan & MolGan & \multicolumn{1}{c|}{p-value} & As-moles & Qas-Moles & p-value & QmolGan & MolGan & \multicolumn{1}{c|}{p-value} & As-moles & Qas-Moles & p-value \\
    """
    
    # Fill rows with data
    for row_name in rows:
        latex_table += f"{row_name} & "  # Row label
        for z in z_values:
            for part in ["old", "new"]:
                headers = old_headers if part == "old" else new_headers
                for header in headers:
                    # Fetch the data or leave blank if missing
                    value = data.get(z, {}).get(part, {}).get(header, {}).get(row_name, "")

                    if isinstance(value, float):
                        value = f"{value:.3f}"

                    latex_table += f"\({value}\) & "
        latex_table = latex_table.rstrip("& ") + r" \\" + "\n"  # End the row
    
    # Close the LaTeX table
    latex_table += r"""
    \end{tabular}
    \end{table}
    """
    return latex_table


def compute_average_metrics(experiment_id, run_id, step_number, metric_names, mlflow_data_path):
    """
    Computes the average of specified metrics for a given experiment ID, run ID, and step number.
    
    Args:
        experiment_id (str): The MLflow experiment ID.
        run_id (str): The MLflow run ID.
        step_number (int): The step number to filter by.
        metric_names (list): A list of metric names to compute the average for.
    
    Returns:
        dict: A dictionary where keys are metric names and values are their averages for the given step.
    """
    mlflow.set_tracking_uri(mlflow_data_path)

    client = MlflowClient()
    
    # Validate the run belongs to the experiment
    run = client.get_run(run_id)
    if run.info.experiment_id != experiment_id:
        raise ValueError("The specified run ID does not belong to the given experiment ID.")
    

    # Fetch all logged metrics for the run
    averages = {}
    
    for metric_name in run.data.metrics.keys():
        if metric_name == "epoch":
            continue


        # Get all metric history for the given metric name
        metric_history = client.get_metric_history(run_id, metric_name)
        

        # Filter metrics by the specified step (step number)
        step_metrics = [
            metric.value for metric in metric_history
            if metric.step == step_number
        ]

        if len(step_metrics) == 0:
            print(f"E: No values found for metric '{metric_name}' at step {step_number}")
        if len(step_metrics) > 1:
            print(f"E: Multiple values found. {len(step_metrics)} values found for metric '{metric_name}' at step {step_number}")




        try:
            if step_metrics:
                averages[metric_name] = step_metrics[0]  # Just return the value for now
            else:
                averages[metric_name] = None  # No values logged for this metric at the step
        except Exception as e:
            print(f"E: Error computing average for metric '{metric_name}': {e}")

    return averages