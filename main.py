import configparser
from mlflow.tracking import MlflowClient
import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
import pprint

from data_structure import data_template
from utils import (
    generate_latex_table,
    update_data_template,
    compute_average_metrics_of_a_step,
    score_step,
    map_metrics_to_template,
)


def load_config(config_file):
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        if not config.sections():
            raise ValueError(f"No sections found in the config file: {config_file}")
    except Exception as e:
        print(f"Failed to load configuration file {config_file}. \n Reason: {e}")
        raise
    return config


if __name__ == "__main__":
    # Load configuration file
    config_file = "config.ini"
    config = load_config(config_file)

    # print(config.sections())

    average_best_metrics = {}
    simplified_best_metrics = {}
    import pdb

    best_steps = {}

    mlflow.set_tracking_uri(config["DEFAULT"]["mlflow_data_path"])
    client = MlflowClient()

    for section in config.sections():
        if section == "DEFAULT":
            continue

        column_id = int(section.split("-")[-1])

        best_step = None
        best_score = float("-inf")

        # Validate the run belongs to the experiment
        run = client.get_run(config[section]["run_id"])

        step_numbers = set()

        # find all steps so we can iterate all the steps later
        for metric in run.data.metrics.keys():
            metric_history = client.get_metric_history(run.info.run_id, metric)
            for metric_data in metric_history:
                step_numbers.add(metric_data.step)

        # find the average metric for each step
        # and in the meanwhile find the best step according to the metric.
        for step in sorted(step_numbers):
            try:
                average_metrics_per_step = compute_average_metrics_of_a_step(
                    client, run, step
                )
                # print("Average metrics per step:", average_metrics_per_step)

            except Exception as e:
                print(
                    f"Failed to compute average metric in {step} in {section}. \n Reason: {e}"
                )

            temporary_dictionary_of_metric_at_step = map_metrics_to_template(
                average_metrics_per_step
            )

            # compute score of the step....
            # weights is just the unit vector for now
            score = score_step(temporary_dictionary_of_metric_at_step)
            if score > best_score:
                best_score = score
                best_step = step
                average_best_metrics[section] = average_metrics_per_step
                simplified_best_metrics[section] = (
                    temporary_dictionary_of_metric_at_step
                )

        best_steps[section] = best_step

        print("Best steps:", best_steps)

    # pdb.set_trace()
    # Update the data template with the average metrics
    print("Average best metrics:")
    pprint.pprint(average_best_metrics)

    print("Average simplified best metrics:")
    pprint.pprint(simplified_best_metrics)

    data = update_data_template(data_template, average_best_metrics)

    # Generate the table
    latex_code = generate_latex_table(data)
    with open("output_table.tex", "w") as file:
        file.write(latex_code)
    print("LaTeX table written to output_table.tex")
