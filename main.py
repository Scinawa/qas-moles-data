import configparser
from mlflow.tracking import MlflowClient

from data_structure import data_template
from utils import generate_latex_table, update_data_template, compute_average_metrics


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

    #print(config.sections())

    average_metrics = {}
    # import pdb
    # pdb.set_trace()

    for section in config.sections():
        if section == "DEFAULT":
            continue

        column_id = int(section.split("-")[-1])

        try:        
            average_metrics[column_id] = compute_average_metrics(
                config[section]['experiment_id'],
                config[section]['run_id'],
                int(config[section]['step_number']),
                None,
                config['DEFAULT']['mlflow_data_path'],
            )
        except Exception as e:
            print(f"Failed to compute average metrics for {section}. \n Reason: {e}")
            average_metrics[column_id] = {}

    print("Average Metrics:", average_metrics)

    data = update_data_template(data_template, average_metrics)


    # Generate the table
    latex_code = generate_latex_table(data)
    with open("output_table.tex", "w") as file:
        file.write(latex_code)
    print("LaTeX table written to output_table.tex")
