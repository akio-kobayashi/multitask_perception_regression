import yaml
import argparse

def update_config_output_csv(config_path: str, output_csv_value: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Ensure 'config' key is handled if present
    if 'config' in config:
        config['config']['output_csv'] = output_csv_value
    else:
        config['output_csv'] = output_csv_value

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Update output_csv field in a YAML config file.")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument('--output_csv_value', type=str, required=True, help="New value for the output_csv field.")
    args = parser.parse_args()

    update_config_output_csv(args.config_path, args.output_csv_value)
