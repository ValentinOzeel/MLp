import yaml 
import os


def set_config(config_key, new_dict_params):
    # Load the YAML content from the file
    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yml'))
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)

    if config_key not in yaml_content.keys():
        raise ValueError('The specified config_key args[0] does not exist in config.yml')
    
    # Modify the matching keys with the new value
    for key, value in new_dict_params.items():
        if key.upper() in yaml_content[config_key]:
            yaml_content[config_key][key] = value

    # Save the updated YAML content back to the file
    with open(yaml_file_path, 'w') as file:
        yaml.dump(yaml_content, file, default_flow_style=False)
    return


def get_config():
    # Load the YAML content from the file
    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.yml'))
    with open(yaml_file_path, 'r') as file:
        yaml_content = yaml.safe_load(file)

        return yaml_content
    
