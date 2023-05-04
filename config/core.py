import yaml
import os
params_file = os.path.join(os.getcwd(),'params.yaml')


class read_params_file:
    def __init__(self):
        self.config_path = params_file

    def params_file(self):
        with open(self.config_path,'rb') as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config 
    
config = read_params_file().params_file()
