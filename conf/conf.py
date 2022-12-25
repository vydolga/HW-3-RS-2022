#for logging and dynaconf
import logging
import os
from dynaconf import Dynaconf
#specifying logging level
logging.basicConfig(level=logging.INFO)


current_directory = os.path.dirname(os.path.realpath(__file__))
print(current_directory)
settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=f'{current_directory}/setting.toml')
path_to_model = "model/conf/decision_tree.pkl"
