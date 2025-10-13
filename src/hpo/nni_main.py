import argparse
from nni.experiment import Experiment
from src.utils.json_handling import json_2_dict, json_2_nni_search_space_dict

parser = argparse.ArgumentParser(description='')
parser.add_argument('--search_space_json', type=str, help='Configuration .json file with the hyperparameters and their ranges.')
parser.add_argument('--config_json', type=str, help='')
args = parser.parse_args()

search_space = json_2_nni_search_space_dict(json_file=args.search_space_json)
config_json = json_2_dict(json_file=args.config_json)

# Step 3: Configure the experiment
# --------------------------------
experiment = Experiment('local')

# Configure trial code
# --------------------------------
experiment.config.trial_command = f'python nni_optmin.py --config_json {args.config_json}'
experiment.config.trial_code_directory = '.'

# Configure search space
# --------------------------------
experiment.config.search_space = search_space

# Configure tuning algorithm
# --------------------------------
experiment.config.experiment_name = config_json['experiment_name']
experiment.config.tuner.name = config_json['tuner_name']
experiment.config.tuner.class_args = {
    'optimize_mode': config_json['optimize_mode'],
    'seed': config_json['seed']
}

# Configure how many trials to run
# --------------------------------
# Here we evaluate 10 sets of hyperparameters in total, and concurrently evaluate 2 sets at a time.
experiment.config.max_trial_number = config_json['max_trial_number']
experiment.config.trial_concurrency = config_json['trial_concurrency']

# Step 4: Run the experiment
# --------------------------
# Now the experiment is ready. Choose a port and launch it. (Here we use port 8080.)
# You can use the web portal to view experiment status: http://localhost:8080.
experiment.run(config_json['port'])

# After the experiment is done
# ----------------------------
input('NNI optimization done. Press enter to quit...')
experiment.stop()