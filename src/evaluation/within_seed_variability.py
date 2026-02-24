# -*- coding: utf-8 -*-
"""
"""
def main():
    import os
    import argparse
    from tqdm import tqdm

    from src.evaluation.TestingConfig import TestingConfig
    from src.evaluation.test_loop import test_within_seed_var

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_json', type=str, help='Configuration .json file describing testing hyperparameters.')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # Load the test loop configuration file
    # -------------------------------------------
    test_cfg = TestingConfig()
    test_cfg.from_json(json_file=os.path.join(project_root, 'configs', args.test_json))

    # Test model
    # -------------------------------------------
    test_iter = range(test_cfg.nb_seeds)
    progress = tqdm(test_iter, desc=f'across seed variation (model ID {test_cfg.model_id})', unit='')

    for s in progress:
        test_cfg.env_seed = test_cfg.starting_seed
        test_within_seed_var(test_cfg)

        test_cfg.starting_seed += 1

if __name__ == "__main__":
    main()