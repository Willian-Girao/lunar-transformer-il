# -*- coding: utf-8 -*-
"""
"""
def main():
    import os
    import argparse

    from src.evaluation.TestingConfig import TestingConfig
    from src.evaluation.test_loop import test

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
    rewards = test(test_cfg) # return exported to file if flag is set on config.

if __name__ == "__main__":
    main()