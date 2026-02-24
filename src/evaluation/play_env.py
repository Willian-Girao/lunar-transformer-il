# -*- coding: utf-8 -*-
"""
"""
def main():
    import os
    import argparse

    from src.evaluation.TestingConfig import TestingConfig
    from src.evaluation.test_loop import play_single_env

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
    play_single_env(test_cfg)

if __name__ == "__main__":
    main()