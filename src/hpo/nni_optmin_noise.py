
def main():
    import os
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torch, nni
    import numpy as np
    from src.evaluation.test_loop import test_hpo
    import argparse
    from src.utils.json_handling import json_2_dict
    from src.models.decoder_only_transformer.DecoderTransformer import DecoderTransformer
    from src.models.decoder_only_transformer.TransformerConfig import TransformerConfig
    from src.training.training_loop import train_model
    from src.training.TrainingConfig import TrainingConfig
    from src.evaluation.TestingConfig import TestingConfig
    from src.hpo.custom_fitness_functions import evaluate_lander_performance

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_json', type=str, help='')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_json = json_2_dict(json_file=args.config_json)

    # Hyperparameters to be tuned
    # ---------------------------
    params = {
        "noise_mean": 0.0,
        "noise_std": 0.1
    }

    # Get optimized hyperparameters
    # -----------------------------
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)

    # Load dataset
    # -------------------------------------------
    # We load a custom Dataset subclass. Requires pickle, so weights_only=False is necessary.
    # The file is locally generated and trusted.

    config_json['dataset'] = "gymnasium-ActorCritic-LunarLander-1000-14-os.pt"

    dataset = torch.load(os.path.join(project_root, 'data', 'processed', config_json['dataset']), weights_only=False)
    classes_weights = dataset.get_classes_weights().to(device)

    torch.manual_seed(config_json['seed'])
    train_dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    # Instantiate model
    # -------------------------------------------
    model_cfg_dict = {}
    model_cfg_dict['nb_actions'] = 4
    model_cfg_dict['state_space_dim'] = 8
    model_cfg_dict['training_seq_len'] = 14
    model_cfg_dict['token_types'] = 2
    model_cfg_dict['depth'] = 1
    model_cfg_dict['num_attention_heads'] = 2
    model_cfg_dict['embedding_dim'] = 64
    model_cfg_dict['intermediate_dim'] = 128
    model_cfg_dict['hidden_dropout_prob'] = 0.157674
    model_cfg_dict['lr'] = 0.000793
    model_cfg_dict['epochs'] = 60
    model_cfg_dict['noise_type'] = "normal"
    model_cfg_dict['noise_mean'] = params["noise_mean"]
    model_cfg_dict['noise_std'] = params["noise_std"]
    
    model_cfg = TransformerConfig()
    model_cfg.from_dict(dict=model_cfg_dict)
    model_cfg.seed = config_json['seed']

    model = DecoderTransformer(model_cfg).to(device)

    # Set criterion and optimizer
    # -------------------------------------------
    criterion = torch.nn.CrossEntropyLoss(weight=classes_weights, reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=model_cfg['lr'])

    # Load the training loop configuration file
    # -------------------------------------------
    train_cfg = TrainingConfig()
    train_cfg.from_json(json_file=os.path.join(project_root, 'configs', args.config_json))

    train_cfg['seed'] = config_json['seed']
    train_cfg['learning_rate'] = model_cfg['lr']
    train_cfg['batch_size'] = 64
    train_cfg['epochs'] = model_cfg['epochs']
    train_cfg['device'] = device

    # Run training loop
    # -------------------------------------------
    model_id = train_model(
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
        save_checkpoint=False
    )

    # Test model
    # -------------------------------------------
    test_dict = {
        "seed": config_json['seed'],
        "model_id": model_id,
        'device': device,
        "nb_test_episodes": config_json['nb_test_episodes'],
        "save_animation": False,
        "env_noise": [config_json['env_noise'], ""],
        "sequence_length": 12,
        "reward_per_episode": config_json['reward_per_episode']
    }
    
    test_cfg = TestingConfig()
    test_cfg.from_dict(dict=test_dict)

    fitnesses = test_hpo(test_cfg, model, dataset)

    # Report to NNI
    # -------------------------------------------
    nni.report_final_result(np.mean(fitnesses))

if __name__ == "__main__":
    import sys

    try:
        main()
    except Exception as e:
        # DO NOT report final result here
        print(e)
        sys.exit(1)