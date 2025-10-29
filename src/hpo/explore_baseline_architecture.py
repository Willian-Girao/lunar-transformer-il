# -*- coding: utf-8 -*-
"""
...
"""
def main():
    import os, torch
    import argparse
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from src.utils.dataset_handling import subset_dataset
    import numpy as np
    import random

    from src.models.decoder_only_transformer.TransformerConfig import TransformerConfig
    from src.models.decoder_only_transformer.DecoderTransformer import DecoderTransformer
    from src.training.TrainingConfig import TrainingConfig
    from src.training.training_loop import train_model
    from src.utils.json_handling import json_2_dict

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--baseline_architecture_space', type=str, help='...')
    parser.add_argument('--train_json', type=str, help='Configuration .json file describing training hyperparameters.')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # Load the training loop configuration file
    # -------------------------------------------
    train_cfg = TrainingConfig()
    train_cfg.from_json(json_file=os.path.join(project_root, 'configs', args.train_json))

    # Load the architecture search space file
    # -------------------------------------------
    arch_cfg = json_2_dict(json_file=args.baseline_architecture_space)

    # TODO: need to check if every list in `arch_cfg` has the same length.

    # Index each element of lists sequentially for one network setup
    # -------------------------------------------
    nb_arch_cfgs = len(arch_cfg['depth'])
    for idx in range(nb_arch_cfgs):
        # Load the transformer's configuration file
        # -------------------------------------------
        model_cfg = TransformerConfig()
        model_cfg.depth = arch_cfg['depth'][idx]
        model_cfg.num_attention_heads = arch_cfg['num_attention_heads'][idx]
        model_cfg.embedding_dim = arch_cfg['embedding_dim'][idx]
        model_cfg.intermediate_dim = int(arch_cfg['embedding_dim'][idx]*arch_cfg['scale_embedding_dim'][idx])
        model_cfg.training_seq_len = arch_cfg['training_seq_len']
        model_cfg.hidden_dropout_prob = arch_cfg['hidden_dropout_prob']
    
        # Each arch. config. is tried with multiple seeds
        # -------------------------------------------
        for seed in range(arch_cfg['nb_seeds']):
            if seed <= 6:
                pass
            else:

                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                train_cfg.seed = seed
                model_cfg.seed = seed

                # Load dataset
                # -------------------------------------------
                # We load a custom Dataset subclass. Requires pickle, so weights_only=False is necessary.
                # The file is locally generated and trusted.
                dataset = torch.load(os.path.join(project_root, 'data', 'processed', train_cfg.dataset), weights_only=False)
                classes_weights = dataset.get_classes_weights().to(train_cfg.device)

                if dataset.seq_len != model_cfg.training_seq_len:
                    raise ValueError(f'Dataset sequence length ({dataset.seq_len}) differs from the sequence length set for the transformer ({model_cfg.training_seq_len}).')
                
                # train_cfg.subset_dataset is a list: 1st index (bool), should subset or not; 2nd index (int), desired number of samples
                if train_cfg.subset_dataset[0]:
                    dataset = subset_dataset(dataset=dataset, nb_samples=train_cfg.subset_dataset[1], seed=model_cfg.seed)
                train_dataloader = DataLoader(dataset=dataset, batch_size=train_cfg.batch_size, shuffle=train_cfg.data_shuffle)

                # Instantiate model
                # -------------------------------------------
                transformer = DecoderTransformer(model_cfg).to(train_cfg.device)

                # Set criterion and optimizer
                # -------------------------------------------
                criterion = torch.nn.CrossEntropyLoss(weight=classes_weights, reduction="none")
                optimizer = optim.Adam(transformer.parameters(), lr=train_cfg.learning_rate)

                # Run training loop
                # -------------------------------------------
                train_model(
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                    model=transformer,
                    optimizer=optimizer,
                    criterion=criterion,
                    train_dataloader=train_dataloader,
                )

if __name__ == "__main__":
    main()