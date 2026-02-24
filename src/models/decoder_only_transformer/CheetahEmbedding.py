import torch.nn as nn
import torch
from src.models.decoder_only_transformer.LanderEmbedding import LanderEmbedding

class CheetahEmbedding(LanderEmbedding):
    def __init__(self, config):
        super().__init__(config)
        if config.training_seq_len % 2 != 0:
            raise ValueError(f'The argument config.training_seq_len={config.training_seq_len} has to be even.')

        self.action_embedding = nn.Linear(in_features=config.nb_actions, out_features=config.embedding_dim)

    def forward(self, states_seq, actions_seq):
        """
        states_seq:  (batch_size, seq_len, state_dim)
        actions_seq: (batch_size, seq_len-1, action_dim)
        """

        if self.training and actions_seq.size(1) != states_seq.size(1) - 1:
            raise ValueError(
                f'An input sequence should be N states (got {states_seq.size(1)}) '
                f'interleaved with N-1 actions (got {actions_seq.size(1)}).'
            )

        # --- Raw embeddings ---

        stt_embds = self.state_embedding(states_seq)

        # Continuous projection + bounding
        act_embds = self.action_embedding(actions_seq.float())
        act_embds = torch.tanh(act_embds)   # bound between [-1, 1]

        # --- Interleaving logic identical to parent ---

        seq_len = stt_embds.size(-2) + act_embds.size(-2)

        if self.training and seq_len != self.training_seq_len:
            raise ValueError(
                f'Computed seq_len={seq_len} differs from training_seq_len={self.training_seq_len}.'
            )

        if self.position_embedding is not None:
            position_ids = torch.arange(
                self.training_seq_len,
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)
            pos_embds = self.position_embedding(position_ids)

        if self.type_embd_layer is not None:
            stt_type_embds = self.type_embd_layer(
                torch.zeros_like(stt_embds[:, :, 0], dtype=torch.long, device=self.device)
            )
            act_type_embds = self.type_embd_layer(
                torch.ones_like(act_embds[:, :, 0], dtype=torch.long, device=self.device)
            )

        if self.position_embedding is not None and self.type_embd_layer is not None:
            stt_embds += (
                pos_embds[:, :stt_embds.size(-2)*2:2, :] + stt_type_embds
            )
            act_embds += (
                pos_embds[:, 1:act_embds.size(-2)*2:2, :] + act_type_embds
            )

        # --- Interleave state/action ---

        B, S, D = stt_embds.shape
        _, A, _ = act_embds.shape

        embedded_seq = stt_embds.new_zeros((B, S + A, D))

        embedded_seq[:, 0::2, :] = stt_embds
        embedded_seq[:, 1::2, :] = act_embds

        return embedded_seq