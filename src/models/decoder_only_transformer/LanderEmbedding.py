import torch.nn as nn
import torch

class LanderEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        torch.manual_seed(config.seed)

        if config.training_seq_len % 2 != 0:
            raise ValueError(f'The argument config.training_seq_len={config.training_seq_len} has to be even.')

        # By default, `torch.arange`, `torch.zeros_like`, and `torch.ones_like` create CPU tensors, so we
        # need to know the device to move them there in the forward pass.
        self.device = config.device
        self.training_seq_len = config.training_seq_len-1

        self.state_embedding = nn.Linear(in_features=config.state_space_dim, out_features=config.embedding_dim)
        self.action_embedding = nn.Embedding(num_embeddings=config.nb_actions, embedding_dim=config.embedding_dim)

        self.position_embedding = None
        if config.position_embedding:
            self.position_embedding = nn.Embedding(num_embeddings=self.training_seq_len, embedding_dim=config.embedding_dim)

        self.type_embd_layer = None
        if config.type_embd_layer:
            self.type_embd_layer = nn.Embedding(num_embeddings=self.training_seq_len, embedding_dim=config.embedding_dim)

    def forward(self, states_seq, actions_seq):
        """
        states_seq (batch_size, seq_len, feat_dim)
        actions_seq (batch_size, seq_len)
        """
        
        if self.training and actions_seq.size(1) != states_seq.size(1)-1:
            raise ValueError(
                f'An input sequence should be N (got {states_seq.size(1)}) sequential state vectors interleaved with N-1 sequential actions (got {actions_seq.size(1)}).'
            )

        # --- Creating Raw Embeddings ---
        
        stt_embds = self.state_embedding(states_seq)
        act_embds = self.action_embedding(actions_seq)

        # we'll alternate between state/action to form an input seq.
        seq_len = stt_embds.size(-2) + act_embds.size(-2)

        if self.training and seq_len != self.training_seq_len:
            raise ValueError(f'Computed seq_len={seq_len} differs from training_seq_len={self.training_seq_len}.')

        if self.position_embedding is not None:
            # --- Creating Positional Embeddings ---
            position_ids = torch.arange(self.training_seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
            pos_embds = self.position_embedding(position_ids)

        if self.type_embd_layer is not None:
            # --- Creating Type Embeddings ---
            stt_type_embds = self.type_embd_layer(torch.zeros_like(stt_embds[:, :, 0], dtype=torch.long, device=self.device))
            act_type_embds = self.type_embd_layer(torch.ones_like(act_embds[:, :, 0], dtype=torch.long, device=self.device)) 

        if self.position_embedding  is not None and self.type_embd_layer is not None:
            # --- Combining Embeddings (raw + positional + type) ---
            stt_embds += (pos_embds[:, :stt_embds.size(-2)*2:2, :] + stt_type_embds) # states get pos_embeds indices 0, 2, 4, ...
            act_embds += (pos_embds[:, 1:act_embds.size(-2)*2:2, :] + act_type_embds) # actions get pos_embeds indices 1, 3, 5, ...

        # --- Stacking to Create Input Sequence ---

        B, S, D = stt_embds.shape   # (batch, num_states, dim)
        _, A, _ = act_embds.shape   # (batch, num_actions, dim)

        embedded_seq = stt_embds.new_zeros((B, S + A, D)) # room for interleaving: (batch, num_states + num_actions, dim)

        embedded_seq[:, 0::2, :] = stt_embds # even positions with states
        embedded_seq[:, 1::2, :] = act_embds # odd positions with states

        # Stack doesn't work because `act_embds` has one less element in its 2nd dimension by design
        # so the sequence has to be interleaved manually.
        # torch.stack([stt_embds, act_embds], dim=2).view(stt_embds.size(0), -1, act_embds.size(-1))

        return embedded_seq