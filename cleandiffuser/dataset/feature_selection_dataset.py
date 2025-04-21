from cleandiffuser.dataset.base_dataset import BaseDataset
import numpy as np
import torch

class FeatureSelectionTDDataset(BaseDataset):
    def __init__(self, dataset, reward_tune="none"):
        super().__init__()
        observations, actions, next_observations, rewards, terminals = (
            dataset["observations"].astype(np.float32),
            dataset["actions"].astype(np.int32),
            dataset["next_observations"].astype(np.float32),
            dataset["rewards"].astype(np.float32),
            dataset["terminals"].astype(np.float32)
        )
        
        self.normalizers = {"state": None}
        self.obs = torch.tensor(observations)  
        self.act = torch.tensor(actions, dtype=torch.long)
        self.rew = torch.tensor(rewards)[:, None]
        self.tml = torch.tensor(terminals)[:, None]
        self.next_obs = torch.tensor(next_observations)  
        self.size = self.obs.shape[0]
        self.o_dim, self.a_dim = observations.shape[-1], 1  

    def get_normalizer(self):
        return self.normalizers["state"]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'obs': {'state': self.obs[idx]},
            'next_obs': {'state': self.next_obs[idx]},
            'act': self.act[idx],
            'rew': self.rew[idx],
            'tml': self.tml[idx]
        }

class SynthERFeatureSelectionTDDataset(FeatureSelectionTDDataset):
    def __init__(self, save_path, dataset, reward_tune="none"):
        super().__init__(dataset, reward_tune)
        extra_transitions = np.load(save_path + "extra_transitions.npy")
        extra_observations = extra_transitions[:, :self.o_dim]
        extra_actions = extra_transitions[:, self.o_dim:self.o_dim + 60].argmax(axis=-1).astype(np.int32)
        extra_rewards = extra_transitions[:, self.o_dim + 60]
        extra_next_observations = extra_transitions[:, self.o_dim + 61:self.o_dim + 121]
        extra_terminals = extra_transitions[:, -1]
        
        extra_terminals = np.round(extra_terminals).clip(0, 1).astype(np.float32)
        self.act = torch.tensor(np.concatenate([self.act.numpy(), extra_actions], 0), dtype=torch.long)
        self.rew = torch.tensor(np.concatenate([self.rew.numpy(), extra_rewards[:, None]], 0))
        self.tml = torch.tensor(np.concatenate([self.tml.numpy(), extra_terminals[:, None]], 0))
        self.obs = torch.tensor(np.concatenate([self.obs.numpy(), extra_observations], 0))
        self.next_obs = torch.tensor(np.concatenate([self.next_obs.numpy(), extra_next_observations], 0))
        self.size = self.obs.shape[0]