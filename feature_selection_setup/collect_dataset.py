import numpy as np

from cleandiffuser.env.feature_selection_env import FeatureSelectionEnv

def collect_dataset(env, num_transitions=5000):
    dataset = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "terminals": []
    }
    obs = env.reset()
    for _ in range(num_transitions):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        dataset["observations"].append(obs.copy())
        dataset["actions"].append(action)
        dataset["rewards"].append(reward)
        dataset["next_observations"].append(next_obs.copy())
        dataset["terminals"].append(done)
        obs = next_obs
        if done:
            obs = env.reset()
    for key in dataset:
        if key == "actions":
            dataset[key] = np.array(dataset[key], dtype=np.int32)
        else:
            dataset[key] = np.array(dataset[key], dtype=np.float32)
    return dataset

if __name__ == "__main__":
    env = FeatureSelectionEnv()
    dataset = collect_dataset(env)
    np.save("feature_selection_dataset.npy", dataset)