import numpy as np
from cleandiffuser.dataset.feature_selection_dataset import FeatureSelectionTDDataset

def extract_transitions(transition_file, dataset_file, was_normalized=True):
    
    transitions = np.load(transition_file)
    print(f"Loaded {transitions.shape[0]} transitions, each with {transitions.shape[1]} dimensions")

    
    dataset = np.load(dataset_file, allow_pickle=True).item()
    obs = transitions[:, :60]  
    act_one_hot = transitions[:, 60:120]  
    act = np.argmax(act_one_hot, axis=-1).astype(np.int32)  
    rew = transitions[:, 120]  
    next_obs = transitions[:, 121:181]  
    tml = transitions[:, 181]  

    
    if was_normalized:
        
        mean = np.mean(dataset["observations"], axis=0)
        std = np.std(dataset["observations"], axis=0) + 1e-6
        obs = obs * std + mean
        next_obs = next_obs * std + mean
        
        obs = np.round(obs).clip(0, 1).astype(np.int32)
        next_obs = np.round(next_obs).clip(0, 1).astype(np.int32)
    else:
        
        obs = np.round(obs).clip(0, 1).astype(np.int32)
        next_obs = np.round(next_obs).clip(0, 1).astype(np.int32)

    
    tml = np.round(tml).clip(0, 1).astype(np.int32)  
    
    
    

    
    extracted = {
        "observations": obs,
        "actions": act,
        "rewards": rew,
        "next_observations": next_obs,
        "terminals": tml
    }

    return extracted

def save_extracted_transitions(extracted, output_file):
    
    np.save(output_file, extracted)
    print(f"Saved extracted transitions to {output_file}")

if __name__ == "__main__":
    transition_file = "results/synther_feature_selection/feature-selection-v0/extra_transitions.npy"
    dataset_file = "feature_selection_dataset.npy"
    output_file = "results/synther_feature_selection/feature-selection-v0/extracted_transitions.npy"
    
    
    extracted = extract_transitions(transition_file, dataset_file, was_normalized=True)
    
    
    print("\nSample extracted data (first transition):")
    print(f"obs: {extracted['observations'][0]}")  
    print(f"act: {extracted['actions'][0]}")  
    print(f"rew: {extracted['rewards'][0]}")  
    print(f"next_obs: {extracted['next_observations'][0]}")  
    print(f"tml: {extracted['terminals'][0]}")  
    
    
    save_extracted_transitions(extracted, output_file)