import os
from cleandiffuser.dataset.feature_selection_dataset import FeatureSelectionTDDataset
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_diffusion import IDQLMlp
from cleandiffuser.utils import report_parameters
from utils import set_seed

@hydra.main(config_path="../configs/synther/feature_selection", config_name="feature_selection", version_base=None)
def pipeline(args):
    set_seed(args.seed)
    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    os.makedirs(save_path, exist_ok=True)
    
    # Load dataset
    dataset = np.load("feature_selection_dataset.npy", allow_pickle=True).item()
    obs_dim, act_dim = 60, 60  # obs_dim for state, act_dim for one-hot actions
    
    # Diffusion model
    nn_diffusion = IDQLMlp(
        0, obs_dim * 2 + act_dim + 2,  # 60 + 60 + 1 + 60 + 1 = 182
        emb_dim=128,
        hidden_dim=1024,
        n_blocks=6,
        timestep_emb_type="positional"
    )
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")
    synther = DiscreteDiffusionSDE(
        nn_diffusion,
        predict_noise=args.predict_noise,
        optim_params={"lr": args.diffusion_learning_rate},
        diffusion_steps=args.diffusion_steps,
        ema_rate=args.ema_rate,
        device=args.device
    )
    
    if args.mode == "train_diffusion":
        dataset_obj = FeatureSelectionTDDataset(dataset, "none")
        dataloader = DataLoader(
            dataset_obj, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        lr_scheduler = CosineAnnealingLR(synther.optimizer, T_max=args.diffusion_gradient_steps)
        synther.train()
        n_gradient_step = 0
        log = {"avg_diffusion_loss": 0.}
        for batch in loop_dataloader(dataloader):
            obs, act, rew, next_obs, tml = (
                batch["obs"]["state"].to(args.device),
                batch["act"].to(args.device),
                batch["rew"].to(args.device),
                batch["next_obs"]["state"].to(args.device),
                batch["tml"].to(args.device)
            )
            x = torch.cat([obs, F.one_hot(act, num_classes=60).float(), rew, next_obs, tml], -1)
            log["avg_diffusion_loss"] += synther.update(x)["loss"]
            lr_scheduler.step()
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_diffusion_loss"] /= args.log_interval
                print(log)
                log = {"avg_diffusion_loss": 0., "gradient_steps": 0}
            if (n_gradient_step + 1) % args.save_interval == 0:
                synther.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                synther.save(save_path + f"diffusion_ckpt_latest.pt")
            n_gradient_step += 1
            if n_gradient_step >= args.diffusion_gradient_steps:
                break
        print("Diffusion training completed.")
    
    elif args.mode == "dataset_upsampling":
        dataset_obj = FeatureSelectionTDDataset(dataset, "none")
        synther.load(save_path + f"diffusion_ckpt_{args.ckpt}.pt")
        synther.eval()
        ori_size = dataset_obj.obs.shape[0]
        syn_size = 20000 - ori_size
        extra_transitions = []
        prior = torch.zeros((5000, 2 * obs_dim + act_dim + 2)).to(args.device)
        for _ in tqdm(range(syn_size // 5000)):
            syn_transitions, _ = synther.sample(
                prior, solver=args.solver, n_samples=5000, sample_steps=args.sampling_steps, use_ema=args.use_ema)
            extra_transitions.append(syn_transitions.cpu().numpy())
        remaining = syn_size % 5000
        if remaining > 0:
            syn_transitions, _ = synther.sample(
                torch.zeros((remaining, 2 * obs_dim + act_dim + 2)).to(args.device),
                n_samples=remaining, sample_steps=args.sampling_steps, use_ema=args.use_ema, solver=args.solver)
            extra_transitions.append(syn_transitions.cpu().numpy())
        extra_transitions = np.concatenate(extra_transitions, 0)
        np.save(save_path + "extra_transitions.npy", extra_transitions)
        print(f'Synthetic transitions saved to {save_path}extra_transitions.npy')
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    pipeline()