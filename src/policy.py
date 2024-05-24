
from typing import Any, Dict, Optional, Tuple, Type, Union
import argparse
import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    MlpExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
import torch
from sb3_contrib.common.maskable.distributions import MaskableDistribution, make_masked_proba_distribution
from transformers import AutoModel
from sb3_contrib.ppo_mask import MaskablePPO
from src.env import EDCopilotEnv
from src.utils import read_table
import os
import pandas as pd


class MaskableLMActorCriticPolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        model_name_path: str,
        optimizer_kwargs: Dict[str, Any] = {},
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        weight_decay: float = 1e-6,
    ):

        super().__init__(observation_space,action_space,lr_schedule,optimizer_kwargs = optimizer_kwargs)
        # Action distribution
        self.action_dist = make_masked_proba_distribution(action_space)
        self._build_policy_network(model_name_path)
        self._build_optimizer(optimizer_kwargs, weight_decay, optimizer_class)

        
    def _build_optimizer(self, optimizer_kwargs: Dict[str, Any],
                         weight_decay: float, optimizer_class: torch.optim):
        mlp_extractor_params = list(self.mlp_extractor.parameters())
        action_net_params = list(self.action_net.parameters())
        value_net_params = list(self.value_net.parameters())
    
        self.optimizer = optimizer_class(
            mlp_extractor_params+action_net_params+value_net_params, **optimizer_kwargs)

    def _build_policy_network(self,model_name_path) -> None:
        self.base_model = AutoModel.from_pretrained(os.path.join(model_name_path,'base_model'))
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        self.mlp_extractor = MlpExtractor(
            self.base_model.config.hidden_size,
            net_arch = dict(pi=[self.base_model.config.hidden_size]*2, vf=[self.base_model.config.hidden_size]*2),
            activation_fn=nn.SiLU,
            device=self.device,
        )

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        return self.get_distribution(observation, action_masks).get_actions(deterministic=deterministic)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :param action_masks: Action masks to apply to the action distribution
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic, action_masks=action_masks)
            # Convert to numpy
            actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions.squeeze(axis=0)

        return actions, None

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> MaskableDistribution:
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution
    
    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
            )
        )
        return data
    
    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def extract_features(self, obs):
        obs = {k: v.long() for k, v in obs.items()}
        outputs = self.base_model(input_ids=obs["input_ids"],
                            attention_mask=obs["attention_mask"])[0]
        batch_size = obs["input_ids"].shape[0]  
        sequence_lengths =(torch.eq(obs['input_ids'], self.base_model.config.pad_token_id).int().argmax(-1) - 1).to(outputs.device)
        features = outputs[torch.arange(batch_size, device=outputs.device), sequence_lengths] 
        return features


    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def save(self,ckpt_path):
        torch.save(self.mlp_extractor.state_dict(),os.path.join(ckpt_path, 'mlp_extractor.pt'))
        torch.save(self.action_net.state_dict(),os.path.join(ckpt_path, 'action_net.pt'))
        torch.save(self.value_net.state_dict(),os.path.join(ckpt_path, 'value_net.pt'))
        
    def load(self,ckpt_path):
        self.mlp_extractor.load_state_dict(torch.load(os.path.join(ckpt_path, 'mlp_extractor.pt')))  
        self.action_net.load_state_dict(torch.load(os.path.join(ckpt_path, 'action_net.pt')))  
        self.value_net.load_state_dict(torch.load(os.path.join(ckpt_path, 'value_net.pt')))  
    
    
# #Unit test the class of EDCopilotEnv
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--data_input_path',  type=str,default = '/home/liwens/healthcare/Lightning-Pretrain/copilot_v2/data/triage_lab')
#     parser.add_argument('--model_input_path',  type=str, default = '/data/user_data/liwens/clinical_models/Clinical-T5-Base' )    
#     parser.add_argument('--output_path',  type=str, default = '/home/liwens/healthcare/Lightning-Pretrain/copilot_v2/outputs/triage_lab_clinicalt5_reweight')    
#     parser.add_argument('--outcome',  type=str, default = "outcome_critical") 


#     parser.add_argument('--batch_size', type=int, default=256)  

#     parser.add_argument('--buffer_steps', type=int, default=512)
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--ppo_epochs', type=int, default=10)  
#     parser.add_argument('--learning_rate', type=float, default=1e-4)
#     parser.add_argument('--weight_decay', type=float, default=0.01)
#     parser.add_argument('--adam_epsilon', type=float, default=1e-8)
#     parser.add_argument('--max_grad_norm', type=float, default=1.0)  

#     parser.add_argument('--total_timesteps', type=float, default=20000)  
#     parser.add_argument('--penalty_ratio', type=int, default=20)
#     parser.add_argument('--wrong_prediction_penalty', type=int, default=100)
#     parser.add_argument("--random_seed", type=int, default=42)
#     args = parser.parse_args()
    
#     args = parser.parse_args()
#     df_valid = pd.read_csv((os.path.join(args.data_input_path, 'valid.csv')))
#     torch.set_float32_matmul_precision("high")
#     header,valid_table,valid_label = read_table(df_valid,args.outcome)
#     cost = {
#         "outcome_critical":{
#             "VS":5,
#             "RNHX":10,
#             "BMP": 60,
#             "CBC":30,
#         },
#         "hospital_expire_flag":{
#             "CBC":44,
#             "CMP":48,
#             "APTT":473,
#             "ABG":26,
#             "SOFA":0
#         }
#     }

#     valid_env = EDCopilotEnv((valid_table,valid_label),header,cost[args.outcome],args)
#     policy_kwargs = {
#         # "model_name_path":args.model_input_path,
#         "model_name_path":args.output_path,
#         "weight_decay":args.weight_decay,
#         "optimizer_class":torch.optim.AdamW,
#         "optimizer_kwargs":{
#             "eps":args.adam_epsilon
#         }
#     }  
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     model = MaskablePPO(MaskableLMActorCriticPolicy, 
#                         valid_env, 
#                         learning_rate = args.learning_rate, 
#                         n_steps = args.buffer_steps, 
#                         batch_size = args.batch_size,
#                         policy_kwargs=policy_kwargs,
#                         verbose=1,
#                         tensorboard_log="run",
#                         device=device,
#                         max_grad_norm = args.max_grad_norm)

#     model.learn(1000, callback=None, progress_bar = True)

    # Test valid
    #Test save and load:
    # ckpt_path = os.path.join(args.output_path,f'{10}_{100}')
    # os.makedirs(ckpt_path, exist_ok = True)
    # model.policy.save(ckpt_path)
    # model.policy.load(ckpt_path)
