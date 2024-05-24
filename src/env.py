
import numpy as np
from gym import Env, spaces
from gym.spaces.dict import Dict as DictSpace
from gym.spaces.discrete import Discrete
import argparse
from src.observation import Observation
from src.utils import read_table,personalize_cost,ed_lab_idx
from transformers import AutoTokenizer
import pandas as pd
import os



class EDCopilotEnv(Env):
    def __init__(self,dataset,header,args,train = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_input_path)
        self.train = train
        self.task = args.outcome
        # self.tokenizer.pad_token = "<|padding|>"
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.patients_ehr_data = dataset[0][0]
        self.patients_lab_data = dataset[0][1]
        self.patients_label = dataset[1][0]
        self.patients_lab_label = dataset[1][1]
        self.num_patients = len(self.patients_label)
        self.ehr_header = header[0]
        self.lab_header = header[1]
        #Set observation space
        self.observation_space = DictSpace(
            {
                "input_ids": spaces.Box(
                    low=0, high=self.tokenizer.vocab_size, shape=(656,)
                ),
                "attention_mask": spaces.Box(
                    low=0, high=1, shape=(656,)
                )
            }
        )
        
        self.lab_idx_grouped = ed_lab_idx
        self.lab_header_grouped = []
        for group in self.lab_idx_grouped :
            group_header = []
            for i in group:
                group_header.append(self.lab_header[i])
            self.lab_header_grouped.append(group_header)
            
        self.num_lab = len(ed_lab_idx)    
        # num_test + number of labels
        self.action_space = Discrete(n=self.num_lab+2)
        self.invalid_actions = None

        # track the patient
        self.current_patient_ehr_x = None
        self.current_patient_lab_x = None
        self.current_patient_lab_y = None
        self.current_patient_y = None
        self.current_patient_obs = None
        self.current_patient_cost = None
        
        # Set reward
        self.penalty_ratio = args.penalty_ratio
        self.wrong_prediction_penalty = args.wrong_prediction_penalty
        

        # self.off_idx = 0
        self.reset()

    def action_masks(self):
            return [action not in self.invalid_actions for action in range(self.num_lab+2)]
        
        
    def step(self, action):
        if action < self.num_lab:
            self.invalid_actions += [action]

            feature_value = []
            for lab_idx in self.lab_idx_grouped[action]:
                feature_value.append(self.current_patient_lab_x[lab_idx])
            reward = - list(self.current_patient_cost.values())[action]
            # if action in self.current_patient_lab_y:
            self.current_patient_obs = self.current_patient_obs.update(self.lab_header_grouped[action], feature_value,self.tokenizer)

            done = False
        else:
            if not self.current_patient_y : # healthy patient
                reward = - self.wrong_prediction_penalty * (action == (self.num_lab+1))   # prediction penalty of predicting wrongly
            else: # ill patient -> penalize more
                reward = - self.wrong_prediction_penalty * self.penalty_ratio * (action == self.num_lab)
            done = True            

        # populate additional info
        info = {
            "action_history": self.current_patient_obs.action_history,
            "current_text": self.current_patient_obs.current_text
        }

        return self.current_patient_obs.to_dict(), reward, done, info
        
    def reset(self):
        # index = self.off_idx%self.num_patients
        # self.off_idx+=1
        
        # if self.train :
        #     if self.task == "outcome_icu_transfer_12h":
        index = np.random.randint(self.num_patients)
        # index = 67
            # else:
            #     index = self.sampled_indices[index] 
                
        self.current_patient_ehr_x = self.patients_ehr_data[index]
        self.current_patient_lab_x = self.patients_lab_data[index]
        
        self.current_patient_obs = Observation.init_from_sample(
            self.ehr_header,
            self.patients_ehr_data[index],
            self.tokenizer,
        )             
        self.current_patient_y = self.patients_label[index]
        self.current_patient_lab_y = self.patients_lab_label[index]
        # change to free form if unmask
        self.invalid_actions = list(set(range(self.num_lab))-set(self.current_patient_lab_y))
        self.current_patient_cost = personalize_cost(self.current_patient_lab_y)

        # self.invalid_actions = []
        # self.current_patient_cost = personalize_cost(list(range(0,12)))        
        
        return  self.current_patient_obs.to_dict()

    def render(self):
        pass


# # Unit test the class of EDCopilotEnv
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--data_input_path',  type=str,default = '/home/liwens/healthcare/Lightning-Pretrain/copilot_v3/data/abb_lab_los')
#     parser.add_argument('--model_input_path',  type=str, default ="microsoft/biogpt")    
#     parser.add_argument('--outcome',  type=str, default = "outcome_critical") 
#     parser.add_argument('--penalty_ratio', type=int, default=10)
#     parser.add_argument('--wrong_prediction_penalty', type=int, default=100)
    
#     args = parser.parse_args()
#     df_valid = pd.read_csv((os.path.join(args.data_input_path, 'test.csv')))
#     header,valid_table,valid_label = read_table(df_valid,args.outcome)

#     valid_env = EDCopilotEnv((valid_table,valid_label),header,args,False)
#     embed()


