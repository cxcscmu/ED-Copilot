from dataclasses import dataclass
from typing import List
import torch
from copy import deepcopy
from transformers import AutoTokenizer

import os

@dataclass
class Observation:
    # encoded input
    input_ids: torch.tensor
    # attention mask for the input
    attention_mask: torch.tensor
    # current triage textual feature
    current_text: str
    # list of actions
    action_history: List[str]

    def to_dict(self): 
        dict_obs = {
            "input_ids": self.input_ids.numpy().flatten(),
            "attention_mask": self.attention_mask.numpy().flatten()
        }
        return dict_obs

    def update(self,next_feature_name,next_feature_value,tokenizer):    

        new_action_history = deepcopy(self.action_history)
        new_action_history.append(next_feature_name)

        next_lab_text = ' | '.join([h + ' : ' + str(c) for h, c in zip(next_feature_name, next_feature_value)])
        total_text = self.current_text + next_lab_text + tokenizer.eos_token
        
        tokenized_dict = tokenizer(total_text,
                                padding="max_length",
                                max_length=656,
                                return_tensors="pt",
                                return_attention_mask=True,
                                add_special_tokens=False,
                                truncation=True)      
              
        obs = Observation(input_ids=tokenized_dict.input_ids,
                          attention_mask=tokenized_dict.attention_mask,
                          current_text=total_text,
                          action_history=new_action_history)
        return obs
    

    
    @ classmethod
    def init_from_sample(cls,header,row,tokenizer):
        ehr_text = ' | '.join([h + ' : ' + str(c) for h, c in zip(header, row) if (type(c) == str and str(c) != "nan") or (str(c) != "nan" and float(c))]) + tokenizer.eos_token
        tokenized_dict = tokenizer(ehr_text,
                                   padding="max_length",
                                   max_length=656,
                                   return_tensors="pt",
                                   return_attention_mask=True,
                                   add_special_tokens=False,
                                   truncation=True)

        obs = Observation(input_ids=tokenized_dict.input_ids,
                          attention_mask=tokenized_dict.attention_mask,
                          current_text=ehr_text,
                          action_history=[])
        return obs
    
#Unit test the class of observation
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m' )   
    tokenizer.pad_token = "<|padding|>"
    obs = Observation.init_from_sample(
        ["temperature","heartrate"],
        [1,0],
        tokenizer=tokenizer,
    )
    obs = obs.update("bmi,sofa,sirs,insurance".split(","),[-0.4988,0.7585,-0.9846,-0.20255],tokenizer)
    embed()