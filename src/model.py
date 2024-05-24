from transformers import  AutoModel, logging
logging.set_verbosity_error()
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, base_model_path,baseline):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_path)
        self.baseline = baseline
        # self.selector = nn.Linear(self.base_model.config.hidden_size,12)

        self.selector = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.base_model.config.hidden_size,12)
        )

        # self.classifier = nn.Linear(self.base_model.config.hidden_size,2)

        self.classifier =  nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.base_model.config.hidden_size,2),
        )


    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids,
                            attention_mask=attention_mask)[0]
        batch_size = input_ids.shape[0]  
        if self.baseline:
            sequence_lengths = torch.eq(input_ids, self.base_model.config.pad_token_id).int().argmax(-1) - 1
            last_token_hidden_states = outputs[torch.arange(batch_size), sequence_lengths]
            return self.classifier(last_token_hidden_states)            
        else:
            action_mask = torch.eq(input_ids, self.base_model.config.eos_token_id) 
            label_mask = torch.eq(input_ids, self.base_model.config.pad_token_id).int().argmax(-1)-1
            action_mask[torch.arange(batch_size), label_mask] = False
            action_hidden_states = outputs[action_mask]
            label_hidden_states = outputs[torch.arange(batch_size), label_mask]
            return self.selector(action_hidden_states),self.classifier(label_hidden_states)
        