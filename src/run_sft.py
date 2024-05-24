import torch
import numpy as np
from tqdm import tqdm, trange
from transformers import (
    get_linear_schedule_with_warmup
)
from functools import partial
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import pandas as pd
from src.dataset import ClinicalDataset
from src.utils import read_table,convert_to_numpy,auc_score
from transformers import AutoTokenizer
from src.model import Transformer
from sklearn.metrics import f1_score

def ddp_setup(local_rank):
    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

def train(args):
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        ddp_setup(args.local_rank)
        device = torch.device("cuda",args.local_rank)
    args.world_size = torch.cuda.device_count()  
    print(f"Process rank: {args.local_rank}, device: {device}, distributed training: {bool(args.local_rank != -1)}")  
      
    tokenizer = AutoTokenizer.from_pretrained(args.model_input_path)  
    model = Transformer(args.model_input_path,args.baseline).to(device)
    model.base_model.config.pad_token_id = 1 
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)
        
    df_train = pd.read_csv(os.path.join(args.data_input_path, 'train.csv'))
    df_valid = pd.read_csv(os.path.join(args.data_input_path, 'valid.csv'))
    train_header,train_table,train_label = read_table(df_train,args.outcome,True,args.baseline)
    valid_header,valid_table,valid_label = read_table(df_valid,args.outcome,True,args.baseline)
    train_set = ClinicalDataset(train_header,train_table,train_label,tokenizer.eos_token,args.baseline)
    valid_set = ClinicalDataset(valid_header,valid_table,valid_label,tokenizer.eos_token,args.baseline)
    
    train_sampler = RandomSampler(train_set) if args.local_rank == -1 else DistributedSampler(train_set)    
    valid_sampler = SequentialSampler(valid_set) if args.local_rank == -1 else DistributedSampler(valid_set)    
    
    train_dataloader = DataLoader(train_set,sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=partial(train_set.batchify, tokenizer=tokenizer))
    valid_dataloader = DataLoader(valid_set,sampler=valid_sampler, batch_size=args.valid_batch_size,
                                  collate_fn=partial(valid_set.batchify, tokenizer=tokenizer))

    criterion_label = torch.nn.CrossEntropyLoss(weight=torch.tensor([1,args.class_weight]).float().cuda())
    criterion_action = torch.nn.CrossEntropyLoss()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = (len(train_dataloader) * args.epochs)//args.gradient_accumulation_steps
    eval_step = t_total//args.epochs
    warmup_steps = int(args.warmup_percent * t_total)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    
    model.zero_grad()
    train_iterator = trange(int(args.epochs), desc="Epoch")
    global_step = 0
    tr_loss = 0.0
    best_f1 = -1
    for epoch in train_iterator:
        if args.local_rank!=-1:    
            train_dataloader.sampler.set_epoch(epoch)
        epoch_iterator = tqdm(train_dataloader, desc="Training",disable=args.local_rank not in [-1,0])
        for step, (inputs,*targets) in enumerate(epoch_iterator):
            model.train()
            inputs = {k: v.to(device) for k, v in inputs.items()}  
            if args.baseline:
                targets = targets[0].to(device) 
                logits = model(**inputs)
                loss = criterion_label(logits,targets) 
            else:
                action_targets,targets = targets
                targets = targets.to(device) 
                action_targets = action_targets.to(device)
                action_logits,label_logits = model(**inputs)
                action_loss = criterion_action(action_logits, action_targets) 
                label_loss = criterion_label(label_logits, targets) 
                loss = action_loss + label_loss   
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()

            if (step+1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                global_step+=1
                optimizer.step()
                scheduler.step()  
                model.zero_grad()
                if args.local_rank in [-1,0]:
                    epoch_iterator.set_description(f"Loss:{tr_loss/global_step}")
                    
                if global_step and global_step % eval_step == 0:
                    print(f"Rank {args.local_rank} Start validation at global step {global_step}")
                    total_preds,total_targets = validate(args,model, valid_dataloader)
                    if args.local_rank != -1:
                        gather_preds = [torch.ones_like(total_preds) for _ in range(args.world_size)]
                        gather_targets = [torch.ones_like(total_targets) for _ in range(args.world_size)]
                        dist.all_gather(gather_preds, total_preds)
                        dist.all_gather(gather_targets, total_targets)
                        total_preds = torch.cat(gather_preds,dim=0)
                        total_targets =  torch.cat(gather_targets,dim=0)  
                          
                    if args.local_rank in [-1, 0]:
                        total_preds,total_targets = convert_to_numpy(total_preds,total_targets)
                        f1 = f1_score(total_targets, total_preds > 0.5)
                        roc_auc,average_precision,sensitivity,specificity,threshold = auc_score(total_targets,total_preds)
                        results = {
                            "F1":f1,
                            "AUC":roc_auc,
                            "AUPRC":average_precision,
                            "Sensitivity":sensitivity,
                            "Specificity":specificity,
                            "Score Threshold":threshold
                        }
                        print(f"*************** Validation at global step {global_step} Results***************")
                        for key, value in results.items():
                            print(f"{key:25}: {value}")
                        # wandb.log(results)                
                        if f1 > best_f1:
                            base_ckpt_path = os.path.join(args.output_path, 'base_model')
                            classifier_ckpt_path = os.path.join(args.output_path, 'classifier.pt')
                            model_to_save = (model.module if hasattr(
                                model, "module") else model)
                            model_to_save.base_model.save_pretrained(base_ckpt_path)
                            torch.save(model_to_save.classifier.state_dict(), classifier_ckpt_path)
                            print(f"Base Model saved to {base_ckpt_path}")
                            print(f"Classifier Model saved to {classifier_ckpt_path}")
                            best_f1 = f1
                        if args.local_rank == 0:
                            torch.distributed.barrier()
                    else:
                        torch.distributed.barrier()
            
    if args.local_rank != -1:
        destroy_process_group()
        
@torch.no_grad()
def validate(args,model, dataloader):
    model.eval()
    total_preds = []
    total_targets = []
    for inputs,*targets in tqdm(dataloader, desc="Evaluating"):
        inputs = {k: v.cuda() for k, v in inputs.items()}  
        if args.baseline:
            targets = targets[0].cuda()
            logits = model(**inputs)
            logits = torch.nn.functional.softmax(logits,dim=1)
        else:
            action_targets,targets = targets
            targets = targets.cuda()
            action_targets = action_targets.cuda()
            action_logits,label_logits = model(**inputs)
            logits = torch.nn.functional.softmax(label_logits,dim=1)
            
        total_preds.append(logits[:,1])
        total_targets.append(targets)

    total_preds = torch.cat(total_preds,dim=0)
    total_targets = torch.cat(total_targets,dim=0)
    return total_preds,total_targets

def test(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_input_path)     
    model = Transformer(os.path.join(args.output_path, 'base_model'),args.baseline).cuda()
    model.classifier.load_state_dict(torch.load(os.path.join(args.output_path, 'classifier.pt')))
    df_test = pd.read_csv(os.path.join(args.data_input_path, 'test.csv'))
    test_header,test_table,test_label = read_table(df_test,args.outcome,True,args.baseline)
    test_set = ClinicalDataset(test_header,test_table,test_label,tokenizer.eos_token,args.baseline)
    test_dataloader = DataLoader(test_set,shuffle=False, batch_size=args.test_batch_size,
                                collate_fn=partial(test_set.batchify, tokenizer=tokenizer))
    
    total_preds,total_targets = validate(args,model,test_dataloader)
    total_preds,total_targets = convert_to_numpy(total_preds,total_targets)
    f1 = f1_score(total_targets, total_preds > 0.5)
    roc_auc,average_precision,sensitivity,specificity,threshold = auc_score(total_targets,total_preds)
    results = {
        "F1":f1,
        "AUC":roc_auc,
        "AUPRC":average_precision,
        "Sensitivity":sensitivity,
        "Specificity":specificity,
        "Score Threshold":threshold
    }
    print("***************Test Results***************")
    for key, value in results.items():
        print(f"{key:25}: {value}")
    results_path = os.path.join(args.output_path, f'results.txt')
    with open(results_path,"a") as f:
        f.write(f"{results['F1']}\t{results['AUC']}\t{results['AUPRC']}\t{results['Sensitivity']}\t{results['Specificity']}\n")