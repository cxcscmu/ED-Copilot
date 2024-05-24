import torch
import pandas as pd
import os
from src.utils import read_table,cost,auc_score
import wandb
from src.env import EDCopilotEnv
import torch
from sb3_contrib.ppo_mask import MaskablePPO
import os
import pandas as pd
import wandb
from wandb.integration.sb3 import WandbCallback
from src.policy import MaskableLMActorCriticPolicy
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score


import pickle
def train(args):
    policy_kwargs = {
        "model_name_path":args.output_path,
        "weight_decay":args.weight_decay,
        "optimizer_class":torch.optim.AdamW,
        "optimizer_kwargs":{
            "eps":args.adam_epsilon
        }
    }  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

    df_train = pd.read_csv((os.path.join(args.data_input_path, 'train.csv')))
    df_valid = pd.read_csv((os.path.join(args.data_input_path, 'valid.csv')))
    header,train_table,train_label = read_table(df_train,args.outcome)
    header,valid_table,valid_label = read_table(df_valid,args.outcome)
    train_env = EDCopilotEnv((train_table,train_label),header,args,True)
    valid_env = EDCopilotEnv((valid_table,valid_label),header,args)


    model = MaskablePPO(MaskableLMActorCriticPolicy, 
                        train_env, 
                        learning_rate = args.learning_rate, 
                        n_steps = args.buffer_steps, 
                        n_epochs = args.ppo_epochs,
                        batch_size = args.batch_size,
                        policy_kwargs=policy_kwargs,
                        verbose=0,
                        tensorboard_log="run",
                        device=device,
                        max_grad_norm = args.max_grad_norm)
    
    # wandb.init(project="HealthCare", 
    #            entity="zhiyuan-chenyan-zhenghao-group",
    #            group="ED Copilot",
    #            name=f"Biogpt abb ratio = {args.penalty_ratio} penalty = {args.wrong_prediction_penalty} task = {args.outcome}",
    #            sync_tensorboard=True)
    
    # wandb.config.update(args)  
    # wandb.config.update(cost)
    # wandb.config.update(policy_kwargs)

    ckpt_path = os.path.join(args.output_path, f'{args.penalty_ratio}_{args.wrong_prediction_penalty}')
    os.makedirs(ckpt_path, exist_ok = True)

    best_f1 = 0
    for _ in tqdm(range(args.epochs),desc="Training"):
        model.learn(args.total_timesteps,tb_log_name = f"lab_{args.penalty_ratio}_{args.wrong_prediction_penalty}_{args.outcome}", progress_bar = True)
        results = validate(model,valid_env)
        for key, value in results.items():
            print(f"{key:25}: {value}")
        # wandb.log(results)
        if best_f1 < results["F1"]:
            best_f1 = results["F1"]
            model.policy.save(ckpt_path)
            print(f"Model saved to {ckpt_path}")

    test(args)
@torch.no_grad()
def validate(model,env):
    model.policy.set_training_mode(False)
    max_episodes = env.num_patients
    
    cumulative_reward = 0
    n_tested_ave = 0
    cost_tested_ave = 0
    n_healthy = 0
    n_ill = 0
    n_healthy_acc_predict = 0
    n_ill_acc_predict = 0

    predict_probs = np.zeros(max_episodes)
    true_labels = np.zeros(max_episodes)
    pred_labels = np.zeros(max_episodes)
    action_dist = {}
    total_action_preds = []
    total_action_targets =[]
    
    for episode in tqdm(range(max_episodes),desc="Evaluating"):
        obs = env.reset()
        done = False
        true_labels[episode] = env.current_patient_y
        total_action_targets.append(env.current_patient_lab_y)
        
        action_preds = []
        if not env.current_patient_y:
            n_healthy += 1
        else:
            n_ill += 1

        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic = True, action_masks = action_masks)

            obs, reward, done, info = env.step(action)
            cumulative_reward += reward
            if action < env.num_lab:
                n_tested_ave += 1

                cost_pair = list(cost.items())[action]
                feature_name = cost_pair[0]
                cost_tested_ave += cost_pair[1]
                                     
                action_dist[feature_name] = action_dist.get(feature_name, 0) + 1
                
                action_preds.append(int(action))
            else:
                obs_tensor = model.policy.obs_to_tensor(obs)[0]
                action_list = torch.tensor(list(range(env.num_lab + 2))).cuda()
                prob = torch.exp(model.policy.evaluate_actions(obs_tensor, action_list, action_masks)[1])[-2:]
                predict_probs[episode] = (prob[-1] / torch.sum(prob)).item() #predict prob of ill patient, class 1

                pred_labels[episode] = (action == (env.num_lab+1))
                
                if not env.current_patient_y:
                    n_healthy_acc_predict += (action == env.num_lab)
                else:
                    n_ill_acc_predict += (action == (1+env.num_lab))
                    
                total_action_preds.append(action_preds)

    f1 = f1_score(true_labels, pred_labels)
    # pickle.dump(total_action_preds,open("/home/liwens/healthcare/Lightning-Pretrain/copilot_v3/analysis_free_form/pred_len_dist.pkl","wb"))
    # pickle.dump(total_action_targets,open("/home/liwens/healthcare/Lightning-Pretrain/copilot_v3/analysis_free_form/gt_len_dist.pkl","wb"))
    # pickle.dump(pred_labels,open("/home/liwens/healthcare/Lightning-Pretrain/copilot_v3/analysis_free_form/pred_label.pkl","wb"))
    # pickle.dump(true_labels,open("/home/liwens/healthcare/Lightning-Pretrain/copilot_v3/analysis_free_form/gt_label.pkl","wb"))
    # pickle.dump(predict_probs,open("/home/liwens/healthcare/Lightning-Pretrain/copilot_v3/analysis/pred_prob.pkl","wb"))
    # bleu_scores = [sentence_bleu([ref], pred, weights=(1, 0, 0, 0)) for pred, ref in zip(total_action_preds, total_action_targets)]

    
    print(f"{'n_healthy':25}: {n_healthy}")
    print(f"{'n_ill':25}: {n_ill}")
    print(f"{'n_healthy_acc_predict':25}: {n_healthy_acc_predict}")
    print(f"{'n_ill_acc_predict':25}: {n_ill_acc_predict}")
    print(f"{'action_distribution':25}: {action_dist}")
    # print(f"{'average bleu-1':25}: { sum(bleu_scores) / len(bleu_scores)}")

    roc_auc,average_precision,sensitivity,specificity,threshold = auc_score(true_labels,predict_probs)
    results = {
        "F1":f1,
        "AUC":roc_auc,
        "AUPRC":average_precision,
        "Sensitivity":sensitivity,
        "Specificity":specificity,
        "Threshold":threshold,
        "Cumulative reward":cumulative_reward / max_episodes,
        "Cost average":cost_tested_ave / max_episodes,
        "Average number of test":n_tested_ave / max_episodes,
    }
    return results
    
        
def test(args):
    df_test = pd.read_csv((os.path.join(args.data_input_path, 'test.csv')))
    header,test_table,test_label = read_table(df_test,args.outcome)
    test_env = EDCopilotEnv((test_table,test_label),header,args)

    policy_kwargs = {
        "model_name_path":args.output_path,
        "weight_decay":args.weight_decay,
        "optimizer_class":torch.optim.AdamW,
        "optimizer_kwargs":{
            "eps":args.adam_epsilon
        }
    }  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MaskablePPO(MaskableLMActorCriticPolicy, 
                        test_env, 
                        learning_rate = args.learning_rate, 
                        n_steps = args.buffer_steps, 
                        batch_size = args.batch_size,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        tensorboard_log="run",
                        device=device,
                        max_grad_norm = args.max_grad_norm)
    
    ckpt_path = os.path.join(args.output_path, f'{args.penalty_ratio}_{args.wrong_prediction_penalty}')
    model.policy.load(ckpt_path)

    results = validate(model,test_env)
    for key, value in results.items():
        print(f"{key:25}: {value}")
        
    results_path = os.path.join(args.output_path, f'cost_effective_curve.txt')
    with open(results_path,"a") as f:
        f.write(f"{results['F1']}\t{results['AUC']}\t{results['AUPRC']}\t{results['Sensitivity']}\t{results['Specificity']}\t{results['Cost average']}\t{args.penalty_ratio}\t{args.wrong_prediction_penalty}\n")
        
        
@torch.no_grad()
def infer(args):
    df_test = pd.read_csv((os.path.join(args.data_input_path, 'test.csv')))
    header,test_table,test_label = read_table(df_test,args.outcome)
    test_env = EDCopilotEnv((test_table,test_label),header,args)

    policy_kwargs = {
        "model_name_path":args.output_path,
        "weight_decay":args.weight_decay,
        "optimizer_class":torch.optim.AdamW,
        "optimizer_kwargs":{
            "eps":args.adam_epsilon
        }
    }  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model = MaskablePPO(MaskableLMActorCriticPolicy, 
                        test_env, 
                        learning_rate = args.learning_rate, 
                        n_steps = args.buffer_steps, 
                        batch_size = args.batch_size,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        tensorboard_log="run",
                        device=device,
                        max_grad_norm = args.max_grad_norm)
    
    ckpt_path = os.path.join(args.output_path, f'{args.penalty_ratio}_{args.wrong_prediction_penalty}')
    model.policy.load(ckpt_path)
    
    model.policy.set_training_mode(False)
    obs = test_env.reset()
    test_name = list(cost.keys())
    done = False
    suggested_sequence = []
    while not done:
        print(test_env.current_patient_obs)
        action_masks = test_env.action_masks()
        action, _ = model.predict(obs, deterministic = True, action_masks = action_masks)
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        action_list = torch.tensor(list(range(test_env.num_lab + 2))).cuda()
        
        prob = torch.exp(model.policy.evaluate_actions(obs_tensor, action_list, action_masks)[1]).cpu().numpy()
        print(f"{'Action Probability':25}: {np.around(prob,2)}")
        
        obs, _, done, _ = test_env.step(action)
        if action < test_env.num_lab:
            cost_pair = list(test_env.current_patient_cost.items())[action]
            feature_name = cost_pair[0]
            print(f"{'Selected Test':25}: {feature_name}")
            suggested_sequence.append(feature_name)
        else:
            if action == test_env.num_lab:
                prediction = "negative"
            else:
                prediction = "positive"
                
            if not test_env.current_patient_y:
                label = "negative"
            else:
                label = "positive" 
                            
            print(f"{'Ground Truth Sequence':25}: {[ test_name[lab_idx] for lab_idx in test_env.current_patient_lab_y]}")
            print(f"{'Predicted Label':25}: {prediction}")
            print(f"{'Ground Truth label':25}: {label}")
            print(f"{'Suggested Sequence':25}: {suggested_sequence}")
            
@torch.no_grad()
def generate(args):
    df_test = pd.read_csv((os.path.join(args.data_input_path, 'test.csv')))
    header,test_table,test_label = read_table(df_test,args.outcome)
    test_env = EDCopilotEnv((test_table,test_label),header,args)

    policy_kwargs = {
        "model_name_path":args.output_path,
        "weight_decay":args.weight_decay,
        "optimizer_class":torch.optim.AdamW,
        "optimizer_kwargs":{
            "eps":args.adam_epsilon
        }
    }  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model = MaskablePPO(MaskableLMActorCriticPolicy, 
                        test_env, 
                        learning_rate = args.learning_rate, 
                        n_steps = args.buffer_steps, 
                        batch_size = args.batch_size,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        tensorboard_log="run",
                        device=device,
                        max_grad_norm = args.max_grad_norm)
    
    ckpt_path = os.path.join(args.output_path, f'{args.penalty_ratio}_{args.wrong_prediction_penalty}')
    model.policy.load(ckpt_path)
    
    model.policy.set_training_mode(False)
    max_episodes = test_env.num_patients



    for threshold in tqdm(range(0,700,50)):
        n_tested_ave = 0
        predict_probs = np.zeros(max_episodes)
        true_labels = np.zeros(max_episodes)
        pred_labels = np.zeros(max_episodes)
        for episode in tqdm(range(max_episodes),desc="Generating"):
            obs = test_env.reset()
            done = False
            cur_lab_cost = 0
            true_labels[episode] = test_env.current_patient_y
            
            while not done:
                action_masks = test_env.action_masks()

                #get prob dist
                obs_tensor = model.policy.obs_to_tensor(obs)[0]
                action_list = torch.tensor(list(range(test_env.num_lab + 2))).cuda()
                prob = torch.exp(model.policy.evaluate_actions(obs_tensor, action_list, action_masks)[1])[-2:]
                
                #get next action
                action, _ = model.predict(obs, deterministic = True, action_masks = action_masks)
                
                obs, _, done, _ = test_env.step(action)

                if cur_lab_cost <= threshold:
                    if action < test_env.num_lab:
                        n_tested_ave += 1
                        next_lab_cost = list(cost.values())[action]
                        cur_lab_cost += next_lab_cost
                        if cur_lab_cost > threshold:
                            n_tested_ave-=1
                            predict_probs[episode] = (prob[-1] / torch.sum(prob)).item() #predict prob of ill patient, class 1
                            pred_labels[episode] = (action == (test_env.num_lab+1))     
                            break
                    else:
                        predict_probs[episode] = (prob[-1] / torch.sum(prob)).item() #predict prob of ill patient, class 1
                        pred_labels[episode] = (action == (test_env.num_lab+1))    

        f1 = f1_score(true_labels, pred_labels)
        roc_auc,average_precision,sensitivity,specificity,pred_threshold = auc_score(true_labels,predict_probs)
        results_path = os.path.join(args.output_path, f'cost_effective_curve_threshold_mask.txt')
        with open(results_path,"a") as f:
            f.write(f"{f1 }\t{roc_auc}\t{sensitivity}\t{specificity}\t{n_tested_ave/max_episodes}\t{threshold}\n")
                        
                        