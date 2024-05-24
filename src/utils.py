import pandas as pd
import random
import numpy
import torch
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, average_precision_score,f1_score
import numpy as np
from tqdm import tqdm
from ast import literal_eval

ed_ehr = [
    "age", "gender", 
            
    "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
    "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d",
    
    "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", 
    "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", 
    "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", 
    "cci_Cancer2", "cci_HIV",  

    "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2", 
    "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
    "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
    "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression",
    
    # VS
    "triage_temperature",
    "triage_heartrate",
    "triage_resprate", 
    "triage_o2sat",
    "triage_sbp",
    "triage_dbp",
    #RNHX
    "triage_pain",
    "triage_acuity",
    "chiefcomplaint",
]


ed_ehr_dummy = [
    'feature' + str(i+1) for i in range(len(ed_ehr))
]

ed_lab = [
    #CBC
    'hematocrit',
    'white blood cells',
    'hemoglobin',
    'red blood cells',
    'mean corpuscular volume',
    'mean corpuscular hemoglobin',
    'mean corpuscular hemoglobin concentration',
    'red blood cell distribution width',
    'platelet count',
    'basophils',
    'eosinophils',
    'lymphocytes',
    'monocytes',
    'neutrophils',
    'red cell distribution width (standard deviation)',
    'absolute lymphocyte count',
    'absolute basophil count',
    'absolute eosinophil count',
    'absolute monocyte count',
    'absolute neutrophil count',
    'bands',
    'atypical lymphocytes',
    'nucleated red cells',
    #CHEM
    'urea nitrogen',
    'creatinine',
    'sodium',
    'chloride',
    'bicarbonate',
    'glucose (chemistry)',
    'potassium',
    'anion gap',
    'calcium, total',
    #COAG
    'prothrombin time', 'inr(pt)', 'ptt',
    #UA
    'ph (urine)',
    'specific gravity',
    'red blood count (urine)',
    'white blood count (urine)',
    'epithelial cells',
    'protein',
    'hyaline casts',
    'ketone',
    'urobilinogen',
    'glucose (urine)',
    #LACTATE
    'lactate',
    #LFTs
    'alkaline phosphatase',
    'asparate aminotransferase (ast)',
    'alanine aminotransferase (alt)',
    'bilirubin, total',
    'albumin',
    #LIPASE
    'lipase',
    #LYTES
    'magnesium', 'phosphate',
    #CARDIO,
    'ntprobnp', 'troponin t',
    #BLOOD_GAS
    'potassium, whole blood',
    'ph (blood gas)',
    'calculated total co2',
    'base excess',
    'po2',
    'pco2',
    'glucose (blood gas)',
    'sodium, whole blood',
    #TOX
    'ethanol',
    #INFLAMMATION
    'creatine kinase (ck)', 'c-reactive protein'
]

ed_lab_dummy = [
    'feature' + str(i+len(ed_ehr)+1) for i in range(len(ed_lab))
]


ed_lab_idx = [
    list(range(0,23)),#CBC
    list(range(23,32)),#CHEM
    list(range(32,35)),#COAG
    list(range(35,45)),#UA
    list(range(45,46)),#LACTATE
    list(range(46,51)),#LFTs
    list(range(51,52)),#LIPASE
    list(range(52,54)),#LYTES
    list(range(54,56)),#CARDIO
    list(range(56,64)),#BLOOD_GAS
    list(range(64,65)),#TOX
    list(range(65,67))#INFLAMMATION
]

cost = {
    "CBC":30,
    "CHEM":60,    
    "COAG":48,
    "UA":40,
    "LACTATE":4,
    "LFTs":104,
    "LIPASE":100,
    "LYTES":89,
    "CARDIO":122,
    "BLOOG_GAS":12,
    "TOX":70,
    "INFLAMMATION":178
}

def set_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def auc_score(y_test_roc,probs):
    fpr, tpr, threshold = metrics.roc_curve(y_test_roc,probs)
    roc_auc = metrics.auc(fpr, tpr)
    average_precision = average_precision_score(y_test_roc, probs)
    a=np.sqrt(np.square(fpr-0)+np.square(tpr-1)).argmin()
    sensitivity = tpr[a]
    specificity = 1-fpr[a]
    threshold = threshold[a]
    return roc_auc,average_precision,sensitivity,specificity,threshold

def convert_to_numpy(preds,targets):
    return preds.detach().cpu().numpy(),targets.detach().cpu().numpy()
#Remeber to modify this when using dummpy
def split_table(row_data,lab_idx):
    header = []
    for idx in lab_idx:
        # header.append(ed_lab[idx])
        header.append(ed_lab_dummy[idx])
    row_table_data = row_data[header].values.tolist()
    return header,row_table_data

def read_table(df_data,task,sft = False, baseline = False):
    # df_data['triage_temperature'] = df_data['triage_temperature'].apply(lambda x: round(x,1))
    diagnostic_label_list = list(map(int,df_data[task].copy().tolist()))
    raw_lab_group_idx_list = df_data["lab_group_idx"].copy().tolist()
    lab_group_label_list = list(map(lambda x: literal_eval(x), raw_lab_group_idx_list))
    
    # lab_group_label_list = [[element for element in inner_list if element != 0] for inner_list in lab_group_label_list]
    
    if sft :
        if baseline:
            header = ed_ehr+ed_lab
            table_data = df_data[header].copy().values.tolist()
        else:
            header = []
            table_data = []
            for index, row in df_data.iterrows():
                current_total_header = [ed_ehr_dummy]
                current_total_table_data = [row[ed_ehr_dummy].values.tolist()]
                current_lab_group = lab_group_label_list[index]

                for lab_idx in current_lab_group:
                    current_lab_header,current_lab_table_data = split_table(row,ed_lab_idx[lab_idx])
                    current_total_header.append(current_lab_header)
                    current_total_table_data.append(current_lab_table_data)

                assert len(current_total_header) == len(current_total_table_data)
                assert len(current_total_header) == len(current_lab_group)+1
                
                header.append(current_total_header)
                table_data.append(current_total_table_data)
            
    else:
        # X_ehr_data = df_data[ed_ehr].copy()
        # X_lab_data = df_data[ed_lab].copy()
        # header = [ed_ehr,ed_lab]
        X_ehr_data = df_data[ed_ehr_dummy].copy()
        X_lab_data = df_data[ed_lab_dummy].copy()
        header = [ed_ehr_dummy,ed_lab_dummy]
        table_data = [X_ehr_data.values.tolist(),X_lab_data.values.tolist()]
    
    return header,table_data,[diagnostic_label_list,lab_group_label_list]

def table2string(headers,table_values,eos_token,baseline = False):
    if baseline:
        total_table_strings = []
        for row in table_values:
            row_rep = ' | '.join([h + ' : ' + str(c) for h, c in zip(headers, row) if (type(c) == str and str(c) != "nan") or (str(c) != "nan" and float(c))]) 
            total_table_strings.append(row_rep+eos_token)  
    else:    
        total_table_strings = []
        for current_total_header,current_total_table_value in zip(headers,table_values):
            current_table_strings = []
            for current_header_idx, (current_header,current_table_value) in enumerate(zip(current_total_header,current_total_table_value)):
                if current_header_idx == 0:
                    row_rep = ' | '.join([h + ' : ' + str(c) for h, c in zip(current_header, current_table_value) if (type(c) == str and str(c) != "nan") or (str(c) != "nan" and float(c))]) 
                else:
                    row_rep =  ' | '.join([h + ' : ' + str(c) for h, c in zip(current_header, current_table_value)]) 
                current_table_strings.append(row_rep)    
            total_table_strings.append(eos_token.join(current_table_strings)+eos_token)  
    return total_table_strings

def personalize_cost(current_patient_lab_y):
    
    lab_names = list(cost.keys())
    current_patient_cost = cost.copy()
    
    # Calculate the sum of values to be normalized
    total = sum(current_patient_cost[lab_names[i]] for i in current_patient_lab_y)

    # Normalize the values
    for i in current_patient_lab_y:
        current_patient_cost[lab_names[i]] =  (current_patient_cost[lab_names[i]]/total)*len(current_patient_lab_y)

    return current_patient_cost

# def sample_indices(patients_label,pos_ratio):
#     total_samples = len(patients_label)
#     num_pos_samples = int(total_samples*pos_ratio)
#     num_neg_samples = total_samples-num_pos_samples
#     pos_indices = np.where(patients_label==1)[0]
#     neg_indices = np.where(patients_label==0)[0]
    
#     sampled_pos_indices = np.random.choice(pos_indices, num_pos_samples, replace=True)
#     sampled_neg_indices = np.random.choice(neg_indices, num_neg_samples, replace=False)    
    
#     sampled_indices = np.concatenate((sampled_pos_indices, sampled_neg_indices))
#     np.random.shuffle(sampled_indices)
#     return sampled_indices   

# from transformers import AutoTokenizer 
# import pandas as pd
# from IPython import embed
# df_data = pd.read_csv("../data/new_los/train.csv")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt") 

# tokenizer = AutoTokenizer.from_pretrained( "gpt2-medium")
# header,table_data,label_list = read_table(df_data,"outcome_critical",True,False)
# embed()
# # cost_list = list(cost.values())
# # def calculate_total_costs(list_of_lists, cost_list):
# #     total_costs = [sum(cost_list[index] for index in sublist) for sublist in list_of_lists]
# #     return total_costs

# # list_of_lists = label_list[1]
# # total_costs = calculate_total_costs(list_of_lists, cost_list)
# # average_cost = sum(total_costs) / len(total_costs)
# # print(average_cost)
# table_strings = table2string(header,table_data,tokenizer.eos_token,False) 
# print(table_strings[0])
# print(tokenizer(table_strings[0],add_special_tokens=False)['input_ids'])
# length = [len(tokenizer(text)['input_ids']) for text in table_strings]
# # # # # def Average(lst): 
# # # # #     return sum(lst) / len(lst) 
# print("Max length of tokenized inputs:", max(length))

# # print(personalize_cost([0,1,2]))