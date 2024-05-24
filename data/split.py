
import pandas as pd
import os

df_master = pd.read_csv("master.csv")


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
    
    #VS
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
    'creatine kinase (ck)', 'c-reactive protein',
    
]

outcome = [
    'outcome_critical',
    'outcome_ed_los',
    'lab_group_idx'
]

df_master = df_master[ed_ehr+ed_lab+outcome].copy()
df_master['triage_temperature'] = df_master['triage_temperature'].apply(lambda x: round(x,1))
new_column_names = ['feature' + str(i+1) for i in range(df_master.shape[1]-3)]+outcome
df_master.columns = new_column_names

df_train = df_master.sample(frac=0.8,random_state=42) #random state is a seed value
df_valid_test = df_master.drop(df_train.index)

df_valid = df_valid_test.sample(frac=1/2,random_state=10) 
df_test = df_valid_test.drop(df_valid.index)

df_train.to_csv("train.csv", index=False)
df_valid.to_csv("valid.csv", index=False)
df_test.to_csv("test.csv", index=False)