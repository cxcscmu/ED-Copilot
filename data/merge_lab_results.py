import pandas as pd
from ast import literal_eval
df_master = pd.read_csv('master_dataset.csv')
#Only hospitalized patients has lab results during ED
df_master_hospitalized = df_master[df_master["outcome_hospitalization"]==True]
df_master_lab = pd.read_csv('master_lab_results.csv',converters={'lab_measure':literal_eval,'lab_results':literal_eval})
df_master_lab_item_group = pd.read_csv("hosp/lab_item_frequency_group.csv")
itemid_to_label = df_master_lab_item_group.set_index('itemid')['label'].to_dict()
itemid_to_grouping =  df_master_lab_item_group.set_index('itemid')['grouping'].to_dict()
#Convert into lower case
itemid_to_label = {key: value.lower() for key, value in itemid_to_label.items()}
#Reserve for two groups for triage
group2idx = {
    'cbc':0,
    'chem':1,
    'coag':2,
    'UA':3,
    'lactate':4,
    'LFTs':5,
    'lipase':6,
    'lytes':7,
    'cardio':8,
    'blood gas':9,
    'tox':10,
    'inflammation':11
}

#Merge lab values into original df
import random
itemid_list = list(itemid_to_label.keys())
label_list =  list(itemid_to_label.values())
holder_list = []
for index, row in df_master_lab.iterrows():
    lab_list = []
    lab_category_list = []
    for itemid_idx in itemid_list:
        if itemid_idx not in row["lab_measure"]:
            lab_list.append(0)
        else:
            lab_list.append(row["lab_results"][row["lab_measure"].index(itemid_idx)])
            if group2idx[itemid_to_grouping[itemid_idx]] not in lab_category_list: 
                lab_category_list.append(group2idx[itemid_to_grouping[itemid_idx]])
    lab_list.append(lab_category_list)
    holder_list.append(lab_list)
label_list.append("lab_group_idx")
df_lab_values = pd.DataFrame(holder_list, columns = label_list)
df_master_lab.reset_index(drop=True, inplace=True)
df_lab_values.reset_index(drop=True, inplace=True)
df_master_lab = pd.concat([df_master_lab,df_lab_values], axis=1).drop(['lab_measure','lab_results'],axis=1)
#Filter our those patients that only did rare lab test
df_master_lab = df_master_lab[df_master_lab['lab_group_idx'].apply(lambda x: len(x) != 0)]
df_master_hospitalized_lab = pd.merge(df_master_hospitalized,df_master_lab,on = ['subject_id', 'hadm_id','stay_id'], how='right')

print('Before filtering for "age" >= 18 : master dataset size = ', len(df_master_hospitalized_lab))
df_master_hospitalized_lab = df_master_hospitalized_lab[df_master_hospitalized_lab['age'] >= 18]
print('After filtering for "age" >= 18 : master dataset size = ', len(df_master_hospitalized_lab))

print('Before filtering for non-null "triage_acuity" >= 18 : master dataset size = ', len(df_master_hospitalized_lab))
df_master_hospitalized_lab = df_master_hospitalized_lab[df_master_hospitalized_lab['triage_acuity'].notnull()]
print('After filtering for non-null "triage_acuity" >= 18 : master dataset size = ', len(df_master_hospitalized_lab))

vitals_valid_range = {
    'temperature': {'outlier_low': 14.2, 'valid_low': 26, 'valid_high': 45, 'outlier_high':47},
    'heartrate': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 350, 'outlier_high':390},
    'resprate': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 300, 'outlier_high':330},
    'o2sat': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 100, 'outlier_high':150},
    'sbp': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 375, 'outlier_high':375},
    'dbp': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 375, 'outlier_high':375},
    'pain': {'outlier_low': 0, 'valid_low': 0, 'valid_high': 10, 'outlier_high':10},
    'acuity': {'outlier_low': 1, 'valid_low': 1, 'valid_high': 5, 'outlier_high':5},
}
import numpy as np
def convert_temp_to_celcius(df_master):
    for column in df_master.columns:
        column_type = column.split('_')[1] if len(column.split('_')) > 1 else None
        if column_type == 'temperature':
            # convert to celcius
            df_master[column] -= 32
            df_master[column] *= 5/9
    return df_master

def display_outliers_count(df_master, vitals_valid_range):
    display_df =[]
    for column in df_master.columns:
        column_type = column.split('_')[1] if len(column.split('_')) > 1 else None
        if column_type in vitals_valid_range:
            column_range = vitals_valid_range[column_type]
            display_df.append({'variable': column,
                   '< outlier_low': len(df_master[df_master[column] < column_range['outlier_low']]),
                   '[outlier_low, valid_low)': len(df_master[(column_range['outlier_low'] <= df_master[column])
                                                             & (df_master[column] < column_range['valid_low'])]),
                   '[valid_low, valid_high]': len(df_master[(column_range['valid_low'] <= df_master[column])
                                                            & (df_master[column] <= column_range['valid_high'])]),
                   '(valid_high, outlier_high]': len(df_master[(column_range['valid_high'] < df_master[column])
                                                               & (df_master[column] <= column_range['outlier_high'])]),
                   '> outlier_high': len(df_master[df_master[column] > column_range['outlier_high']])
            })
            
    return pd.DataFrame(display_df)


df_master_hospitalized_lab = convert_temp_to_celcius(df_master_hospitalized_lab)



def outlier_removal_imputation(column_type, vitals_valid_range):
    column_range = vitals_valid_range[column_type]
    def outlier_removal_imputation_single_value(x):
        if x < column_range['outlier_low'] or x > column_range['outlier_high']:
            # set as missing
            return np.nan
        elif x < column_range['valid_low']:
            # impute with nearest valid value
            return column_range['valid_low']
        elif x > column_range['valid_high']:
            # impute with nearest valid value
            return column_range['valid_high']
        else:
            return x
    return outlier_removal_imputation_single_value


def remove_outliers(df_master, vitals_valid_range):
    for column in df_master.columns:
        column_type = column.split('_')[1] if len(column.split('_')) > 1 else None
        if column_type in vitals_valid_range:
            df_master[column] = df_master[column].apply(outlier_removal_imputation(column_type, vitals_valid_range))
    return df_master

df_master_hospitalized_lab = remove_outliers(df_master_hospitalized_lab, vitals_valid_range)
vitals_cols = [col for col in df_master_hospitalized_lab.columns if len(col.split('_')) > 1 and 
                                                   col.split('_')[1] in vitals_valid_range and
                                                   col.split('_')[1] != 'acuity']
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df_master_hospitalized_lab[vitals_cols] = imputer.fit_transform(df_master_hospitalized_lab[vitals_cols])
df_master_hospitalized_lab['outcome_ed_los'] = df_master_hospitalized_lab['ed_los_hours'] > 24
df_master_hospitalized_lab.to_csv("master.csv",index=False)