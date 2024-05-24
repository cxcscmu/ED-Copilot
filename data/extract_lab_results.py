import pandas as pd
df_master = pd.read_csv('master_dataset.csv')
#Only hospitalized patients has lab results during ED
df_master_hospitalized = df_master[df_master["outcome_hospitalization"]==True]
# This will take 3 min
df_lab_master = pd.read_csv('hosp/labevents.csv')
df_lab_hospitalized = df_lab_master[pd.notna(df_lab_master['hadm_id'])]
df_labitems = pd.read_csv('hosp/d_labitems_labeled.csv')
#extract common labs
labitems_common_list = df_labitems[df_labitems["ed_labs"]==1]["itemid"].tolist()
df_lab_hospitalized_common = df_lab_hospitalized[df_lab_hospitalized["itemid"].isin(labitems_common_list)]
df_lab_hospitalized_common = df_lab_hospitalized_common[pd.notna(df_lab_hospitalized_common["valuenum"])]
df_master_hospitalized_lab = pd.merge(df_master_hospitalized, df_lab_hospitalized_common[['subject_id', 'hadm_id', 'itemid','charttime','storetime','valuenum']], on = ['subject_id', 'hadm_id'], how='left')
# Remove those lab not in ED
df_master_hospitalized_lab = df_master_hospitalized_lab[(pd.to_datetime(df_master_hospitalized_lab['storetime'])<=pd.to_datetime(df_master_hospitalized_lab['outtime'])) & (pd.to_datetime(df_master_hospitalized_lab['intime'])<=pd.to_datetime(df_master_hospitalized_lab['storetime']) )]
df_master_hospitalized_lab_grouped = df_master_hospitalized_lab[['subject_id', 'hadm_id','stay_id','itemid','valuenum']]
df_master_hospitalized_lab_results_names = df_master_hospitalized_lab_grouped.groupby(['subject_id', 'hadm_id','stay_id']).apply(lambda x:x['itemid'].astype(int).tolist()).reset_index()
df_master_hospitalized_lab_results_names.columns = ['subject_id', 'hadm_id', 'stay_id','lab_measure']
df_master_hospitalized_lab_results_nums = df_master_hospitalized_lab_grouped.groupby(['subject_id', 'hadm_id','stay_id']).apply(lambda x:x['valuenum'].tolist()).reset_index()
df_master_hospitalized_lab_results_nums.columns = ['subject_id', 'hadm_id', 'stay_id','lab_results']
df_master_hospitalized_lab_results = pd.merge(df_master_hospitalized_lab_results_names, df_master_hospitalized_lab_results_nums, on=['subject_id','hadm_id', 'stay_id'])

def has_duplicates(lst):
    return len(lst) != len(set(lst))
df_master_hospitalized_lab_results['has_duplicates'] = df_master_hospitalized_lab_results['lab_measure'].apply(has_duplicates)
#filter out those patients that have duplicate lab measures
filtered_df_master_hospitalized_lab_results = df_master_hospitalized_lab_results[~df_master_hospitalized_lab_results['has_duplicates']]
# Drop the 'has_duplicates' column 
filtered_df_master_hospitalized_lab_results = filtered_df_master_hospitalized_lab_results.drop(columns=['has_duplicates'])
filtered_df_master_hospitalized_lab_results.to_csv("master_lab_results.csv",index=False)