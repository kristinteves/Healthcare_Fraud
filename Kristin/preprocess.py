import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

def extract_data(first, second, third, fourth):
    """
    Extract csv files for four files
    """
    parse_dates = ['ClaimStartDt','DOB','ClaimEndDt','AdmissionDt','DOD','DischargeDt']

    first_df = pd.read_csv(first,parse_dates=['ClaimStartDt','ClaimEndDt','AdmissionDt','DischargeDt'])
    second_df = pd.read_csv(second, parse_dates=['ClaimStartDt','ClaimEndDt'])
    third_df = pd.read_csv(third,parse_dates=['DOB','DOD'])
    fourth_df = pd.read_csv(fourth)
    return first_df, second_df, third_df, fourth_df


def procedure_update(data):
    """
    Update ClmProcedureCode columns to create 2 new columns: Binary column if ClmProcedureCode exists and ClmProcedureCode count
    Drop the ClmProcedureCode features
    """
    lst = ['ClmProcedureCode_1','ClmProcedureCode_2','ClmProcedureCode_3','ClmProcedureCode_4',\
                    'ClmProcedureCode_5','ClmProcedureCode_6']
    clmpro = data[lst]
    data['ClmProcedureCode'] = np.nan
    data['ClmProcedureCode_count'] = np.nan

    for i in range(0,len(clmpro)):
        if clmpro.iloc[i].isnull().all():
            data['ClmProcedureCode'][i] = 0
        else:
            data['ClmProcedureCode'][i] = 1

    for i in range(0,len(clmpro)):
        if clmpro.iloc[i].isnull().all():
            data['ClmProcedureCode_count'][i] = 0
        if clmpro.iloc[i].dropna().any():
            data['ClmProcedureCode_count'][i] = len(clmpro.iloc[i].dropna())
    data = data.drop(lst, axis=1)
            
    return data

def diagnosis_count(data):
    """
    Couunt ClmDiagnosisCodes
    """
    lst = ['ClmDiagnosisCode_1','ClmDiagnosisCode_2','ClmDiagnosisCode_3','ClmDiagnosisCode_4',\
                    'ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7','ClmDiagnosisCode_8',\
                   'ClmDiagnosisCode_9','ClmDiagnosisCode_10']
    clmdia = data[lst]
    data['ClmDiagnosisCode_count'] = np.nan
    
    for i in range(0,len(clmdia)):
        if clmdia.iloc[i].isnull().all():
            data['ClmDiagnosisCode_count'][i] = 0
        if clmdia.iloc[i].dropna().any():
            data['ClmDiagnosisCode_count'][i] = len(clmdia.iloc[i].dropna())
    
    return data

def fill_none_missing(data,features):
    '''
    Impute NaN with 'None'
    '''
    for feature in features:
        data[feature] = data[feature].fillna('None')
    return data

def update_inpatient(df):
    '''
    Include new columm to identify inpatients.
    Impute DeductibleAmtPaid with mode value
    Update procedure and diagnosis claim columns
    Impute NA with 'None'
    '''
    df['Inpatient'] = 1
    df['DeductibleAmtPaid'] = df['DeductibleAmtPaid'].fillna(float(df['DeductibleAmtPaid'].mode()))
    df = procedure_update(df)
    df = diagnosis_count(df)
    none_features = ['ClmDiagnosisCode_1','ClmDiagnosisCode_2','ClmDiagnosisCode_3','ClmDiagnosisCode_4',\
                    'ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7','ClmDiagnosisCode_8',\
                   'ClmDiagnosisCode_9','ClmDiagnosisCode_10','OperatingPhysician','ClmAdmitDiagnosisCode',\
                'OtherPhysician','AttendingPhysician']
    df = fill_none_missing(df,none_features)
    return df

def update_outpatient(df):
    '''
    Update procedure and diagnosis claim columns
    Impute NA with 'None'
    '''

    df = procedure_update(df)
    df = diagnosis_count(df)
    none_features = ['ClmDiagnosisCode_1','ClmDiagnosisCode_2','ClmDiagnosisCode_3','ClmDiagnosisCode_4',\
                    'ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7','ClmDiagnosisCode_8',\
                   'ClmDiagnosisCode_9','ClmDiagnosisCode_10','OperatingPhysician','ClmAdmitDiagnosisCode',\
                'OtherPhysician','AttendingPhysician']
    df = fill_none_missing(df, none_features)

def update_train(df):
    '''
    Transform PotentialFraud - replace Yes, No with 1, 0
    '''
    df['PotentialFraud'] = df['PotentialFraud'].replace(['Yes','No'], [1,0])
    return df

def update_bene(df):
    '''
    Introduce binary values to ChronicCond.
    Impute None to DOD.
    '''
    Chron_Conditions = ['ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
           'ChronicCond_KidneyDisease', 'ChronicCond_Cancer',
           'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
           'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
           'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis',
           'ChronicCond_stroke']

    df['RenalDiseaseIndicator'] = df['RenalDiseaseIndicator'].replace(['0', 'Y'], [0, 1], inplace=True)

    for col in Chron_Conditions:
        df[col].replace({2:0}, inplace=True)
        df[col] = df[col].astype(object)
        
    df['num_chronic'] = df[[x for x in df.columns if "ChronicCond" in x]].sum(axis = "columns")

    df = fill_none_missing(df,['DOD'])


    return df


def combine_df(in_,out_,be,tr):
    
    inpatient, outpatient, bene, train = extract_data(in_, out_, be, tr)
    inpatient = update_inpatient(inpatient)
    outpatient = update_outpatient(outpatient)
    train = update_train(train)
    bene = update_bene(bene)

    patients = pd.concat([inpatient, outpatient], axis=0, ignore_index=True, sort=False)
    patients['Inpatient'] = patients['Inpatient'].fillna(0)
    pat_none_feat = ['AdmissionDt','DischargeDt','DiagnosisGroupCode']
    patients = fill_none_missing(patients,pat_none_feat)

    merge_1 = pd.merge(left = patients, right = bene, on = 'BeneID')
    merge_2 = pd.merge(left = merge_1, right = train, on = 'Provider')
    merge_2['Age'] = merge_2.apply(lambda x: relativedelta(x['ClaimStartDt'], x['DOB']).years, axis=1)
    merge_2.drop(['DOB'], inplace=True, axis=1)

    return merge_2

# 'Train_Inpatientdata.csv','Train_Outpatientdata.csv','Train_Beneficiarydata.csv','Train.csv'

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)

df = combine_df('Train_Inpatientdata.csv','Train_Outpatientdata.csv','Train_Beneficiarydata.csv','Train.csv')
df.to_csv('Preprocess.csv', index=False)

