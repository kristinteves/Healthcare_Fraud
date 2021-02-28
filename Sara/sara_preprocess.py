import pandas as pd
import numpy as np

def extract_data(first, second, third):
    """
    Extract csv files for four files
    """

    first_df = pd.read_csv(first)
    second_df = pd.read_csv(second)
    third_df = pd.read_csv(third)

    return first_df, second_df, third_df

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
    df['ClmAdmitDiagnosisCode'] = df['ClmAdmitDiagnosisCode'].apply(lambda x: 1 if(pd.notnull(x)) else 0)
    df = procedure_update(df)
    df = diagnosis_count(df)
    none_features = ['ClmDiagnosisCode_1','ClmDiagnosisCode_2','ClmDiagnosisCode_3','ClmDiagnosisCode_4',\
                    'ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7','ClmDiagnosisCode_8',\
                   'ClmDiagnosisCode_9','ClmDiagnosisCode_10','ClmProcedureCode_1', 'ClmProcedureCode_2',\
                   'ClmProcedureCode_3','ClmProcedureCode_4','ClmProcedureCode_5','ClmProcedureCode_6',\
                   'OperatingPhysician','OtherPhysician','AttendingPhysician']
    df = fill_none_missing(df,none_features)
    df.drop(['DiagnosisGroupCode'], inplace=True, axis=1)
    
    return df

def update_outpatient(df):
    '''
    Update procedure and diagnosis claim columns
    Impute NA with 'None'
    '''

    df = procedure_update(df)
    df = diagnosis_count(df)
    df['ClmAdmitDiagnosisCode'] = df['ClmAdmitDiagnosisCode'].apply(lambda x: 1 if(pd.notnull(x)) else 0)
    none_features = ['ClmDiagnosisCode_1','ClmDiagnosisCode_2','ClmDiagnosisCode_3','ClmDiagnosisCode_4',\
                    'ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7','ClmDiagnosisCode_8',\
                   'ClmDiagnosisCode_9','ClmDiagnosisCode_10','ClmProcedureCode_1', 'ClmProcedureCode_2',\
                   'ClmProcedureCode_3','ClmProcedureCode_4','ClmProcedureCode_5','ClmProcedureCode_6',\
                   'OperatingPhysician','OtherPhysician','AttendingPhysician']
    df = fill_none_missing(df, none_features)
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

    df['RenalDiseaseIndicator'] = df['RenalDiseaseIndicator'].str.replace('Y', '1').astype('float')

    for col in Chron_Conditions:
        df[col].replace({2:0}, inplace=True)
        df[col] = df[col].astype(object)

    df = fill_none_missing(df,['DOD'])


    return df


def combine_df(in_,out_,be):
    
    inpatient, outpatient, bene = extract_data(in_, out_, be)
    inpatient = update_inpatient(inpatient)
    outpatient = update_outpatient(outpatient)
    bene = update_bene(bene)

    patients = pd.concat([inpatient, outpatient], axis=0, ignore_index=True, sort=False)
    patients['Inpatient'] = patients['Inpatient'].fillna(0)

    merge_1 = pd.merge(left = patients, right = bene, on = 'BeneID')

    merge_1['DOD'] = [0 if x == 'None' else 1 for x in merge_1['DOD']]
    
    merge_1['Gender'] = [1 if x == 2 else 0 for x in merge_1['Gender']]

    return merge_1

# 'Train_Inpatientdata.csv','Train_Outpatientdata.csv','Train_Beneficiarydata.csv'

pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)

df = combine_df('Train_Inpatientdata.csv','Train_Outpatientdata.csv','Train_Beneficiarydata.csv')
df.to_csv('sara_preprocess.csv', index=False)


