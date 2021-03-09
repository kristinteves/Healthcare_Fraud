import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

class PreProcess:
    list_of_files = None
    df = None

    def __init__(self, list_of_files):
        self.list_of_files = list_of_files

    def extract_data(self):
        """
        Extract csv files for four files
        """
        file_data = self.list_of_files
        if not file_data:
            raise Exception("Missing list of csv files")

        first = file_data[0].name
        second = file_data[1].name
        third = file_data[2].name
        fourth = file_data[3].name

        first_df = pd.read_csv(first,parse_dates=['ClaimStartDt','ClaimEndDt','AdmissionDt','DischargeDt'])
        second_df = pd.read_csv(second, parse_dates=['ClaimStartDt','ClaimEndDt'])
        third_df = pd.read_csv(third,parse_dates=['DOB','DOD'])
        fourth_df = pd.read_csv(fourth)
        return first_df, second_df, third_df, fourth_df

    def procedure_update(self, data):
        """
        Update ClmProcedureCode columns to create 2 new columns: Binary column if ClmProcedureCode exists and ClmProcedureCode count
        Drop the ClmProcedureCode features
        """
        
        proc_codes = [4019, 9904, 2724, 8154, 66, 3893, 3995, 4516, 3722, 8151, 8872]
        proc_cols = [x for x in data.columns if "ClmProcedureCode" in x]

        for col in proc_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce').astype(pd.Int64Dtype())

        for col in proc_codes:
            data[str(col)+'_proc'] = np.where((data[proc_cols].eq(col)).any(1, skipna=True), 1, 0)
        
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

    def diagnosis_count(self, data):
        """
        Couunt ClmDiagnosisCodes
        """
        diag_codes = ['4019','25000','2724','V5869','4011','42731','V5861','2720','2449','4280','53081','41401',\
                   '496','2859','41400','78079','5990','28521','3051','2809','311','73300','58881','71590',\
                   '5859','V4581','2722']
        diag_cols = [x for x in data.columns if "ClmDiagnosisCode" in x]

        for col in diag_codes:
            data[str(col)+'_diag'] = np.where((data[diag_cols].eq(col)).any(1, skipna=True), 1, 0)

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

        

    def duplicate_claims(self, data):
        """
        This function creates a feature that codes for whether a claim is a duplicate.
        """

        duplicate_columns = [col for col in data.columns if 'ClmDiagnosis' in col or 'ClmProcedureCode'in col 
                        or 'ClmAdmitDiagnosisCode' in col]
        data['duplicate'] = data.duplicated(duplicate_columns, keep=False)
        data['duplicate'] = [1 if x == True else 0 for x in data.duplicate]
        
        return data

    def fill_none_missing(self, data,features):
        '''
        Impute NaN with 'None'
        '''
        for feature in features:
            data[feature] = data[feature].fillna('None')
        return data

    def update_inpatient(self, df):
        '''
        Include new columm to identify inpatients.
        Impute DeductibleAmtPaid with mode value
        Update procedure and diagnosis claim columns
        Impute NA with 'None'
        '''
        df['Inpatient'] = 1
        df['DeductibleAmtPaid'] = df['DeductibleAmtPaid'].fillna(float(df['DeductibleAmtPaid'].mode()))
        df['ClmAdmitDiagnosisCode'] = df['ClmAdmitDiagnosisCode'].apply(lambda x: 1 if(pd.notnull(x)) else 0)
        df = self.procedure_update(df)
        df = self.diagnosis_count(df)
        none_features = ['ClmDiagnosisCode_1','ClmDiagnosisCode_2','ClmDiagnosisCode_3','ClmDiagnosisCode_4',\
                        'ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7','ClmDiagnosisCode_8',\
                    'ClmDiagnosisCode_9','ClmDiagnosisCode_10','OperatingPhysician',\
                    'OtherPhysician','AttendingPhysician']
        df = self.fill_none_missing(df,none_features)
        df.drop(['DiagnosisGroupCode'], inplace=True, axis=1)
        
        return df

    def update_outpatient(self, df):
        '''
        Update procedure and diagnosis claim columns
        Impute NA with 'None'
        '''

        df = self.procedure_update(df)
        df = self.diagnosis_count(df)
        df['ClmAdmitDiagnosisCode'] = df['ClmAdmitDiagnosisCode'].apply(lambda x: 1 if(pd.notnull(x)) else 0)
        none_features = ['ClmDiagnosisCode_1','ClmDiagnosisCode_2','ClmDiagnosisCode_3','ClmDiagnosisCode_4',\
                        'ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7','ClmDiagnosisCode_8',\
                    'ClmDiagnosisCode_9','ClmDiagnosisCode_10','OperatingPhysician',\
                    'OtherPhysician','AttendingPhysician']
        df = self.fill_none_missing(df, none_features)
        return df

    def update_train(self, df):
        '''
        Transform PotentialFraud - replace Yes, No with 1, 0
        Update to pass for test data.
        '''
        train_col = ['PotentialFraud']
        for col in train_col:
            if col not in df.columns:
                pass
            else:
                df['PotentialFraud'] = df['PotentialFraud'].replace(['Yes','No'], [1,0])
        return df

    def update_bene(self, df):
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
            
        df['num_chronic'] = df[[x for x in df.columns if "ChronicCond" in x]].sum(axis = "columns")

        df = self.fill_none_missing(df,['DOD'])

        state_codes = [5, 10, 33, 45, 14, 39]

        for col in state_codes:
            df[str(col)+'_state'] = np.where((df['State'].eq(col)), 1, 0)

        return df

    def combine_df(self):
        '''
        Consolidate all 4 csv/dataframes into one final dataframe
        '''

        inpatient, outpatient, bene, train = self.extract_data()
        inpatient = self.update_inpatient(inpatient)
        outpatient = self.update_outpatient(outpatient)
        train = self.update_train(train)
        bene = self.update_bene(bene)

        patients = pd.concat([inpatient, outpatient], axis=0, ignore_index=True, sort=False)
        patients['Inpatient'] = patients['Inpatient'].fillna(0)

        treatment_duration = patients.DischargeDt - patients.AdmissionDt
        patients['treatment_days'] = treatment_duration / np.timedelta64(1, 'D')
        patients['treatment_days'] = patients['treatment_days'].fillna(0)
        patients['treatment_days'] = pd.to_numeric(patients['treatment_days'])

        pat_none_feat = ['AdmissionDt','DischargeDt']
        patients = self.fill_none_missing(patients,pat_none_feat)

        merge_1 = pd.merge(left = patients, right = bene, on = 'BeneID')
        merge_2 = pd.merge(left = merge_1, right = train, on = 'Provider')
        merge_2['Age'] = merge_2.apply(lambda x: relativedelta(x['ClaimStartDt'], x['DOB']).years, axis=1)
        merge_2.drop(['DOB'], inplace=True, axis=1)

        merge_2 = self.duplicate_claims(merge_2)

        diag_lst = ['ClmDiagnosisCode_1','ClmDiagnosisCode_2','ClmDiagnosisCode_3','ClmDiagnosisCode_4',\
                        'ClmDiagnosisCode_5','ClmDiagnosisCode_6','ClmDiagnosisCode_7','ClmDiagnosisCode_8',\
                    'ClmDiagnosisCode_9','ClmDiagnosisCode_10']
                    

        proc_lst = ['ClmProcedureCode_1','ClmProcedureCode_2','ClmProcedureCode_3','ClmProcedureCode_4',\
                        'ClmProcedureCode_5','ClmProcedureCode_6']

        proc_cols = [x for x in merge_2.columns if "ClmProcedureCode" in x]
        merge_2.drop(merge_2[diag_lst], inplace=True, axis=1)
        merge_2.drop(merge_2[proc_lst], inplace=True, axis=1)



        return merge_2  
        
    
    # The order for provider_fraud.py -r
    # Train_Inpatientdata.csv Train_Outpatientdata.csv Train_Beneficiarydata.csv Train.csv

