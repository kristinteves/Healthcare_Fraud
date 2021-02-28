
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import networkx as nx

#Read the data and convert date columns to datetime type
parse_dates = ['ClaimStartDt','DOB','ClaimEndDt','AdmissionDt','DOD','DischargeDt']
data = pd.read_csv('sara_preprocess.csv', parse_dates=parse_dates)


def num_chronic (data):
	"""
    This function creates a feature for the number of chronic conditions.
    """
	data['num_chronic'] = data[[x for x in data.columns if "ChronicCond" in x]].sum(axis = "columns")

	return data

def age (data):
	"""
    This function creates a feature for Age.
    """
	data['Age'] = data.apply(lambda x: relativedelta(x['ClaimStartDt'], x['DOB']).years, axis=1)

	return data

def diag_codes (data):
	"""
    This function creates features for the most frequent diagnosis codes.
    """
	diag_codes = ['4019','25000','2724','V5869','4011','42731','V5861','2720','2449','4280','53081','41401',\
                   '496','2859','41400','78079','5990','28521','3051','2809','311','73300','58881','71590',\
                   '5859','V4581','2722']
	diag_cols = [x for x in data.columns if "ClmDiagnosisCode" in x]

	for col in diag_codes:
		data[str(col)+'_diag'] = np.where((data[diag_cols].eq(col)).any(1, skipna=True), 1, 0)
	return data

def proc_codes (data):
	"""
    This function creates features for the most frequent procedure codes.
    """
	proc_codes = [4019, 9904, 2724, 8154, 66, 3893, 3995, 4516, 3722, 8151, 8872]
	proc_cols = [x for x in data.columns if "ClmProcedureCode" in x]

	for col in proc_cols:
		data[col] = pd.to_numeric(data[col], errors='coerce').astype(pd.Int64Dtype())

	for col in proc_codes:
		data[str(col)+'_proc'] = np.where((data[proc_cols].eq(col)).any(1, skipna=True), 1, 0)
	return data

def duplicate_claims (data):
    """
    This function creates a feature that codes for whether a claim is a duplicate.
    """
    duplicate_columns = [col for col in data.columns if 'ClmDiagnosis' in col or 'ClmProcedureCode'in col 
                    or 'ClmAdmitDiagnosisCode' in col]
    data['duplicate'] = data.duplicated(duplicate_columns, keep=False)
    data['duplicate'] = [1 if x == True else 0 for x in data.duplicate]
    
    return data


def duration(data):
    """
    This function creates new features for claim duration and treatment duration in number of days.
    NOTE: This code requires date columns to be of datetime64 dtype, so data needs to be read with parse_dates code.
    """
    claim_duration = data.ClaimEndDt - data.ClaimStartDt
    data['claim_days'] = claim_duration / np.timedelta64(1, 'D')
    
    treatment_duration = data['DischargeDt'] - data['AdmissionDt']
    data['treatment_days'] = treatment_duration / np.timedelta64(1, 'D')

    data['treatment_days'] = data['treatment_days'].fillna('None')
    
    data['treatment_days'] = [0 if x == 'None' else x for x in data['treatment_days']]

    return data


def network_connections(data):
    """
    This function conducts social network analysis and returns two added features to the data set:
    1) Number of physicians connected to each provider
    2) Number of patients connected to each provider
    """
    Full_soc_dataProv = data.groupby(['Provider','AttendingPhysician'])['IPAnnualReimbursementAmt'].count().reset_index()
    Full_soc_dataPatient = data.groupby(['Provider','BeneID'])['IPAnnualReimbursementAmt'].count().reset_index()
    
    G1 = nx.Graph()
    G1 = nx.from_pandas_edgelist(Full_soc_dataProv, 'Provider','AttendingPhysician')
    prov_phys_degree = pd.DataFrame(G1.degree)
    prov_phys_degree.columns = ['Provider','AttPhys_Connections']
    prov_phys_degree2 = prov_phys_degree[prov_phys_degree['Provider'].str.contains('PRV')]
    data = pd.merge(prov_phys_degree2, data, how="outer", on="Provider")
    
    G2 = nx.Graph()
    G2 = nx.from_pandas_edgelist(Full_soc_dataPatient, 'Provider','BeneID')
    prov_patient_degree = pd.DataFrame(G2.degree)
    prov_patient_degree.columns = ['Provider','Patient_Connections']
    prov_patient_degree2 = prov_patient_degree[prov_patient_degree['Provider'].str.contains('PRV')]
    data = pd.merge(prov_patient_degree2, data, how="outer", on="Provider")
    
    return data


def groupby_provider(data):
    '''
    This function converts DOD and gender to binary code and groups data by provider.
    '''
    
    num_claims = data.groupby('Provider')['ClaimID'].count()
    
    sum_features = data.groupby('Provider')[['DOD', 'ClmAdmitDiagnosisCode','Inpatient','ClmProcedureCode','ClmProcedureCode_count',\
                             'ClmDiagnosisCode_count', 'RenalDiseaseIndicator', 'duplicate','V5869_diag',\
                            'V5861_diag','2724_diag','4011_diag',\
                            '2449_diag','4019_diag','25000_diag','42731_diag',\
                            '4019_proc','9904_proc', '2724_proc', '8154_proc', '66_proc',\
                            '3893_proc', '3995_proc', '4516_proc', '3722_proc', '8151_proc',\
                            '8872_proc']].sum().reset_index() 
    
    mean_features = data.groupby('Provider')[['Patient_Connections','AttPhys_Connections','InscClaimAmtReimbursed',\
                            'DeductibleAmtPaid','treatment_days','Gender', 'IPAnnualReimbursementAmt',\
                                  'NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 'IPAnnualDeductibleAmt',\
                                  'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'num_chronic',\
                                  'Age','claim_days']].mean().reset_index()
    
    data2 = pd.merge(num_claims, sum_features, how="outer", on="Provider")
    
    data3 = pd.merge(data2, mean_features, how="outer", on="Provider")
    
    return data3


def transform_data(data):
    
    data1 = num_chronic(data)
    data2 = age(data1)
    data3 = diag_codes(data2)
    data4 = proc_codes(data3)
    data5 = duplicate_claims(data4)
    data6 = duration(data5)
    data7 = network_connections(data6)
    data8 = groupby_provider(data7)

    return data8

data = transform_data(data)
data.to_csv('final_features.csv', index=False)


