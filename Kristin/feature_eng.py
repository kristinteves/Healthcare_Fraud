import pandas as pd
import numpy as np
import networkx as nx

class FeatEng: 
    preprocessed_file = None

    def __init__(self, preprocessed_file):
        self.preprocessed_file = preprocessed_file


    def duration(self, data):
        """
        This function creates new features for claim duration and treatment duration in number of days.
        NOTE: This code requires date columns to be of datetime64 dtype, so data needs to be read with parse_dates code.
        """
        claim_duration = data.ClaimEndDt - data.ClaimStartDt
        data['claim_days'] = claim_duration / np.timedelta64(1, 'D')
        
        return data

    def network_connections(self, data):
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

    # def newdf_diag_highfreq(self, df):
    #     '''
    #     Create new features with ClmDiagnosisCodes that make upmore than 1%.
    #     Drop the ClmDiagnosisCode columns.
    #     '''
    #     columns = []
    #     for i in range(1,11):
    #         columns.append('ClmDiagnosisCode_'+str(i))
            
    #     freq_diag = ['V5869', 'V5861', '2724', '4011', '2449', '4019', '25000', '42731']    
    #     for col in freq_diag:
    #         df[str(col)+'_diagcode'] = np.where((df[columns].eq(col)).any(1, skipna=True), 1, 0)
        
    #     df = df.drop(df[columns],axis=1)
    #     return df

    def drop_PHY(self, df):
        '''
        Drop PHY data
        '''
        columns = ['AttendingPhysician','OperatingPhysician','OtherPhysician'] 
        
        # common_phys = ['PHY338032','PHY341578','PHY357120','PHY330576','PHY337425','PHY314027','PHY327046',\
        # 'PHY412132','PHY350277','PHY423534']
        
        df = df.drop(df[columns],axis=1)
        
        return df

    def groupby_provider(self, data):
        '''
        This function converts DOD and gender to binary code and groups data by provider.
        '''
        # data['DOD'] = [0 if x == 'None' else 1 for x in data['DOD']]
        
        data['Gender'] = [1 if x == 2 else 0 for x in data['Gender']]
        
        num_claims = data.groupby('Provider')['ClaimID'].count()
        
        sum_features = data.groupby('Provider')[['ClmAdmitDiagnosisCode','Inpatient','ClmProcedureCode','ClmProcedureCode_count',\
                             'ClmDiagnosisCode_count', 'RenalDiseaseIndicator', 'duplicate','V5869_diag',\
                            'V5861_diag','2724_diag','4011_diag',\
                            '2449_diag','4019_diag','25000_diag','42731_diag',\
                            '4019_proc','9904_proc', '2724_proc', '8154_proc', '66_proc',\
                            '3893_proc', '3995_proc', '4516_proc', '3722_proc', '8151_proc',\
                            '8872_proc','5_state','10_state','33_state','45_state','14_state',
                            '39_state']].sum().reset_index() 
    
        mean_features = data.groupby('Provider')[['Patient_Connections','AttPhys_Connections','InscClaimAmtReimbursed',\
                                'DeductibleAmtPaid','treatment_days','Gender', 'IPAnnualReimbursementAmt',\
                                    'NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 'IPAnnualDeductibleAmt',\
                                    'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'num_chronic',\
                                    'Age','claim_days']].mean().reset_index()
        
        data2 = pd.merge(num_claims, sum_features, how="outer", on="Provider")
        
        data3 = pd.merge(data2, mean_features, how="outer", on="Provider")
        
        return data3


    def feateng_update(self):
        '''
        Run and apply feature engineering updates from csv to one final dataframe.
        '''
        file_data = self.preprocessed_file
        if not file_data:
            raise Exception("Missing preprocessed csv file")

        extract_file_data = pd.read_csv(file_data[0].name, parse_dates=['ClaimStartDt','ClaimEndDt','AdmissionDt','DOD','DischargeDt'])

        # new_df = self.duplicate_claims(extract_file_data)
        new_df2 = self.duration(extract_file_data)
        new_df3 = self.network_connections(new_df2)
        new_df4 = self.newdf_diag_highfreq(new_df3)
        new_df5 = self.drop_PHY(new_df4)
        final_df = self.groupby_provider(new_df5)

        return final_df

    def feateng_update_steptwo(self):
        '''
        Run and apply feature engineering updates from dataframe to one final dataframe.
        '''

        process_data = self.preprocessed_file
        if process_data.empty:
            raise Exception("Error with preprocessed data")


        # new_df = self.duplicate_claims(process_data)
        new_df2 = self.duration(process_data)
        new_df3 = self.network_connections(new_df2)
        # new_df4 = self.newdf_diag_highfreq(new_df3)
        new_df5 = self.drop_PHY(new_df3)
        final_df = self.groupby_provider(new_df5)

        return final_df 

