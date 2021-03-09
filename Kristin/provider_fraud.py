import pandas as pd
import numpy as np
import os
import csv
import sys
import logging
import argparse
import preprocess
import feature_eng

import argparse

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

parser = argparse.ArgumentParser(description='Provider fraud')
parser.add_argument('--raw', '-r', type=argparse.FileType('r'), metavar='file.csv', nargs='*',
                    help='Input 4 csv files containing raw data in the following order: Inpatient, Outpatient, Beneficiary, Target (Train or Test data)')
parser.add_argument('--clean', '-c', type=argparse.FileType('r'), metavar='file.csv', nargs='*',
                    help='Input 1 already preprocessed csv file or leave blank if running --raw')
parser.add_argument('path', nargs='?', default=os.getcwd(), 
                    help='save output csv to path')

args = parser.parse_args()

run_raw = False
run_clean = False
run_raw_and_clean = False

# Determine what to run
if args.raw != None:
    # only raw, 4 input: run raw
    if len(args.raw) == 4:
        run_raw = True
    else:
        logger.error("--raw args requires 4 csv's")
        sys.exit()
if args.clean != None:
    # only clean, 1 input: run clean
    if not run_raw and len(args.clean) == 1:
        run_clean = True
        
    # raw 4 input, clean 0 input: run raw-> result-> run clean
    elif run_raw and len(args.clean) == 0:
        run_raw_and_clean = True
        
    # raw 4 input, clean 1 inpute: throw error
    elif run_raw and len(args.clean) != 0:
        logger.error("--raw is set to True and --clean recieved 1 input. To run both set --clean with 0 input")
        sys.exit()
    else:
        logger.error("--clean args requires 0 or 1 csv")
        sys.exit()

logger.debug(f"Running --raw: {run_raw}")
logger.debug(f"Running --clean: {run_clean}")
logger.debug(f"Running --raw and --clean: {run_raw_and_clean}")

if run_raw:
    if not run_raw_and_clean:
        p = preprocess.PreProcess(args.raw)
        p.combine_df().to_csv('Preprocess.csv', index=False)
        print('Preprocess complete. Preprocess.csv file saved')
    else:
        print('Two step process, preprocess and feature engineer, will run')

if run_clean:
    f = feature_eng.FeatEng(args.clean)
    f.feateng_update().to_csv('Clean_data.csv', index=False)
    print('Feature engineer complete. Clean_data.csv file saved')
    

if run_raw_and_clean:
    p = preprocess.PreProcess(args.raw) 

    # Code for debugging purposes
    # result = pd.read_csv('Train_Preprocess.csv',parse_dates=['ClaimStartDt','ClaimEndDt','AdmissionDt','DOD','DischargeDt'])
    # r = feature_eng.FeatEng(result)
    # r = r.feateng_update_steptwo()
    # print(r)
    
    result = p.combine_df()
    r = feature_eng.FeatEng(result)
    two_step_clean = r.feateng_update_steptwo()
    two_step_clean.to_csv('Clean_data.csv', index=False)
    print("Two step clean and feature engineer step complete. Clean_data.csv file saved")

    if result.empty:
        logger.error(f"result is {result}")
        sys.exit()
    print("Running clean function using raw results")
