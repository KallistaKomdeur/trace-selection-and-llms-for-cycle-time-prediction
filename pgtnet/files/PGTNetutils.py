# Provide the class name for graph dataset (see: PGTNetlogHandler.py)
def eventlog_class_provider(name_of_dataset):
    if name_of_dataset == "BPIC15_1":
        pyg_class_name = "EVENTBPIC15M1"
    elif name_of_dataset == "Confidential":
        pyg_class_name = "EVENTConfidential"
    elif name_of_dataset == "Production":
        pyg_class_name = "EVENTProduction"
    elif name_of_dataset == "Purchasing":
        pyg_class_name = "EVENTPurchasing"
    elif name_of_dataset == "Bpi12w":
        pyg_class_name = "EVENTBpi12w"
    elif name_of_dataset == "Bpi11Half":
        pyg_class_name = "EVENTBpi11Half"
    elif name_of_dataset == "Bpi11Calib":
        pyg_class_name = "EVENTBpi11Calib"
    elif name_of_dataset == "BPIC15_2":
        pyg_class_name = "EVENTBPIC15M2"
    elif name_of_dataset == "BPIC15_3":
        pyg_class_name = "EVENTBPIC15M3"
    elif name_of_dataset == "BPIC15_4":
        pyg_class_name = "EVENTBPIC15M4"
    elif name_of_dataset == "BPIC15_5":
        pyg_class_name = "EVENTBPIC15M5"
    elif name_of_dataset == "BPI_Challenge_2012":
        pyg_class_name = "EVENTBPIC12"
    elif name_of_dataset == "BPI_Challenge_2012A":
        pyg_class_name = "EVENTBPIC12A"
    elif name_of_dataset == "BPI_Challenge_2012O":
        pyg_class_name = "EVENTBPIC12O"
    elif name_of_dataset == "BPI_Challenge_2012W":
        pyg_class_name = "EVENTBPIC12W"
    elif name_of_dataset == "BPI_Challenge_2012C":
        pyg_class_name = "EVENTBPIC12C"
    elif name_of_dataset == "BPI_Challenge_2012CW":
        pyg_class_name = "EVENTBPIC12CW"
    elif name_of_dataset == "BPI_Challenge_2013C" or name_of_dataset == "2013C":
        pyg_class_name = "EVENTBPIC13C"
    elif name_of_dataset == "BPI_Challenge_2013I" or name_of_dataset == "2013I":
        pyg_class_name = "EVENTBPIC13I"
    elif name_of_dataset == "BPIC20_DomesticDeclarations" or name_of_dataset == "2020D":
        pyg_class_name = "EVENTBPIC20D"
    elif name_of_dataset == "BPIC20_InternationalDeclarations" or name_of_dataset == "2020I":
        pyg_class_name = "EVENTBPIC20I"
    elif name_of_dataset == "env_permit" or name_of_dataset.lower() == "envpermit":
        pyg_class_name = "EVENTEnvPermit"
    elif name_of_dataset == "HelpDesk" or name_of_dataset.lower() == "helpdesk":
        pyg_class_name = "EVENTHelpDesk"
    elif name_of_dataset == "Hospital" or name_of_dataset.lower() == "hospital":
        pyg_class_name = "EVENTHospital"
    elif name_of_dataset == "Sepsis" or name_of_dataset.lower() == "sepsis":
        pyg_class_name = "EVENTSepsis"
    elif name_of_dataset == "Traffic_Fines" or name_of_dataset.lower() == "trafficfines":
        pyg_class_name = "EVENTTrafficfines"
    elif name_of_dataset == "Baseline" or name_of_dataset == "baseline":
        pyg_class_name = "EVENTBaseline"
    elif name_of_dataset == "BaselineDrifted" or name_of_dataset == "baseline_drifted":
        pyg_class_name = "EVENTBaselineDrifted"
    elif name_of_dataset == "BPIC2011" or name_of_dataset == "bpic2011":
        pyg_class_name = "EVENTBPIC2011"
    elif name_of_dataset == "BPIC2015" or name_of_dataset == "bpic2015":
        pyg_class_name = "EVENTBPIC2015"
    elif name_of_dataset == "HelpDeskV2":
        pyg_class_name = "EVENTHelpDeskV2"
    elif name_of_dataset == "BPIC12V2":
        pyg_class_name = "EVENTBPIC12V2"
    elif name_of_dataset == "BPIC17":
        pyg_class_name = "EVENTBPIC17"
    elif name_of_dataset == "BPIC20DomesticV2":
        pyg_class_name = "EVENTBPIC20DomesticV2"
    elif name_of_dataset == "BPIC20InternationalV2":
        pyg_class_name = "EVENTBPIC20InternationalV2"
    elif name_of_dataset == "BPIC20Prepaid":
        pyg_class_name = "EVENTBPIC20Prepaid"
    elif name_of_dataset == "BPIC20RFP":
        pyg_class_name = "EVENTBPIC20RFP"
    elif name_of_dataset == "BPIC20Permit":
        pyg_class_name = "EVENTBPIC20Permit"
    elif name_of_dataset == "HospitalBilling":
        pyg_class_name = "EVENTHospitalBilling"
    elif name_of_dataset == "TrafficFinesV2":
        pyg_class_name = "EVENTTrafficFinesV2"
    elif name_of_dataset == "BPIC20_InternationalDeclarations":
        pyg_class_name = "EVENTBPIC20IPaper"
    elif name_of_dataset == "RequestForPayment" or name_of_dataset == "BPIC20RFPPaper":
        pyg_class_name = "EVENTBPIC20RFPPaper"
    elif name_of_dataset == "travel_permit_data" or name_of_dataset == "BPIC20PermitPaper":
        pyg_class_name = "EVENTBPIC20PermitPaper"
    else:
        pyg_class_name = None
        print('Error! no Pytorch Geometric dataset class is defined for this event log')
    return pyg_class_name


def mean_cycle_norm_factor_provider(dataset):
    
    if dataset.lower() == 'helpdesk' or ("EVENTHelpDesk" in dataset):
        mean_cycle = 40.90
        normalization_factor = 59.99496528
    elif dataset == "BPIC20_InternationalDeclarations" or dataset == "2020I" or ("EVENTBPIC20I" in dataset):
        mean_cycle = 94.70
        normalization_factor = 742
    elif dataset == "BPIC20_DomesticDeclarations" or dataset == "2020D" or ("EVENTBPIC20D" in dataset):
        mean_cycle = 11.50
        normalization_factor = 469.2363
    elif dataset.lower() == 'envpermit' or dataset == "env_permit" or ("EVENTEnvPermit" in dataset):
        mean_cycle = 5.41
        normalization_factor = 275.8396
    elif dataset == "BPI_Challenge_2013I" or dataset == '2013I' or ("EVENTBPIC13I" in dataset):
        mean_cycle = 12.08
        normalization_factor = 771.351770833333
    elif dataset == "BPI_Challenge_2013C" or dataset == '2013C' or ("EVENTBPIC13C" in dataset):
        mean_cycle = 178.88
        normalization_factor = 2254.84850694444 
    elif dataset == "BPI_Challenge_2012" or dataset == '2012' or ("EVENTBPIC12" in dataset):
        mean_cycle = 8.60
        normalization_factor = 137.22148162037
    elif dataset == "BPI_Challenge_2012C" or dataset == '2012C' or ("EVENTBPIC12C" in dataset):
        mean_cycle = 8.61
        normalization_factor = 91.4552796412037
    elif dataset == "BPI_Challenge_2012W" or dataset == '2012W' or ("EVENTBPIC12W" in dataset):
        mean_cycle = 11.70
        normalization_factor = 137.220982743055
    elif dataset == "BPI_Challenge_2012CW" or dataset == '2012CW' or ("EVENTBPIC12CW" in dataset):
        mean_cycle = 11.40
        normalization_factor = 91.040850324074
    elif dataset == "BPI_Challenge_2012O" or dataset == '2012O' or ("EVENTBPIC12O" in dataset):
        mean_cycle = 17.18
        normalization_factor = 89.5486824537037
    elif dataset == "BPI_Challenge_2012A" or dataset == '2012A' or ("EVENTBPIC12A" in dataset):
        mean_cycle = 8.08
        normalization_factor = 91.4552796412037
    elif dataset == "BPIC15_1" or dataset == '2015m1' or ("EVENTBPIC15M1" in dataset):
        mean_cycle = 95.90
        normalization_factor = 1486
    elif dataset == "BPIC15_2" or dataset == '2015m2' or ("EVENTBPIC15M2" in dataset):
        mean_cycle = 160.30
        normalization_factor = 1325.9583
    elif dataset == "BPIC15_3" or dataset == '2015m3' or ("EVENTBPIC15M3" in dataset):
        mean_cycle = 62.20
        normalization_factor = 1512
    elif dataset == "BPIC15_4" or dataset == '2015m4' or ("EVENTBPIC15M4" in dataset):
        mean_cycle = 116.90
        normalization_factor = 926.9583
    elif dataset == "BPIC15_5" or dataset == '2015m5' or ("EVENTBPIC15M5" in dataset):
        mean_cycle = 98
        normalization_factor = 1343.9583
    elif dataset.lower() == 'sepsis' or ("EVENTSepsis" in dataset):
        mean_cycle = 28.48
        normalization_factor = 422.323946759259
    elif dataset.lower() == 'trafficfines' or dataset == "Traffic_Fines" or  ("EVENTTrafficfines" in dataset):
        mean_cycle = 341.60
        normalization_factor = 4372
    elif dataset.lower() == 'hospital' or ("EVENTHospital" in dataset):
        mean_cycle = 127.24
        normalization_factor = 1035.4212037037
    elif dataset == "Baseline" or dataset == "baseline" or dataset == "EVENTBaseline":
        mean_cycle = 1.27
        normalization_factor = 13.6446
    elif dataset == "BaselineDrifted" or dataset == "baseline_drifted" or ("EVENTBaselineDrifted" in dataset):
        mean_cycle = 1.40
        normalization_factor = 13.6446
    elif dataset == "BPIC2011" or dataset == "bpic2011" or ("EVENTBPIC2011" in dataset):
        mean_cycle = 418.06
        normalization_factor = 1083.0
    elif dataset == "BPIC2015" or dataset == "bpic2015" or ("EVENTBPIC2015" in dataset):
        mean_cycle = 59.47
        normalization_factor = 1260.4645
    elif dataset == "HelpDeskV2" or ("EVENTHelpDeskV2" in dataset):
        mean_cycle = 41.11
        normalization_factor = 59.994965
    elif dataset == "BPIC20Prepaid" or ("EVENTBPIC20Prepaid" in dataset):
        mean_cycle = 40.09
        normalization_factor = 324.975995
    elif dataset == "BPIC20InternationalV2" or ("EVENTBPIC20InternationalV2" in dataset):
        mean_cycle = 94.70
        normalization_factor = 742.0
    elif dataset == "BPIC20RFP" or ("EVENTBPIC20RFP" in dataset):
        mean_cycle = 11.92
        normalization_factor = 406.0347
    elif dataset == "BPIC20Permit" or ("EVENTBPIC20Permit" in dataset):
        mean_cycle = 91.49
        normalization_factor = 1190.3287
    elif dataset == "BPIC20DomesticV2" or ("EVENTBPIC20DomesticV2" in dataset):
        mean_cycle = 11.29
        normalization_factor = 249.9805
    elif dataset == "BPIC12V2" or ("EVENTBPIC12V2" in dataset):
        mean_cycle = 9.05
        normalization_factor = 91.4136
    elif dataset == "HospitalBilling" or ("EVENTHospitalBilling" in dataset):
        mean_cycle = 67.15
        normalization_factor = 256.1249
    elif dataset == "BPIC17" or ("EVENTBPIC17" in dataset):
        mean_cycle = 22.11
        normalization_factor = 286.0724
    elif dataset == "TrafficFinesV2" or ("EVENTTrafficFinesV2" in dataset):
        mean_cycle = 351.73
        normalization_factor = 4372.0
    elif dataset == "BPIC20IPaper" or ("EVENTBPIC20IPaper" in dataset):
        mean_cycle = 86.50           # placeholder; updated after conversion
        normalization_factor = 742.0
    elif dataset == "BPIC20RFPPaper" or dataset == "RequestForPayment" or ("EVENTBPIC20RFPPaper" in dataset):
        mean_cycle = 11.92
        normalization_factor = 406.0347
    elif dataset == "BPIC20PermitPaper" or dataset == "travel_permit_data" or ("EVENTBPIC20PermitPaper" in dataset):
        mean_cycle = 91.49
        normalization_factor = 1190.3287

    else:
        print('Dataset is not recognized')
        mean_cycle = None
        normalization_factor = None

    return normalization_factor, mean_cycle

def eventlog_name_provider(name_of_class):
    if name_of_class == "EVENTBPIC15M1":
        event_log_name = "BPIC15_1"
    elif name_of_class == "EVENTBPIC15M2":
        event_log_name = "BPIC15_2"
    elif name_of_class == "EVENTBPIC15M3":
        event_log_name = "BPIC15_3"
    elif name_of_class == "EVENTBPIC15M4":
        event_log_name = "BPIC15_4"
    elif name_of_class == "EVENTBPIC15M5":
        event_log_name = "BPIC15_5"
    elif name_of_class == "EVENTBPIC12":
        event_log_name = "BPI_Challenge_2012"
    elif name_of_class == "EVENTBPIC12A":
        event_log_name = "BPI_Challenge_2012A"
    elif name_of_class == "EVENTBPIC12O":
        event_log_name = "BPI_Challenge_2012O"
    elif name_of_class == "EVENTBPIC12W":
        event_log_name = "BPI_Challenge_2012W"
    elif name_of_class == "EVENTBPIC12C":
        event_log_name = "BPI_Challenge_2012C"
    elif name_of_class == "EVENTBPIC12CW":
        event_log_name = "BPI_Challenge_2012CW"
    elif name_of_class == "EVENTBPIC13C":
        event_log_name = "BPI_Challenge_2013C"
    elif name_of_class == "EVENTBPIC13I":
        event_log_name = "BPI_Challenge_2013I"
    elif name_of_class == "EVENTBPIC20D":
        event_log_name = "BPIC20_DomesticDeclarations"
    elif name_of_class == "EVENTBPIC20I":
        event_log_name = "BPIC20_InternationalDeclarations" 
    elif name_of_class == "EVENTEnvPermit":
        event_log_name = "env_permit"
    elif name_of_class == "EVENTBaseline":
        event_log_name = "Baseline"
    elif name_of_class == "EVENTBaselineDrifted":
        event_log_name = "BaselineDrifted"
    elif name_of_class == "EVENTBPIC2011":
        event_log_name = "BPIC2011"
    elif name_of_class == "EVENTBPIC2015":
        event_log_name = "BPIC2015"
    elif name_of_class == "EVENTHelpDesk":
        event_log_name = "HelpDesk"
    elif name_of_class == "EVENTHospital":
        event_log_name = "Hospital"
    elif name_of_class == "EVENTSepsis":
        event_log_name = "Sepsis"
    elif name_of_class == "EVENTTrafficfines":
        event_log_name = "Traffic_Fines"
    elif name_of_class == "EVENTHelpDeskV2":
        event_log_name = "HelpDeskV2"
    elif name_of_class == "EVENTBPIC20Prepaid":
        event_log_name = "BPIC20Prepaid"
    elif name_of_class == "EVENTBPIC20InternationalV2":
        event_log_name = "BPIC20InternationalV2"
    elif name_of_class == "EVENTBPIC20RFP":
        event_log_name = "BPIC20RFP"
    elif name_of_class == "EVENTBPIC20Permit":
        event_log_name = "BPIC20Permit"
    elif name_of_class == "EVENTBPIC20DomesticV2":
        event_log_name = "BPIC20DomesticV2"
    elif name_of_class == "EVENTBPIC12V2":
        event_log_name = "BPIC12V2"
    elif name_of_class == "EVENTHospitalBilling":
        event_log_name = "HospitalBilling"
    elif name_of_class == "EVENTBPIC17":
        event_log_name = "BPIC17"
    elif name_of_class == "EVENTTrafficFinesV2":
        event_log_name = "TrafficFinesV2"
    elif name_of_class == "EVENTBPIC20IPaper":
        event_log_name = "BPIC20IPaper"
    elif name_of_class == "EVENTBPIC20RFPPaper":
        event_log_name = "BPIC20RFPPaper"
    elif name_of_class == "EVENTBPIC20PermitPaper":
        event_log_name = "BPIC20PermitPaper"
    else:
        event_log_name = None
        print('Error! no event log is related to this pythorch geometric dataset class.')
    return event_log_name