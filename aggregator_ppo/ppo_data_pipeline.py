import argparse
import os
import glob
import pandas as pd
import numpy as np

# Constants from the paper and ns-3 configuration
CELL_LIST = [2, 3, 4, 5, 6, 7, 8]  # The 7 mmWave gNBs
TOTAL_PRBS = 139.0                 # Assumed PRBs from hierarchical_env.py
P_TX = 30.0                        # Example Transmit Power (Watts) - Adjust to your ns-3 config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract ns-3 traces into a DRL-ready CSV dataset.")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Path to the simulation output folder")
    parser.add_argument('-o', '--output_csv', type=str, required=True, help="Path to save the generated dataset CSV")
    return parser.parse_args()

def get_time_mapper(input_dir):
    """Creates a function to map UNIX milliseconds back to Simulation Seconds."""
    state_file = os.path.join(input_dir, 'bsState.txt')
    if not os.path.exists(state_file):
        return lambda x: x
        
    # Read both Timestamp and UNIX to establish the baseline
    df = pd.read_csv(state_file, sep='\s+', usecols=['Timestamp', 'UNIX'])
    if df.empty:
        return lambda x: x
        
    # Calculate the UNIX timestamp at Simulation Second 0.0
    first_row = df.iloc[0]
    sim_sec = first_row['Timestamp']
    unix_ms = first_row['UNIX']
    start_unix_ms = unix_ms - (sim_sec * 1000.0)
    
    def map_timestamp(ts):
        if ts > 1000000: # It's a UNIX timestamp
            return round((ts - start_unix_ms) / 1000.0, 1)
        return round(ts, 1) # Already simulation seconds
        
    return map_timestamp

def extract_bs_state(input_dir):
    """Extracts cell ON/OFF states and calculates activation costs."""
    state_file = os.path.join(input_dir, 'bsState.txt')
    if not os.path.exists(state_file):
        return pd.DataFrame()

    df = pd.read_csv(state_file, sep='\s+', usecols=['Timestamp', 'Id', 'State'])
    df['Timestamp'] = df['Timestamp'].round(1)
    
    df_pivoted = df.pivot(index='Timestamp', columns='Id', values='State').fillna(1.0)
    
    cost_data = []
    active_duration = {cell: 0.0 for cell in CELL_LIST}
    
    for timestamp, row in df_pivoted.iterrows():
        step_costs = {}
        for cell in CELL_LIST:
            state = row.get(cell, 1.0)
            if state == 1.0:
                active_duration[cell] += 100.0  # 100ms periodicity
                cost = 0.9 ** (0.01 * active_duration[cell])
            else:
                active_duration[cell] = 0.0
                cost = 0.0
            
            step_costs[f'State_Cell_{cell}'] = state
            step_costs[f'Cost_Cell_{cell}'] = cost
            
        step_costs['Total_BsON'] = sum(row.get(c, 1.0) for c in CELL_LIST)
        step_costs['Timestamp'] = timestamp
        cost_data.append(step_costs)
        
    return pd.DataFrame(cost_data).set_index('Timestamp')

def extract_throughput(input_dir, time_mapper):
    """Extracts PDCP Volume (Throughput) per cell from cu-up files."""
    cu_files = glob.glob(os.path.join(input_dir, 'cu-up-cell-*.txt'))
    dfs = []
    
    for file in cu_files:
        cell_id = int(file.split('-cell-')[-1].split('.txt')[0])
        if cell_id not in CELL_LIST:
            continue
            
        df = pd.read_csv(file)
        if 'timestamp' in df.columns and 'QosFlow.PdcpPduVolumeDL_Filter' in df.columns:
            # Standardize time to Simulation Seconds
            df['timestamp'] = df['timestamp'].apply(time_mapper)
            df_agg = df.groupby('timestamp')['QosFlow.PdcpPduVolumeDL_Filter'].sum().rename(f'Throughput_Cell_{cell_id}')
            dfs.append(df_agg)
            
    if dfs:
        throughput_df = pd.concat(dfs, axis=1)
        throughput_df.index.name = 'Timestamp'
        return throughput_df
    return pd.DataFrame()

def extract_du_metrics(input_dir, time_mapper):
    """Extracts Throughput, PRB usage, and Modulation metrics from du-cell files."""
    du_files = glob.glob(os.path.join(input_dir, 'du-cell-*.txt'))
    dfs = []
    
    for file in du_files:
        cell_id = int(file.split('-cell-')[-1].split('.txt')[0])
        if cell_id not in CELL_LIST:
            continue
            
        df = pd.read_csv(file)
        if 'timestamp' not in df.columns:
            continue
            
        df['timestamp'] = df['timestamp'].apply(time_mapper)
        
        # Safely aggregate columns if they exist in the file
        agg_dict = {}
        if 'RRU.PrbUsedDl' in df.columns: agg_dict['RRU.PrbUsedDl'] = 'sum'
        if 'TB.TotNbrDlInitial.64Qam' in df.columns: agg_dict['TB.TotNbrDlInitial.64Qam'] = 'sum'
        if 'TB.TotNbrDlInitial' in df.columns: agg_dict['TB.TotNbrDlInitial'] = 'sum'
        if 'TB.TotNbrDl.1' in df.columns: agg_dict['TB.TotNbrDl.1'] = 'sum'
        if 'QosFlow.PdcpPduVolumeDL_Filter' in df.columns: agg_dict['QosFlow.PdcpPduVolumeDL_Filter'] = 'sum' # THE FIX
        
        if not agg_dict:
            continue
            
        df_agg = df.groupby('timestamp').agg(agg_dict)
        
        # Rename columns to match our 85-element state vector
        rename_dict = {
            'RRU.PrbUsedDl': f'PRB_Num_Cell_{cell_id}',
            'TB.TotNbrDlInitial.64Qam': f'QAM64_Num_Cell_{cell_id}',
            'TB.TotNbrDlInitial': f'TotalPDUs_Cell_{cell_id}',
            'TB.TotNbrDl.1': f'EnergyPDUs_Cell_{cell_id}',
            'QosFlow.PdcpPduVolumeDL_Filter': f'Throughput_Cell_{cell_id}' # THE FIX
        }
        df_agg = df_agg.rename(columns=rename_dict)
        dfs.append(df_agg)
        
    if dfs:
        du_df = pd.concat(dfs, axis=1)
        du_df.index.name = 'Timestamp'
        return du_df
    return pd.DataFrame()

def extract_rlf(input_dir, time_mapper):
    """Extracts Radio Link Failures (SINR < -5dB)."""
    rlf_file = os.path.join(input_dir, 'rlf_metrics.csv')
    if not os.path.exists(rlf_file):
        return pd.DataFrame()
        
    df = pd.read_csv(rlf_file)
    df['Timestamp'] = df['Timestamp'].apply(time_mapper)
    df_pivoted = df.pivot(index='Timestamp', columns='CellID', values='BadUEs').add_prefix('RLF_Num_Cell_')
    return df_pivoted

def build_derivative_metrics(merged_df):
    """Calculates the derivative KPMs required by the paper to reach the 85 state elements."""
    final_df = merged_df.copy()
    
    for cell in CELL_LIST:
        energy_pdus = final_df.get(f'EnergyPDUs_Cell_{cell}', 0)
        final_df[f'Energy_Cell_{cell}'] = energy_pdus * P_TX
        
        throughput = final_df.get(f'Throughput_Cell_{cell}', 0)
        energy = final_df[f'Energy_Cell_{cell}']
        final_df[f'EE_Ratio_Cell_{cell}'] = throughput / (energy + 1e-5)
        
        rlf_num = final_df.get(f'RLF_Num_Cell_{cell}', 0)
        final_df[f'RLF_Pct_Cell_{cell}'] = rlf_num / 9.0 
        
        prb_num = final_df.get(f'PRB_Num_Cell_{cell}', 0)
        final_df[f'PRB_Pct_Cell_{cell}'] = prb_num / TOTAL_PRBS
        
        qam64_num = final_df.get(f'QAM64_Num_Cell_{cell}', 0)
        total_pdus = final_df.get(f'TotalPDUs_Cell_{cell}', 1e-5)
        final_df[f'QAM64_Pct_Cell_{cell}'] = qam64_num / (total_pdus + 1e-5)
        
        final_df[f'PhyBytes_Cell_{cell}'] = throughput * 1.1  
        
    return final_df

def main():
    args = parse_arguments()
    input_dir = args.input_dir
    output_csv = args.output_csv
    
    print(f"Extracting data from: {input_dir}")
    
    # 0. Create Time Mapper to sync UNIX MS to Sim Seconds
    time_mapper = get_time_mapper(input_dir)
    
    # 1. Extract raw data
    df_state = extract_bs_state(input_dir)
    #df_throughput = extract_throughput(input_dir, time_mapper)
    df_du = extract_du_metrics(input_dir, time_mapper)
    df_rlf = extract_rlf(input_dir, time_mapper)
    
    # 2. Align everything on the Timestamp index
    dataframes = [df for df in [df_state, df_du, df_rlf] if not df.empty]
    if not dataframes:
        print("No valid data found in directory.")
        return
        
    merged_df = pd.concat(dataframes, axis=1).sort_index()
    
    # Forward-fill state columns in case of missing log lines, fill initial with 1.0 (ON)
    state_cols = [c for c in merged_df.columns if 'State_Cell' in c or 'Cost_Cell' in c or c == 'Total_BsON']
    merged_df[state_cols] = merged_df[state_cols].ffill()
    for col in state_cols:
        if 'State' in col: merged_df[col] = merged_df[col].fillna(1.0)
        elif 'Cost' in col: merged_df[col] = merged_df[col].fillna(0.0)
        elif 'BsON' in col: merged_df[col] = merged_df[col].fillna(len(CELL_LIST))
        
    # Fill remaining metric NaNs with 0
    merged_df = merged_df.fillna(0.0)
    
    # 3. Calculate Derivative Metrics
    final_dataset = build_derivative_metrics(merged_df)
    
    # 4. Clean up and select the exact 85 columns
    expected_columns = ['Total_BsON']
    for cell in CELL_LIST:
        expected_columns.extend([
            f'State_Cell_{cell}', f'Throughput_Cell_{cell}', f'Energy_Cell_{cell}',
            f'EE_Ratio_Cell_{cell}', f'RLF_Num_Cell_{cell}', f'RLF_Pct_Cell_{cell}',
            f'PRB_Num_Cell_{cell}', f'PRB_Pct_Cell_{cell}', f'QAM64_Num_Cell_{cell}',
            f'QAM64_Pct_Cell_{cell}', f'PhyBytes_Cell_{cell}', f'Cost_Cell_{cell}'
        ])
    
    for col in expected_columns:
        if col not in final_dataset.columns:
            final_dataset[col] = 0.0
            
    final_dataset = final_dataset[expected_columns]
    
    final_dataset.to_csv(output_csv)
    print(f"Successfully saved {len(final_dataset)} rows and {len(final_dataset.columns)} columns to {output_csv}")

if __name__ == "__main__":
    main()