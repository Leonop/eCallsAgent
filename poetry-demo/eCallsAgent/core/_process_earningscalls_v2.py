import pandas as pd
import os
from datetime import datetime
from eCallsAgent.core import data_handler
from eCallsAgent.config import global_options as gl

def process_earnings_calls(data_file_path: str, start_year: int, end_year: int):
    # Load the data
    dh = data_handler.DataHandler(data_file_path, start_year, end_year)
    df = dh.load_data()
    print(f"Successfully loaded data with {len(df)} rows")  
    df['mostimportantdateutc'] = pd.to_datetime(df['mostimportantdateutc'])
    # 1. Add quarter column based on mostimportantdateutc
    df['quarter'] = df['mostimportantdateutc'].dt.quarter
    df['year'] = df['mostimportantdateutc'].dt.year
    df['componentorder'] = df['componentorder'].astype(int)

    
    from tqdm import tqdm
    tqdm.pandas()

    # Sort the dataframe
    df = df.sort_values(['transcriptid', 'companyid', 'mostimportantdateutc', 'transcriptcomponenttypename', 'componentorder'], ascending=True)

    # Define the grouping columns
    groupby_cols = ['companyid', 'year', 'quarter', 'transcriptcomponenttypename', 'componentorder']

    rows_to_keep = ['Presenter Speech', 'Question', 'Answer']
    # Create a new dataframe with first occurrence of metadata and last occurrence of text
    result = []
    count_skipped = 0
    for name, group in tqdm(df.groupby(groupby_cols), total=len(df), colour="green"):
        # Get the first row for metadata
        if group['transcriptcomponenttypename'].iloc[0] in rows_to_keep:
            first_row = group.iloc[0].copy()
            # Replace the text with the last row's text
            first_row['componenttext'] = group['componenttext'].iloc[-1]
            first_row['transcriptid'] = group['transcriptid'].iloc[0]
            result.append(first_row)
        else:
            count_skipped += 1
    print(f"Skipped {count_skipped} rows for not in rows_to_keep")

    df_unique_calls = pd.DataFrame(result)
    print(f"There are {len(df_unique_calls)} unique calls")
    save_data(df_unique_calls, 'transcriptid', 'componenttext', start_year, end_year)
    return df_unique_calls

def save_data(df: pd.DataFrame, id_name: str, column_name: str, start_year: int, end_year: int):
    # Create output directory if it doesn't exist
    os.makedirs(gl.output_folder, exist_ok=True)
    
    # Create a new directory for each transcriptid
    output_file = os.path.join(gl.output_folder, f'{column_name}_{start_year}_{end_year}.csv')
    with open(output_file, 'w') as f:
        for id_value in df[id_name].unique():
            # Select the rows for this ID and get just the column_name as a series
            subset = df[df[id_name] == id_value][column_name]
            # Write the concatenated text with newlines
            if not subset.empty:
                f.write(subset.str.cat(sep='\n'))

# if __name__ == "__main__":
    
#     # Process the dataset
#     result_df = process_earnings_calls()
    
#     # Save the processed data
#     output_file = f"processed_earnings_calls_{gl.start_year}_{gl.end_year}.csv"
#     result_df.to_csv(output_file, index=False)
#     print(f"Processed data saved to {output_file}") 