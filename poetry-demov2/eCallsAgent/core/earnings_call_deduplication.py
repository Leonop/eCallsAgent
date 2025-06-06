import pandas as pd
import os
from datetime import datetime
import sys
from eCallsAgent.core import data_handler

# Function to determine quarter from date
def get_quarter(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    month = date_obj.month
    quarter = (month - 1) // 3 + 1
    return quarter

def deduplicate_ecc_transcripts():
    # Load the data
    dh = data_handler.load_data()

    try:
        df = dh.load_data()
        print(f"Successfully loaded data with {len(df)} rows")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

    # Print initial shape
    print(f"Initial data shape: {df.shape}")

    # Add quarter column based on mostimportantdateutc
    try:
        df['quarter'] = df['mostimportantdateutc'].apply(get_quarter)
    except Exception as e:
        print(f"Error creating quarter column: {e}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Sort by mostimportantdateutc (ascending) to keep the earliest occurrence
    df = df.sort_values(['transcriptid', 'companyid', 'mostimportantdateutc', 'transcriptcomponenttypename', 'componentorder'], ascending=True)
    columns_to_keep = ['transcriptid', 'companyid', 'year', 'quarter', 'mostimportantdateutc', 'transcriptcomponenttypename', 'componentorder']
    # Group by call_id and keep the first occurrence of each call
    df_unique_calls = df[[columns_to_keep]].drop_duplicates(subset=['companyid', 'year', 'quarter', 'transcriptcomponenttypename', 'componentorder'], keep='first')
    df_unique_calls['text'] = df.groupby(['companyid', 'year', 'quarter', 'transcriptcomponenttypename', 'componentorder'])['componenttext'].transform('last')

    print(f"There are {len(df_unique_calls)} unique calls")

    # Print final shape
    print(f"Final data shape after deduplication: {df_unique_calls.shape}")

    # Save to output file
    output_file = f"ecc_transcripts_deduplicated_{gl.start_year}_{gl.end_year}.csv"
    final_df = df_unique_calls
    final_df.to_csv(output_file, index=False)
    print(f"Deduplicated data saved to {output_file}")

# Quick summary
print("\nSummary statistics:")
print(f"Original number of rows: {df.shape[0]}")
print(f"Number of unique calls: {len(call_ids)}")
print(f"Final number of rows after deduplication: {final_df.shape[0]}")
print(f"Reduction: {df.shape[0] - final_df.shape[0]} rows ({((df.shape[0] - final_df.shape[0]) / df.shape[0] * 100):.2f}%)") 