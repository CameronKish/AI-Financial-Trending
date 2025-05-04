import pandas as pd
import sys
from datetime import datetime

# --- Configuration ---
EXCEL_FILE_PATH = 'Fake Data Summary.xlsx' # <--- Keep your file path
PL_SHEET_NAME = 'PL-Wide'                  # <--- Keep your P&L sheet name

# *** MODIFIED: Define a LIST of JE sheet names ***
# !! IMPORTANT: Update this list with the ACTUAL names of ALL your JE sheets !!
JE_SHEET_NAMES = ['TXN-FY21', 'TXN-FY22', 'TXN-FY23'] # <--- EXAMPLE: Add all relevant sheet names here

# --- Column Name Definitions ---
# (Keep these matching your columns)
PL_ID_COLUMN = 'Account ID'
PL_MAP1_COLUMN = 'Mapping 1'
PL_MAP2_COLUMN = 'Mapping 2'

# --- *** ADD THESE TWO LINES to identify the lowest level mapping *** ---
PL_MAP_COLUMN = PL_MAP2_COLUMN  # Or PL_MAP_COLUMN = 'Mapping 2' - Use lowest level name col
JE_AMOUNT_COLUMN = 'Amount (Presentation Currency)' # Define the JE Amount column name
# --- *** END ADDITIONS *** ---

JE_ID_COLUMN = 'Account Number/Code'
JE_DATE_COLUMN = 'Transaction Date'
JE_DETAIL_COLUMNS = ["Transaction Id", JE_ID_COLUMN, JE_DATE_COLUMN, 'Account Name', 'Amount (Presentation Currency)', 'Memo', 'Customer']

# --- 1. Load Data ---

# --- Load P&L Data (No change here) ---
try:
    pl_wide_df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=PL_SHEET_NAME)
    # Use print for console feedback when running data_processor directly
    # In production with Streamlit, these prints go to the console, not the UI
    print(f"Successfully loaded P&L sheet: '{PL_SHEET_NAME}'")
except FileNotFoundError:
    print(f"CRITICAL Error: Excel file not found at '{EXCEL_FILE_PATH}'")
    sys.exit()
except ValueError as e:
    print(f"CRITICAL Error: P&L sheet name '{PL_SHEET_NAME}' not found or invalid. Details: {e}")
    sys.exit()
except Exception as e:
    print(f"CRITICAL An unexpected error occurred loading the P&L sheet: {e}")
    sys.exit()


# --- *** MODIFIED: Load MULTIPLE JE Sheets and Concatenate *** ---
all_je_dfs = []
successful_loads = []
failed_loads = []
print(f"\nAttempting to load JE sheets: {JE_SHEET_NAMES}")

for sheet in JE_SHEET_NAMES:
    try:
        # Read each specified JE sheet
        temp_je_df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet)
        all_je_dfs.append(temp_je_df)
        successful_loads.append(sheet)
        print(f"  Successfully loaded JE sheet: '{sheet}' ({len(temp_je_df)} rows)")
    except ValueError: # Handles sheet not found error specifically
         print(f"  Warning: JE sheet '{sheet}' not found in '{EXCEL_FILE_PATH}'. Skipping.")
         failed_loads.append(sheet)
    except Exception as e:
         print(f"  Warning: Error loading JE sheet '{sheet}': {e}. Skipping.")
         failed_loads.append(sheet)

# Check if any JE data was loaded
if not all_je_dfs:
    print(f"\nCRITICAL Error: No valid JE data sheets could be loaded from the specified list: {JE_SHEET_NAMES}.")
    sys.exit("Exiting due to missing JE data.")
else:
    # Concatenate all loaded JE DataFrames into one
    je_detail_df = pd.concat(all_je_dfs, ignore_index=True)
    print(f"\nSuccessfully combined {len(successful_loads)} JE sheets: {successful_loads}")
    print(f"Total JE rows combined: {len(je_detail_df)}")
    if failed_loads:
        print(f"Skipped sheets (not found or error): {failed_loads}")
# --- *** END MODIFICATION *** ---


# --- 2. Transform P&L Data (Wide to Flat) ---
# (No change needed here)
pl_id_vars = [PL_ID_COLUMN, PL_MAP1_COLUMN, PL_MAP2_COLUMN]
if not all(col in pl_wide_df.columns for col in pl_id_vars):
    print(f"CRITICAL Error: Missing required columns {pl_id_vars} in sheet '{PL_SHEET_NAME}'.")
    sys.exit()
try:
    pl_flat_df = pd.melt(pl_wide_df, id_vars=pl_id_vars, var_name='Period', value_name='Amount')
    print("\nSuccessfully transformed P&L data from wide to flat format.")
except Exception as e:
    print(f"CRITICAL Error during P&L data transformation (melt): {e}")
    sys.exit()


# --- 3. Clean Data & Prepare for Linking ---
# (No change needed here, applies to combined je_detail_df)
print("\nStarting Data Cleaning...")
# P&L Linking Column
if PL_ID_COLUMN in pl_flat_df.columns:
    pl_flat_df[PL_ID_COLUMN] = pl_flat_df[PL_ID_COLUMN].astype(str).str.strip()
else:
    print(f"CRITICAL Error: Linking column '{PL_ID_COLUMN}' unexpectedly missing after melt.")
    sys.exit()

# JE Linking Column
if JE_ID_COLUMN in je_detail_df.columns:
    je_detail_df[JE_ID_COLUMN] = je_detail_df[JE_ID_COLUMN].astype(str).str.strip()
else:
    print(f"CRITICAL Error: Linking column '{JE_ID_COLUMN}' not found in combined JE sheets. Cannot link data.")
    sys.exit()

# JE Date Column
if JE_DATE_COLUMN in je_detail_df.columns:
    try:
        original_dtype = je_detail_df[JE_DATE_COLUMN].dtype
        print(f"  Attempting to convert JE Date column ('{JE_DATE_COLUMN}') from type {original_dtype}...")
        je_detail_df[JE_DATE_COLUMN] = pd.to_datetime(je_detail_df[JE_DATE_COLUMN], errors='coerce')
        if je_detail_df[JE_DATE_COLUMN].isnull().any():
            print(f"  Warning: Some dates in '{JE_DATE_COLUMN}' could not be parsed and were set to NaT.")
        print(f"  Successfully converted JE Date column to datetime.")
    except Exception as e:
        print(f"CRITICAL Error converting JE date column '{JE_DATE_COLUMN}' to datetime: {e}")
        sys.exit()
else:
    print(f"CRITICAL Error: Date column '{JE_DATE_COLUMN}' not found in combined JE sheets. Cannot filter JEs by period.")
    sys.exit()

print("Data cleaning and type conversion complete.")


# --- 4. Define Lookup Function ---
# (No change needed here)
def get_journal_entries(account_id, period_str, je_df):
    """ Filters the JE DataFrame for a specific account ID and period string. """
    account_id = str(account_id).strip()
    try:
        period_date = pd.to_datetime(period_str, errors='coerce') # Should handle 'YYYY-MM' format
        if pd.isna(period_date):
             try: # Fallback for "Month YYYY" potentially
                 month_str, year_str = period_str.split()
                 period_date = pd.to_datetime(f"{month_str} 1 {year_str}", errors='coerce')
             except: # Final fallback if parsing fails completely
                 # print(f"Warning: Could not parse period string '{period_str}' in get_journal_entries.") # Removed print
                 # Return empty or filter only by account? Filter by account only might be confusing. Return empty.
                 return pd.DataFrame(columns=JE_DETAIL_COLUMNS)

        target_year = period_date.year
        target_month = period_date.month

        # Ensure date column is datetime type before filtering
        if not pd.api.types.is_datetime64_any_dtype(je_df[JE_DATE_COLUMN]):
             # This shouldn't happen if cleaning worked, but defensive check
             je_df[JE_DATE_COLUMN] = pd.to_datetime(je_df[JE_DATE_COLUMN], errors='coerce')

        # Filter JE DataFrame
        mask = (
            (je_df[JE_ID_COLUMN] == account_id) &
            (je_df[JE_DATE_COLUMN].dt.year == target_year) &
            (je_df[JE_DATE_COLUMN].dt.month == target_month) &
            (je_df[JE_DATE_COLUMN].notna()) # Exclude rows where date conversion failed
        )
        filtered_je = je_df.loc[mask].copy() # Use .loc[mask] for efficiency

        # Ensure only desired columns are returned
        relevant_cols = [col for col in JE_DETAIL_COLUMNS if col in filtered_je.columns]
        return filtered_je[relevant_cols]

    except Exception as e:
        # print(f"Error during JE filtering for {account_id}/{period_str}: {e}") # Removed print
        # Optionally log the error here instead of printing
        return pd.DataFrame(columns=JE_DETAIL_COLUMNS) # Return empty DataFrame on error


# --- 5. Ready for Interface ---
# (Print statements here are for console feedback when running this script directly)
print("\n--- data_processor.py finished ---")
print(f"Prepared 'pl_flat_df' ({len(pl_flat_df)} rows)")
print(f"Prepared 'je_detail_df' ({len(je_detail_df)} rows)")
print("\nP&L Flat DataFrame Head:")
#print(pl_flat_df.head())
# print("\nJE Detail DataFrame Head:")
# print(je_detail_df.head())

# --- Example Usage Removed ---
# The example usage block is removed as it's not needed when run via exec() in Streamlit