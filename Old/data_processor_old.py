import pandas as pd
import sys
from datetime import datetime

# --- Configuration ---
EXCEL_FILE_PATH = 'Fake Data Summary.xlsx' # <--- CHANGE THIS to your file path
PL_SHEET_NAME = 'PL-Wide'
JE_SHEET_NAME = 'TXN-FYCOM'

# --- Column Name Definitions ---
# !! IMPORTANT: Adjust these names to match your actual Excel column headers !!
PL_ID_COLUMN = 'Account ID'        # Linking column in PL sheet
PL_MAP_COLUMN = 'Mapping'          # Descriptive column in PL sheet
JE_ID_COLUMN = 'Account Number/Code'         # Linking column in JE sheet
JE_DATE_COLUMN = 'Transaction Date' # Date column in JE sheet
# Add other JE columns you want to display later
JE_DETAIL_COLUMNS = ["Transaction Id", JE_ID_COLUMN, JE_DATE_COLUMN, 'Account Name', 'Amount (Presentation Currency)', 'Memo', 'Customer'] # Example JE columns

# --- 1. Load Data ---

try:
    # Load P&L Wide Data
    pl_wide_df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=PL_SHEET_NAME)
    print(f"Successfully loaded P&L sheet: '{PL_SHEET_NAME}'")

    # Load Journal Entry Detail Data
    je_detail_df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=JE_SHEET_NAME)
    print(f"Successfully loaded JE sheet: '{JE_SHEET_NAME}'")

except FileNotFoundError:
    print(f"Error: Excel file not found at '{EXCEL_FILE_PATH}'")
    sys.exit()
except ValueError as e:
    print(f"Error: Problem loading sheets from '{EXCEL_FILE_PATH}'. Check sheet names ('{PL_SHEET_NAME}', '{JE_SHEET_NAME}'). Details: {e}")
    sys.exit()
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    sys.exit()

# --- 2. Transform P&L Data (Wide to Flat) ---

# Check if required ID columns exist in P&L data
pl_id_vars = [PL_ID_COLUMN, PL_MAP_COLUMN]
if not all(col in pl_wide_df.columns for col in pl_id_vars):
    print(f"Error: Missing required columns {pl_id_vars} in sheet '{PL_SHEET_NAME}'. Cannot proceed.")
    sys.exit()

try:
    pl_flat_df = pd.melt(
        pl_wide_df,
        id_vars=pl_id_vars,      # Columns to keep as identifiers
        var_name='Period',       # New column for the month names (e.g., 'Jan 2024')
        value_name='Amount'      # New column for the financial values
    )
    print("Successfully transformed P&L data from wide to flat format.")
    # Optional: Remove rows where Amount is 0 or NaN if they are not meaningful
    # pl_flat_df = pl_flat_df.dropna(subset=['Amount'])
    # pl_flat_df = pl_flat_df[pl_flat_df['Amount'] != 0]

except Exception as e:
    print(f"An error occurred during P&L data transformation (melt): {e}")
    sys.exit()


# --- 3. Clean Data & Prepare for Linking ---

# Ensure linking columns exist and are consistent type (string)
if PL_ID_COLUMN in pl_flat_df.columns:
    pl_flat_df[PL_ID_COLUMN] = pl_flat_df[PL_ID_COLUMN].astype(str).str.strip()
else:
    # This check was done earlier, but defensive programming is good
    print(f"Critical Error: Linking column '{PL_ID_COLUMN}' unexpectedly missing after melt.")
    sys.exit()

if JE_ID_COLUMN in je_detail_df.columns:
    je_detail_df[JE_ID_COLUMN] = je_detail_df[JE_ID_COLUMN].astype(str).str.strip()
else:
    print(f"Error: Linking column '{JE_ID_COLUMN}' not found in JE sheet '{JE_SHEET_NAME}'. Cannot link data.")
    sys.exit()

# Convert JE Date column to datetime objects
if JE_DATE_COLUMN in je_detail_df.columns:
    try:
        # errors='coerce' will turn unparseable dates into NaT (Not a Time)
        je_detail_df[JE_DATE_COLUMN] = pd.to_datetime(je_detail_df[JE_DATE_COLUMN], errors='coerce')
        # Optional: Handle or report NaT dates if necessary
        if je_detail_df[JE_DATE_COLUMN].isnull().any():
            print(f"Warning: Some dates in '{JE_DATE_COLUMN}' could not be parsed and were set to NaT.")
    except Exception as e:
        print(f"Error converting JE date column '{JE_DATE_COLUMN}' to datetime: {e}")
        # Decide if this is critical - maybe proceed without date filtering? For now, exit.
        sys.exit()
else:
    print(f"Error: Date column '{JE_DATE_COLUMN}' not found in JE sheet '{JE_SHEET_NAME}'. Cannot filter JEs by period.")
    # Decide if this is critical. For now, exit.
    sys.exit()

print("Data cleaning and type conversion complete.")


# --- 4. Define Lookup Function ---

def get_journal_entries(account_id, period_str, je_df):
    """
    Filters the JE DataFrame for a specific account ID and period string.

    Args:
        account_id (str): The Account ID to filter by.
        period_str (str): The period string (e.g., 'Jan 2024') from the P&L data.
        je_df (pd.DataFrame): The cleaned Journal Entry DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered journal entries.
    """
    # Ensure account_id is string for comparison
    account_id = str(account_id).strip()

    # --- Parse the period string to get month and year ---
    try:
        # Attempt to parse formats like 'Jan 2024', 'January 2024', '2024-01', '01/2024' etc.
        # This might need adjustment depending on your *exact* column header format
        period_date = pd.to_datetime(period_str, errors='coerce')
        if pd.isna(period_date):
             # Try alternative parsing if needed, e.g., split space
             try:
                 month_str, year_str = period_str.split()
                 period_date = pd.to_datetime(f"{month_str} 1 {year_str}", errors='coerce') # Add day 1
             except:
                 print(f"Warning: Could not parse period string '{period_str}'. Cannot filter by date.")
                 # Return JEs matching only account ID if date parsing fails? Or empty?
                 # Let's return matching account only with a warning.
                 return je_df[je_df[JE_ID_COLUMN] == account_id].copy()


        target_year = period_date.year
        target_month = period_date.month

        # Filter JE DataFrame
        # 1. Filter by Account ID
        filtered_je = je_df[je_df[JE_ID_COLUMN] == account_id].copy() # Use .copy() to avoid SettingWithCopyWarning

        # 2. Filter by Date (matching Year and Month)
        filtered_je = filtered_je[
            (filtered_je[JE_DATE_COLUMN].dt.year == target_year) &
            (filtered_je[JE_DATE_COLUMN].dt.month == target_month)
        ]
        
        # Ensure only desired columns are returned
        relevant_cols = [col for col in JE_DETAIL_COLUMNS if col in filtered_je.columns]
        return filtered_je[relevant_cols]

    except Exception as e:
        print(f"An error occurred during JE filtering for {account_id} / {period_str}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error


# --- 5. Ready for Interface ---
# At this point, you have:
# 1. `pl_flat_df`: Your P&L data in a clean, long format.
# 2. `je_detail_df`: Your cleaned Journal Entry data.
# 3. `get_journal_entries()`: A function to link them dynamically based on selection.

print("\n--- DataFrames are ready for use in an interface (e.g., Streamlit) ---")
print("\nP&L Flat DataFrame Head:")
print(pl_flat_df.head())
print("\nJE Detail DataFrame Head:")
print(je_detail_df.head())

# --- Example Usage of the Lookup Function ---
# Simulate selecting 'Revenue' (assuming its ID is '40000') for 'Feb 2024'
example_account_id = '4001' # <--- Replace with a real Account ID from your data
example_period = '2021-01-01' # <--- Replace with a real Period from your data (matching P&L column header)

print(f"\n--- Example: Fetching JEs for Account ID: {example_account_id}, Period: {example_period} ---")
example_jes = get_journal_entries(example_account_id, example_period, je_detail_df)

if not example_jes.empty:
    print(example_jes.to_string())
else:
    # Check if the P&L entry exists first
    pl_entry_exists = not pl_flat_df[
        (pl_flat_df[PL_ID_COLUMN] == example_account_id) &
        (pl_flat_df['Period'] == example_period)
    ].empty
    if pl_entry_exists:
         print(f"No Journal Entries found matching Account ID '{example_account_id}' for the period '{example_period}'.")
    else:
         print(f"The combination of Account ID '{example_account_id}' and Period '{example_period}' was not found in the P&L data.")