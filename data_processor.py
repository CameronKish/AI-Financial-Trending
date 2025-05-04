# data_processor.py
import pandas as pd
from datetime import datetime
import logging # Use logging instead of print for library-like behavior

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---
EXCEL_FILE_PATH = 'Fake Data Summary.xlsx'
PL_SHEET_NAME = 'PL-Wide'
JE_SHEET_NAMES = ['TXN-FY21', 'TXN-FY22', 'TXN-FY23'] # Keep this updated

# --- Column Name Definitions ---
PL_ID_COLUMN = 'Account ID'
PL_MAP1_COLUMN = 'Mapping 1'
PL_MAP2_COLUMN = 'Mapping 2'
PL_MAP_COLUMN = PL_MAP2_COLUMN # Lowest level mapping for display name
JE_ID_COLUMN = 'Account Number/Code'
JE_DATE_COLUMN = 'Transaction Date'
JE_AMOUNT_COLUMN = 'Amount (Presentation Currency)'
# Ensure all columns expected by get_journal_entries are listed if needed elsewhere,
# or rely on the function's internal definition.
JE_DETAIL_COLUMNS_BASE = ["Transaction Id", JE_ID_COLUMN, JE_DATE_COLUMN, 'Account Name', JE_AMOUNT_COLUMN, 'Memo', 'Customer']


def load_and_process_data(excel_path=EXCEL_FILE_PATH, pl_sheet=PL_SHEET_NAME, je_sheets=None):
    """
    Loads P&L and JE data from the specified Excel file and performs
    initial processing and cleaning.

    Args:
        excel_path (str): Path to the Excel file.
        pl_sheet (str): Name of the P&L sheet.
        je_sheets (list): List of JE sheet names.

    Returns:
        tuple: Contains (pl_flat_df, je_detail_df) or (None, None) on critical error.
               Returns the dataframes and essential column names.
    """
    if je_sheets is None:
        je_sheets = JE_SHEET_NAMES

    # --- 1a. Load P&L Data ---
    try:
        pl_wide_df = pd.read_excel(excel_path, sheet_name=pl_sheet)
        logging.info(f"Successfully loaded P&L sheet: '{pl_sheet}'")
    except FileNotFoundError:
        logging.error(f"CRITICAL Error: Excel file not found at '{excel_path}'")
        raise # Re-raise the exception to be caught by the caller
    except ValueError as e:
        logging.error(f"CRITICAL Error: P&L sheet name '{pl_sheet}' not found or invalid. Details: {e}")
        raise
    except Exception as e:
        logging.error(f"CRITICAL An unexpected error occurred loading the P&L sheet: {e}")
        raise

    # --- 1b. Load JE Data ---
    all_je_dfs = []
    successful_loads = []
    failed_loads = []
    logging.info(f"Attempting to load JE sheets: {je_sheets}")

    for sheet in je_sheets:
        try:
            temp_je_df = pd.read_excel(excel_path, sheet_name=sheet)
            # Basic check for essential columns before appending
            if JE_ID_COLUMN not in temp_je_df.columns or JE_DATE_COLUMN not in temp_je_df.columns or JE_AMOUNT_COLUMN not in temp_je_df.columns:
                 logging.warning(f"  Skipping JE sheet '{sheet}' due to missing essential columns ({JE_ID_COLUMN}, {JE_DATE_COLUMN}, {JE_AMOUNT_COLUMN}).")
                 failed_loads.append(f"{sheet} (Missing Cols)")
                 continue
            all_je_dfs.append(temp_je_df)
            successful_loads.append(sheet)
            logging.info(f"  Successfully loaded JE sheet: '{sheet}' ({len(temp_je_df)} rows)")
        except ValueError:
            logging.warning(f"  Warning: JE sheet '{sheet}' not found in '{excel_path}'. Skipping.")
            failed_loads.append(f"{sheet} (Not Found)")
        except Exception as e:
            logging.warning(f"  Warning: Error loading JE sheet '{sheet}': {e}. Skipping.")
            failed_loads.append(f"{sheet} (Load Error)")

    if not all_je_dfs:
        logging.error(f"CRITICAL Error: No valid JE data sheets could be loaded from the list: {je_sheets}.")
        raise ValueError("No valid JE data could be loaded.") # Raise error

    je_detail_df = pd.concat(all_je_dfs, ignore_index=True)
    logging.info(f"Successfully combined {len(successful_loads)} JE sheets: {successful_loads}")
    logging.info(f"Total JE rows combined: {len(je_detail_df)}")
    if failed_loads:
        logging.warning(f"Skipped JE sheets: {failed_loads}")

    # --- 2. Transform P&L Data (Wide to Flat) ---
    pl_id_vars = [PL_ID_COLUMN, PL_MAP1_COLUMN, PL_MAP2_COLUMN] # Add other mapping cols if they exist
    # Check if all essential P&L columns exist
    missing_pl_cols = [col for col in pl_id_vars if col not in pl_wide_df.columns]
    if missing_pl_cols:
        logging.error(f"CRITICAL Error: Missing required P&L columns {missing_pl_cols} in sheet '{pl_sheet}'.")
        raise ValueError(f"Missing required P&L columns: {missing_pl_cols}")

    try:
        # Identify value columns (periods) - assumes they are the ones not in id_vars
        value_vars = [col for col in pl_wide_df.columns if col not in pl_id_vars]
        if not value_vars:
             raise ValueError("No period columns found for melting P&L data.")

        pl_flat_df = pd.melt(pl_wide_df, id_vars=pl_id_vars, value_vars=value_vars, var_name='Period', value_name='Amount')
        logging.info("Successfully transformed P&L data from wide to flat format.")
    except Exception as e:
        logging.error(f"CRITICAL Error during P&L data transformation (melt): {e}")
        raise

    # --- 3. Clean Data & Prepare for Linking ---
    logging.info("Starting Data Cleaning...")
    # P&L Linking Column
    pl_flat_df[PL_ID_COLUMN] = pl_flat_df[PL_ID_COLUMN].astype(str).str.strip()

    # JE Linking Column
    je_detail_df[JE_ID_COLUMN] = je_detail_df[JE_ID_COLUMN].astype(str).str.strip()

    # JE Date Column Conversion
    try:
        original_dtype = je_detail_df[JE_DATE_COLUMN].dtype
        logging.info(f"  Attempting to convert JE Date column ('{JE_DATE_COLUMN}') from type {original_dtype}...")
        je_detail_df[JE_DATE_COLUMN] = pd.to_datetime(je_detail_df[JE_DATE_COLUMN], errors='coerce')
        na_dates_count = je_detail_df[JE_DATE_COLUMN].isnull().sum()
        if na_dates_count > 0:
            logging.warning(f"  Warning: {na_dates_count} dates in '{JE_DATE_COLUMN}' could not be parsed and were set to NaT.")
        logging.info(f"  Successfully converted JE Date column to datetime.")
    except Exception as e:
        logging.error(f"CRITICAL Error converting JE date column '{JE_DATE_COLUMN}' to datetime: {e}")
        raise

    logging.info("Data loading and initial processing complete.")
    return pl_flat_df, je_detail_df

# --- 4. Define Lookup Function (Remains largely the same) ---
def get_journal_entries(account_id, period_str, je_df):
    """ Filters the JE DataFrame for a specific account ID and period string (YYYY-MM). """
    # Ensure required columns exist in the input DataFrame
    required_cols = [JE_ID_COLUMN, JE_DATE_COLUMN]
    if not all(col in je_df.columns for col in required_cols):
        logging.error(f"JE DataFrame missing required columns for filtering: {required_cols}")
        return pd.DataFrame(columns=JE_DETAIL_COLUMNS_BASE) # Return empty with expected base columns

    account_id = str(account_id).strip()
    period_date = pd.to_datetime(period_str + '-01', errors='coerce', format='%Y-%m-%d') # Assume YYYY-MM input

    if pd.isna(period_date):
        logging.warning(f"Could not parse period string '{period_str}' in get_journal_entries. Expected 'YYYY-MM'.")
        return pd.DataFrame(columns=JE_DETAIL_COLUMNS_BASE)

    target_year = period_date.year
    target_month = period_date.month

    # Ensure date column is datetime type before filtering (defensive check)
    if not pd.api.types.is_datetime64_any_dtype(je_df[JE_DATE_COLUMN]):
         try:
             je_df[JE_DATE_COLUMN] = pd.to_datetime(je_df[JE_DATE_COLUMN], errors='coerce')
             logging.warning("Re-converted JE Date column to datetime within get_journal_entries.")
         except Exception as e:
             logging.error(f"Failed to convert JE Date column within get_journal_entries: {e}")
             return pd.DataFrame(columns=JE_DETAIL_COLUMNS_BASE)


    # Filter JE DataFrame
    try:
        mask = (
            (je_df[JE_ID_COLUMN] == account_id) &
            (je_df[JE_DATE_COLUMN].dt.year == target_year) &
            (je_df[JE_DATE_COLUMN].dt.month == target_month) &
            (je_df[JE_DATE_COLUMN].notna()) # Exclude rows where date conversion failed
        )
        filtered_je = je_df.loc[mask].copy()

        # Ensure only desired columns are returned, handling missing ones
        actual_detail_cols = [col for col in JE_DETAIL_COLUMNS_BASE if col in filtered_je.columns]
        return filtered_je[actual_detail_cols]

    except Exception as e:
        logging.error(f"Error during JE filtering for {account_id}/{period_str}: {e}")
        return pd.DataFrame(columns=JE_DETAIL_COLUMNS_BASE) # Return empty DataFrame on error

# --- Column Name Constants (Exported for use in other modules) ---
# (Defined at the top)

# No direct execution block (__name__ == "__main__") needed if run via Streamlit