# data_processor.py
import pandas as pd
from datetime import datetime
import logging
# import streamlit as st # Removed: No longer needed after removing debug st.write

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants for the original load_and_process_data (now obsolete for main app flow)
# EXCEL_FILE_PATH = 'Fake Data Summary.xlsx'
# PL_SHEET_NAME = 'PL-Wide'
# JE_SHEET_NAMES = ['TXN-FY21', 'TXN-FY22', 'TXN-FY23']
# PL_ID_COLUMN = 'Account ID'
# PL_MAP1_COLUMN = 'Mapping 1'
# PL_MAP2_COLUMN = 'Mapping 2'
# PL_MAP_COLUMN = PL_MAP2_COLUMN
# JE_ID_COLUMN_ORIGINAL = 'Account Number/Code'
# JE_DATE_COLUMN_ORIGINAL = 'Transaction Date'
# JE_AMOUNT_COLUMN_ORIGINAL = 'Amount (Presentation Currency)'
# JE_DETAIL_COLUMNS_BASE_ORIGINAL = ["Transaction Id", JE_ID_COLUMN_ORIGINAL, JE_DATE_COLUMN_ORIGINAL, 'Account Name', JE_AMOUNT_COLUMN_ORIGINAL, 'Memo', 'Customer']


# --- MODIFIED get_journal_entries ---
def get_journal_entries(account_id: str,
                        period_str: str, # Expected format 'YYYY-MM'
                        je_df: pd.DataFrame,
                        account_id_col_name: str,
                        date_col_name: str,
                        detail_columns_to_return: list):
    """
    Filters the JE DataFrame for a specific account ID and period string (YYYY-MM).
    Uses specified column names for account ID and date.
    Returns a DataFrame with columns specified in detail_columns_to_return, if they exist.
    """
    if detail_columns_to_return is None:
        detail_columns_to_return = []

    if je_df is None or je_df.empty:
        logging.warning("get_journal_entries called with empty or None je_df.")
        return pd.DataFrame(columns=[col for col in detail_columns_to_return if isinstance(col, str)])

    if not all(isinstance(col, str) and col in je_df.columns for col in [account_id_col_name, date_col_name]):
        missing_required = [col for col in [account_id_col_name, date_col_name] if col not in je_df.columns or not isinstance(col, str)]
        logging.error(f"JE DataFrame missing required columns for filtering or column names are not strings: {missing_required}. Available: {je_df.columns.tolist()}")
        return pd.DataFrame(columns=[col for col in detail_columns_to_return if isinstance(col, str) and col in je_df.columns])

    account_id_str = str(account_id).strip()
    try:
        period_date_obj = pd.to_datetime(period_str + '-01', format='%Y-%m-%d', errors='coerce')
        if pd.isna(period_date_obj):
            logging.warning(f"Could not parse period string '{period_str}' in get_journal_entries. Expected 'YYYY-MM'.")
            return pd.DataFrame(columns=[col for col in detail_columns_to_return if isinstance(col, str) and col in je_df.columns])
        target_year = period_date_obj.year
        target_month = period_date_obj.month
    except Exception as e:
        logging.error(f"Error parsing period_str '{period_str}': {e}")
        return pd.DataFrame(columns=[col for col in detail_columns_to_return if isinstance(col, str) and col in je_df.columns])

    temp_je_df = je_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(temp_je_df[date_col_name]):
         try:
             temp_je_df[date_col_name] = pd.to_datetime(temp_je_df[date_col_name], errors='coerce')
             logging.info(f"Converted column '{date_col_name}' to datetime within get_journal_entries.")
         except Exception as e:
             logging.error(f"Failed to convert '{date_col_name}' to datetime in get_journal_entries: {e}")
             return pd.DataFrame(columns=[col for col in detail_columns_to_return if isinstance(col, str) and col in je_df.columns])

    temp_je_df[account_id_col_name] = temp_je_df[account_id_col_name].astype(str).str.strip()

    try:
        mask_account = (temp_je_df[account_id_col_name] == account_id_str)
        mask_valid_date = temp_je_df[date_col_name].notna()
        mask_year = pd.Series(False, index=temp_je_df.index)
        mask_month = pd.Series(False, index=temp_je_df.index)

        if mask_valid_date.any():
            mask_year[mask_valid_date] = (temp_je_df.loc[mask_valid_date, date_col_name].dt.year == target_year)
            mask_month[mask_valid_date] = (temp_je_df.loc[mask_valid_date, date_col_name].dt.month == target_month)

        final_mask = mask_account & mask_valid_date & mask_year & mask_month
        filtered_je = temp_je_df.loc[final_mask]

        final_columns_to_return = [col for col in detail_columns_to_return if isinstance(col, str) and col in filtered_je.columns]
        return filtered_je[final_columns_to_return]

    except Exception as e:
        logging.error(f"Error during JE filtering for {account_id_str}/{period_str} on columns '{account_id_col_name}', '{date_col_name}': {e}")
        return pd.DataFrame(columns=[col for col in detail_columns_to_return if isinstance(col, str) and col in je_df.columns])