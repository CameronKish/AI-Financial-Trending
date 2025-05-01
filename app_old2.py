import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- SET PAGE CONFIG FIRST ---
st.set_page_config(layout="wide")

# --- Add Current Time Context ---
# Using current time of the machine running the script
current_time = datetime.now()
st.sidebar.write(f"Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
# Location context can be added manually if needed
st.sidebar.write(f"Location Context: Denver, CO, USA")


# --- Import data and functions from your processing script ---
# (Using the exec method)
try:
    # Execute the script - this will load data, clean it, and define functions/variables
    exec(open("data_processor.py").read(), globals())
    # Check if necessary variables were loaded
    if 'pl_flat_df' not in globals() or 'je_detail_df' not in globals() or 'get_journal_entries' not in globals():
        st.error("Could not load necessary data or functions from data_processor.py. Please ensure it runs correctly and defines 'pl_flat_df', 'je_detail_df', and 'get_journal_entries'.")
        st.stop()
    # Make sure key column names are accessible
    if 'PL_ID_COLUMN' not in globals() or 'PL_MAP_COLUMN' not in globals() or 'JE_ID_COLUMN' not in globals() or 'JE_DATE_COLUMN' not in globals():
         st.error("Column name variables (e.g., PL_ID_COLUMN) not found. Ensure they are defined globally in data_processor.py.")
         st.stop()

except FileNotFoundError:
    st.error("Error: `data_processor.py` not found in the current directory.")
    st.stop()
except Exception as e:
    st.error(f"Error running `data_processor.py`: {e}")
    st.stop()

# --- Initialize Session State (Optional but can be useful) ---
# We'll use it to remember the last duplicate search parameters
if 'dup_col' not in st.session_state:
    st.session_state.dup_col = None
if 'dup_val' not in st.session_state:
    st.session_state.dup_val = None


# --- Streamlit App Layout ---
st.title("P&L Analyzer with Journal Entry Drilldown")

# --- Prepare P&L Wide View ---
st.header("Profit & Loss Overview (Monthly)")

try:
    # Convert 'Period' to datetime for proper sorting
    pl_flat_df['Period_dt'] = pd.to_datetime(pl_flat_df['Period'], errors='coerce')
    if pl_flat_df['Period_dt'].isnull().any():
        st.warning("Could not parse all period headers into dates. Sorting might be incorrect. Using original Period string.")
        period_col_for_pivot = 'Period'
    else:
        pl_flat_df['Period_Str'] = pl_flat_df['Period_dt'].dt.strftime('%Y-%m')
        period_col_for_pivot = 'Period_Str'

    # Pivot the table
    pnl_wide_view_df = pl_flat_df.pivot_table(
        index=[PL_ID_COLUMN, PL_MAP_COLUMN], # Use both ID and Mapping for index
        columns=period_col_for_pivot,       # Use the sortable period string
        values='Amount'
    ).fillna(0)
    pnl_wide_view_df = pnl_wide_view_df.sort_index(axis=1) # Sort columns chronologically

except Exception as e:
    st.error(f"Error creating pivoted P&L view: {e}")
    st.stop()


# --- Outlier Highlighting Function (Re-added) ---
def highlight_outliers(row, threshold_std_dev=2.0):
    """
    Calculates MoM difference and highlights values where the change
    from the previous month exceeds N standard deviations for that row.
    """
    diffs = row.diff()
    mean_diff = diffs.dropna().mean()
    std_diff = diffs.dropna().std()
    outlier_threshold_value = threshold_std_dev * std_diff if std_diff is not None and std_diff > 1e-6 else np.inf
    styles = [''] * len(row) # Initialize styles as empty strings
    for i, diff in enumerate(diffs):
        if pd.notna(diff) and abs(diff) > outlier_threshold_value:
             # Highlight the cell *after* the large difference occurred
            if i < len(styles): # Check index bounds
                styles[i] = 'background-color: yellow; color: black;'
    return styles

# --- Display Styled P&L Table ---
st.dataframe(
    pnl_wide_view_df.style.apply(highlight_outliers, axis=1).format("{:,.2f}"),
    use_container_width=True
)
st.caption("Cells highlighted yellow indicate a month-over-month change greater than 2 standard deviations for that specific account.")


# --- User Selection for JE Drilldown (Using Widgets) ---
st.sidebar.header("Select Data for JE Lookup")

# Create list of account mappings for the selectbox
account_options = pnl_wide_view_df.index.get_level_values(PL_MAP_COLUMN).unique().tolist()
# Create a mapping from display name back to Account ID
account_name_to_id_map = {name: id_ for id_, name in pnl_wide_view_df.index.unique()}

selected_account_name = st.sidebar.selectbox(
    "Select GL Account:",
    options=sorted(account_options), # Sort alphabetically
    index=0
)

# Get the corresponding Account ID
selected_account_id = account_name_to_id_map.get(selected_account_name)

# Get period options from the columns of the wide table
period_options = pnl_wide_view_df.columns.tolist()
selected_period = st.sidebar.selectbox(
    "Select Period:",
    options=period_options,
    index=len(period_options) - 1 # Default to the last period
)

# --- Fetch and Display Journal Entries ---
st.header("Journal Entry Details")

related_jes = pd.DataFrame() # Initialize empty DataFrame

if selected_account_id and selected_period:
    st.write(f"Showing Journal Entries for: **{selected_account_name} ({selected_account_id})** | Period: **{selected_period}**")
    try:
        related_jes = get_journal_entries(selected_account_id, selected_period, je_detail_df)

        if not related_jes.empty:
            # Define potential amount/date columns for formatting
            je_amount_cols = [col for col in related_jes.columns if 'Amount' in col or 'Debit' in col or 'Credit' in col]
            date_cols = [col for col in related_jes.columns if 'Date' in col] # Find date columns

            # Prepare formatters dictionary
            formatters = {}
            for col in je_amount_cols:
                formatters[col] = "{:,.2f}"
            for col in date_cols:
                 # Assuming dates are datetime objects from data_processor
                 # Format as YYYY-MM-DD. Adjust format string as needed.
                 formatters[col] = lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notna(x) else ''


            st.dataframe(related_jes.style.format(formatters), use_container_width=True)
        else:
            st.info(f"No Journal Entries found for '{selected_account_name}' in the period '{selected_period}'.")

    except Exception as e:
        st.error(f"An error occurred while fetching Journal Entries: {e}")
else:
    st.info("Select a GL Account and Period from the sidebar to view related Journal Entries.")


# --- Duplicate JE Finder (Using Widgets) ---
st.header("Duplicate Value Lookup in All Journal Entries")

# Only show this section if there are JEs currently displayed
if not related_jes.empty:
    # Define columns relevant for duplicate checking (customize this list)
    potential_dup_cols = ['Customer', 'Memo', 'Transaction Id', 'Amount (Presentation Currency)'] # Add/remove as needed
    # Filter list to only include columns actually present in the JE details
    available_dup_cols = [col for col in potential_dup_cols if col in je_detail_df.columns]

    if not available_dup_cols:
        st.warning("No suitable columns found in JE data for duplicate checking.")
    else:
        col1, col2 = st.columns(2) # Create columns for layout

        with col1:
            # Dropdown to select the COLUMN to check for duplicates
            selected_dup_col = st.selectbox(
                "Select Column to Check:",
                options=available_dup_cols,
                key='dup_col_select' # Key helps maintain state
            )

        with col2:
            # Dropdown to select the VALUE from the *currently displayed* JEs
            if selected_dup_col:
                # Get unique, non-null values from the selected column in the *displayed* subset
                value_options = sorted(related_jes[selected_dup_col].dropna().unique())
                selected_dup_val = st.selectbox(
                    f"Select Value from '{selected_dup_col}':",
                    options=value_options,
                    key='dup_val_select' # Key helps maintain state
                )
            else:
                st.selectbox(f"Select Value:", options=[], key='dup_val_select') # Empty if no column selected

        find_duplicates_button = st.button("Find All Duplicates for Selected Value")

        if find_duplicates_button and selected_dup_col and selected_dup_val is not None:
            st.session_state.dup_col = selected_dup_col # Store for display below
            st.session_state.dup_val = selected_dup_val # Store for display below

            st.write(f"Finding all JEs where **{st.session_state.dup_col}** is **'{st.session_state.dup_val}'**...")
            try:
                # Filter the ORIGINAL, FULL je_detail_df
                col_to_check = st.session_state.dup_col
                val_to_find = st.session_state.dup_val

                # Handle type matching carefully for filtering
                if pd.api.types.is_numeric_dtype(je_detail_df[col_to_check].dtype) and pd.api.types.is_number(val_to_find):
                    duplicate_jes_df = je_detail_df[np.isclose(je_detail_df[col_to_check].fillna(np.nan), val_to_find)] # Use isclose for float comparison
                elif pd.api.types.is_datetime64_any_dtype(je_detail_df[col_to_check].dtype) and isinstance(val_to_find, pd.Timestamp):
                     duplicate_jes_df = je_detail_df[je_detail_df[col_to_check] == val_to_find] # Direct compare for datetime
                else:
                    # Default: Compare as strings after stripping whitespace
                    duplicate_jes_df = je_detail_df[je_detail_df[col_to_check].astype(str).str.strip() == str(val_to_find).strip()]


                if not duplicate_jes_df.empty:
                    st.write(f"Found {len(duplicate_jes_df)} entries:")
                    # Apply formatting similar to the main JE table
                    st.dataframe(duplicate_jes_df.style.format(formatters), use_container_width=True)
                else:
                    st.info(f"No other Journal Entries found where '{col_to_check}' is '{val_to_find}'.")

            except KeyError:
                st.error(f"Column '{col_to_check}' not found in the main JE dataset for duplicate checking.")
            except Exception as e:
                st.error(f"An error occurred during duplicate lookup: {e}")
        # Optional: Display last search criteria if needed outside the button click scope
        # elif st.session_state.dup_col:
        #     st.write(f"Last duplicate search was for '{st.session_state.dup_val}' in column '{st.session_state.dup_col}'.")

else:
     st.info("Select an Account and Period with Journal Entries to enable the duplicate lookup.")