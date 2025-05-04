# pages/1_📊_P&L_Analysis_&_Drilldown.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Import shared utilities and data processing functions
import utils
import data_processor # We need get_journal_entries

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="P&L Analysis") # Config can be set per page if needed, or inherit from main
st.markdown(f"<style> h1 {{ color: {utils.EY_DARK_BLUE_GREY}; }} </style>", unsafe_allow_html=True) # Example of applying color

# --- Check if data is loaded ---
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.error("Data not loaded. Please go back to the main page and ensure data is loaded correctly.")
    st.stop()

# --- Retrieve data from Session State ---
# Make sure keys match those set in streamlit_app.py
try:
    pl_flat_df = st.session_state.pl_flat_df
    je_detail_df = st.session_state.je_detail_df
    # Retrieve column names from session state
    col_config = st.session_state.column_config
    PL_ID_COLUMN = col_config["PL_ID"]
    PL_MAP_COLUMN = col_config["PL_MAP_DISPLAY"] # Use the display mapping column
    JE_ID_COLUMN = col_config["JE_ID"]
    JE_DATE_COLUMN = col_config["JE_DATE"]
    JE_AMOUNT_COLUMN = col_config["JE_AMOUNT"]
    JE_ACCOUNT_NAME_COL = col_config["JE_ACCOUNT_NAME"]
except KeyError as e:
     st.error(f"Missing required data or configuration in session state: {e}. Please reload the app.")
     st.stop()


# --- Sidebar Controls Specific to this Page ---
st.sidebar.header("P&L Analysis Controls")
threshold_std_dev = st.sidebar.slider(
    "Outlier Sensitivity (Std Deviations)",
    min_value=1.0,
    max_value=4.0,
    value=st.session_state.get('outlier_threshold', 2.0), # Use session state to remember value
    step=0.1,
    key='outlier_threshold', # Assign key to store value in session state
    help="Lower = More sensitive highlighting for Month-over-Month changes."
)

st.sidebar.markdown("---")
st.sidebar.header("Select Data for JE Lookup")
st.sidebar.caption("Use these OR click row/column in the table.")

# --- Prepare P&L Data for Display (Pivot, Diff, StdDev) ---
# This calculation now happens within the page
try:
    # Convert Period to datetime for sorting/filtering if not already done
    if 'Period_dt' not in pl_flat_df.columns or pl_flat_df['Period_dt'].isnull().any():
         pl_flat_df['Period_dt'] = pd.to_datetime(pl_flat_df['Period'], errors='coerce')
         # Create a string version for display if needed, handle potential NaT
         pl_flat_df['Period_Str'] = pl_flat_df.apply(lambda row: row['Period_dt'].strftime('%Y-%m') if pd.notna(row['Period_dt']) else str(row['Period']), axis=1)
         period_col_for_pivot = 'Period_Str' # Use string for stable column names
    else:
         # If already datetime and clean, create string version for pivot
         pl_flat_df['Period_Str'] = pl_flat_df['Period_dt'].dt.strftime('%Y-%m')
         period_col_for_pivot = 'Period_Str'

    # Pivot Table for P&L View
    pnl_wide_view_df = pl_flat_df.pivot_table(
        index=[PL_ID_COLUMN, PL_MAP_COLUMN], # Use defined columns
        columns=period_col_for_pivot,
        values='Amount'
    ).fillna(0)

    # Sort columns chronologically
    pnl_wide_view_df = pnl_wide_view_df.sort_index(axis=1)

    # Prepare for highlighting: Calculate MoM differences and Std Dev per account
    diff_df = pnl_wide_view_df.diff(axis=1)
    row_std_diff = pnl_wide_view_df.diff(axis=1).std(axis=1, skipna=True).fillna(0) # Std Dev of the differences

    # For table selection processing (need reset index version)
    pnl_wide_view_df_reset = pnl_wide_view_df.reset_index()

    # Get options for dropdowns
    account_options = sorted(pnl_wide_view_df.index.get_level_values(PL_MAP_COLUMN).unique().tolist())
    # Create maps for ID <-> Name lookup
    # Use unique index tuples, handle potential duplicates if structure allows
    unique_indices = pnl_wide_view_df.index.unique()
    account_name_to_id_map = {name: id_ for id_, name in unique_indices}
    account_id_to_name_map = {id_: name for id_, name in unique_indices}

    period_options = pnl_wide_view_df.columns.tolist() # Already sorted

except Exception as e:
    st.error(f"Error preparing P&L data for display: {e}")
    st.exception(e) # Show traceback in Streamlit for debugging
    st.stop()


# --- Sidebar Selection Widgets & Synchronization ---
# Determine default indices based on session state
current_account_name = st.session_state.get('selected_account_name') # Use .get for safety
current_period = st.session_state.get('selected_period')

# Use helper function for robust index finding
account_index = utils.get_index(account_options, current_account_name) if current_account_name else 0
period_index = utils.get_index(period_options, current_period) if current_period else len(period_options) - 1

# Display select boxes
sb_selected_account_name = st.sidebar.selectbox(
    "Select GL Account:",
    options=account_options,
    index=account_index,
    key="sb_account" # Key is important for Streamlit to track the widget state
)

sb_selected_period = st.sidebar.selectbox(
    "Select Period:",
    options=period_options, # Already sorted
    index=period_index,
    key="sb_period"
)

# Synchronize Sidebar selection TO Session State
sidebar_account_id = account_name_to_id_map.get(sb_selected_account_name) # Map selected name to ID
sidebar_period = sb_selected_period

# Check if sidebar interaction changed the selection compared to session state
# This logic updates the central session state if the sidebar widgets are used
if (sidebar_account_id != st.session_state.get('selected_account_id') or
    sidebar_period != st.session_state.get('selected_period')):
    # Update session state
    st.session_state.selected_account_id = sidebar_account_id
    st.session_state.selected_account_name = sb_selected_account_name
    st.session_state.selected_period = sidebar_period
    # Reset dependent states like JE details or duplicate finder
    st.session_state.related_jes_df = pd.DataFrame(columns=st.session_state.column_config["JE_DETAILS_BASE"]) # Reset JEs
    st.session_state.dup_col = None # Reset duplicate finder state
    st.session_state.dup_val = None
    st.session_state.dup_search_triggered = False
    # No rerun needed here, flow will continue and fetch JEs if necessary based on updated state


# --- Fetch Journal Entries based on Session State ---
# Fetch if the selection has changed since the last fetch OR if it's newly selected
should_fetch = False
if st.session_state.selected_account_id and st.session_state.selected_period:
    if (st.session_state.selected_account_id != st.session_state.get('prev_selected_account_id') or
        st.session_state.selected_period != st.session_state.get('prev_selected_period')):
         should_fetch = True

# If fetch is needed, call the function and update session state
# This check ensures we don't refetch unnecessarily on every interaction
if should_fetch:
    try:
        # Use the imported get_journal_entries function
        fetched_jes = data_processor.get_journal_entries(
            st.session_state.selected_account_id,
            st.session_state.selected_period,
            je_detail_df # Pass the main JE dataframe
        )
        st.session_state.related_jes_df = fetched_jes
        # Update previous selection state to prevent re-fetching until changed again
        st.session_state.prev_selected_account_id = st.session_state.selected_account_id
        st.session_state.prev_selected_period = st.session_state.selected_period
        # Reset duplicate finder state when new JEs are loaded
        st.session_state.dup_col = None
        st.session_state.dup_val = None
        st.session_state.dup_search_triggered = False

    except Exception as e_fetch:
        st.error(f"An error occurred fetching Journal Entries: {e_fetch}")
        st.session_state.related_jes_df = pd.DataFrame(columns=st.session_state.column_config["JE_DETAILS_BASE"]) # Reset to empty on error

# --- Page Content: P&L Table, JE Details, Duplicate Finder ---

# 1. P&L Overview Table
st.markdown(f"<h1>P&L Overview (Monthly)</h1>", unsafe_allow_html=True)
st.caption(f"Highlighting ({utils.EY_YELLOW} color) indicates Month-over-Month change > {st.session_state.outlier_threshold:.1f} Std Deviations. Click row index + column header to select data for JE Lookup.")

try:
    # Calculate threshold values based on slider
    outlier_threshold_values = st.session_state.outlier_threshold * row_std_diff
    # Apply styling: Use the helper function from utils
    styled_df = pnl_wide_view_df.style.apply(
        utils.highlight_outliers_pandas, # Use the function from utils
        axis=1, # Apply row-wise
        diffs_df=diff_df,
        thresholds_series=outlier_threshold_values,
        color=utils.EY_YELLOW,
        text_color=utils.EY_TEXT_ON_YELLOW
    ).format("{:,.0f}") # Apply number formatting

    # Display the styled DataFrame using st.dataframe
    # Key is important for selection state
    # on_select="rerun" ensures Streamlit reruns the script when selection changes
    # selection_mode allows selecting a row and a column (though not simultaneously with one click)
    pnl_selection = st.dataframe(
        styled_df,
        use_container_width=True,
        key="pnl_select_df", # Assign a key to access selection state
        on_select="rerun",
        selection_mode=("single-row", "single-column") # Allow row/col selection
    )

except Exception as e_style:
    st.error(f"Error applying styles or displaying P&L table: {e_style}")
    st.exception(e_style)
    # Fallback to display unstyled data
    st.dataframe(pnl_wide_view_df.style.format("{:,.0f}"), use_container_width=True)


# --- Process Table Selection ---
# Check the selection state of the pnl_select_df dataframe
# This runs after the dataframe is rendered and potentially after a rerun trigger
table_selection_state = pnl_selection.selection # Access selection state directly from the returned object

selected_rows_indices = table_selection_state.get('rows', [])
selected_cols_names = table_selection_state.get('columns', [])

# If both a row and a column are selected via multi-click
if selected_rows_indices and selected_cols_names:
    try:
        selected_row_pos_index = selected_rows_indices[0] # Get the positional index
        # Use the reset_index version to easily get ID and Name based on position
        selected_row_data = pnl_wide_view_df_reset.iloc[selected_row_pos_index]

        table_account_id = str(selected_row_data[PL_ID_COLUMN])
        table_account_name = selected_row_data[PL_MAP_COLUMN]
        table_period = selected_cols_names[0] # Get the selected period (column name)

        # If table selection differs from session state, update session state
        if (table_account_id != st.session_state.get('selected_account_id') or
            table_period != st.session_state.get('selected_period')):

            st.session_state.selected_account_id = table_account_id
            st.session_state.selected_account_name = table_account_name
            st.session_state.selected_period = table_period
            # Reset dependent states
            st.session_state.related_jes_df = pd.DataFrame(columns=st.session_state.column_config["JE_DETAILS_BASE"]) # Reset JEs
            st.session_state.dup_col = None
            st.session_state.dup_val = None
            st.session_state.dup_search_triggered = False

            # Trigger a rerun to update sidebar widgets and fetch JEs based on new selection
            # This is crucial for the table selection to sync with the rest of the app
            st.rerun()

    except IndexError:
         st.warning("Could not read selection from table. Please ensure you click the row index first, then the column header.")
    except Exception as e_proc:
        st.warning(f"Could not process Table selection: {e_proc}")


# 2. Journal Entry Details
st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>Journal Entry Details</h2>", unsafe_allow_html=True)

# Display JEs based on the current selection in session state
related_jes_to_display = st.session_state.get('related_jes_df', pd.DataFrame()) # Get from state

if st.session_state.get('selected_account_id') and st.session_state.get('selected_period'):
    st.write(f"Showing JEs for: **{st.session_state.selected_account_name} ({st.session_state.selected_account_id})** | Period: **{st.session_state.selected_period}**")

    if isinstance(related_jes_to_display, pd.DataFrame) and not related_jes_to_display.empty:
        # Apply formatting using the robust helper function for amounts
        je_display_df = related_jes_to_display.copy()
        # Identify amount columns dynamically (more robust)
        je_amount_cols = [col for col in je_display_df.columns if 'Amount' in col or 'Debit' in col or 'Credit' in col or col == JE_AMOUNT_COLUMN]
        for col in je_amount_cols:
            # Use the safe formatter from utils
             je_display_df[col] = je_display_df[col].apply(utils.format_amount_safely)

        # Configure date columns using st.column_config
        je_col_config = {}
        date_cols_to_format = [col for col in je_display_df.columns if 'Date' in col or col == JE_DATE_COLUMN]
        for col in date_cols_to_format:
             je_col_config[col] = st.column_config.DateColumn(
                 label=col, # Use original column name as label
                 format="YYYY-MM-DD",
                 help="Transaction Date" if col == JE_DATE_COLUMN else "Date"
             )

        # Display the formatted JE details
        st.dataframe(
            je_display_df,
            use_container_width=True,
            column_config=je_col_config,
            hide_index=True # Often cleaner for JE lists
        )
    elif isinstance(related_jes_to_display, pd.DataFrame) and related_jes_to_display.empty:
        st.info(f"No Journal Entries found for this account and period.")
    else:
        # This case should ideally not be reached if state is managed correctly
        st.warning("Journal Entry data is currently unavailable or in an unexpected format.")
else:
    st.info("Select an Account and Period (using sidebar or table) to view Journal Entries.")


# 3. Duplicate JE Finder
st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>Duplicate Value Lookup (Across All JEs)</h2>", unsafe_allow_html=True)

# Enable only if there are JEs currently displayed to select values from
if isinstance(related_jes_to_display, pd.DataFrame) and not related_jes_to_display.empty:
    # Define potential columns for checking duplicates (ensure they exist in the main JE detail df)
    potential_dup_cols = ['Customer', 'Memo', 'Transaction Id', JE_AMOUNT_COLUMN] # Use constant for amount
    # Filter based on columns actually present in the full JE dataset
    available_dup_cols = [col for col in potential_dup_cols if col in je_detail_df.columns]

    if available_dup_cols:
        col1, col2 = st.columns(2)
        with col1:
            # Select column to check, remember choice using session state
            last_dup_col = st.session_state.get('dup_col')
            dup_col_index = utils.get_index(available_dup_cols, last_dup_col)
            selected_dup_col = st.selectbox(
                "Select Column to Check:",
                options=available_dup_cols,
                index=dup_col_index,
                key='dup_col_select' # Use key to track widget state
            )

        with col2:
            # Populate value options based on the *currently displayed* JEs for the selected column
            value_options = []
            if selected_dup_col and selected_dup_col in related_jes_to_display.columns:
                # Get unique, non-null values from the displayed JEs
                value_options = sorted(related_jes_to_display[selected_dup_col].dropna().unique())

            # Select value from the options, remember choice using session state
            last_dup_val = st.session_state.get('dup_val')
            # Need to handle potential type mismatches if value options change
            try:
                dup_val_index = utils.get_index(value_options, last_dup_val) if last_dup_val in value_options else 0
            except: # Catch potential comparison errors between last_dup_val and options
                dup_val_index = 0

            selected_dup_val = st.selectbox(
                f"Select Value from Current JEs:",
                options=value_options,
                index=dup_val_index,
                key='dup_val_select', # Use key
                disabled=not value_options # Disable if no options
            )

        # Button to trigger the search across the *entire* JE dataset
        find_duplicates_button = st.button("Find All JEs with Selected Value")

        if find_duplicates_button:
            # Store the selections in session state and set trigger
            st.session_state.dup_col = selected_dup_col
            st.session_state.dup_val = selected_dup_val
            st.session_state.dup_search_triggered = True
            st.rerun() # Rerun to perform search in the next script run

        # Perform search if triggered
        if st.session_state.get('dup_search_triggered'):
            col_to_check = st.session_state.dup_col
            val_to_find = st.session_state.dup_val
            st.write(f"Finding all JEs in the full dataset where **{col_to_check}** is **'{val_to_find}'**...")

            # Ensure search criteria are valid before proceeding
            if col_to_check and val_to_find is not None:
                try:
                    target_dtype = je_detail_df[col_to_check].dtype
                    # Perform filtering based on data type for accuracy
                    if pd.api.types.is_numeric_dtype(target_dtype):
                        # Use np.isclose for floating point comparisons if necessary
                        if pd.api.types.is_float_dtype(target_dtype):
                            # Convert val_to_find to float for comparison
                            try: val_num = float(val_to_find); tolerance=1e-6
                            except ValueError: raise ValueError(f"Value '{val_to_find}' is not a valid number for column '{col_to_check}'.")
                            duplicate_jes_df = je_detail_df[np.isclose(je_detail_df[col_to_check].fillna(np.nan), val_num, atol=tolerance)]
                        else: # Integer comparison
                             try: val_num = int(val_to_find)
                             except ValueError: raise ValueError(f"Value '{val_to_find}' is not a valid integer for column '{col_to_check}'.")
                             duplicate_jes_df = je_detail_df[je_detail_df[col_to_check] == val_num]
                    elif pd.api.types.is_datetime64_any_dtype(target_dtype):
                         # Convert val_to_find to datetime if possible
                         val_dt = pd.to_datetime(val_to_find, errors='coerce')
                         if pd.notna(val_dt):
                             # Compare dates (ignoring time if not relevant)
                             duplicate_jes_df = je_detail_df[je_detail_df[col_to_check].dt.date == val_dt.date()]
                         else: # Fallback to string comparison if date conversion fails
                             duplicate_jes_df = je_detail_df[je_detail_df[col_to_check].astype(str).str.strip() == str(val_to_find).strip()]
                    else: # Default to string comparison
                        duplicate_jes_df = je_detail_df[je_detail_df[col_to_check].astype(str).str.strip() == str(val_to_find).strip()]

                    # Display results
                    if not duplicate_jes_df.empty:
                        st.write(f"Found {len(duplicate_jes_df)} matching entries:")
                        # Format the results table
                        dup_df_display = duplicate_jes_df.copy()
                        dup_amount_cols = [col for col in dup_df_display.columns if 'Amount' in col or col == JE_AMOUNT_COLUMN]
                        for col in dup_amount_cols:
                            dup_df_display[col] = dup_df_display[col].apply(utils.format_amount_safely)

                        dup_col_config = {}
                        dup_date_cols = [col for col in dup_df_display.columns if 'Date' in col or col == JE_DATE_COLUMN]
                        for col in dup_date_cols:
                            dup_col_config[col] = st.column_config.DateColumn(format="YYYY-MM-DD")

                        st.dataframe(dup_df_display, use_container_width=True, column_config=dup_col_config, hide_index=True)
                    else:
                        st.info(f"No other JEs found where '{col_to_check}' is '{val_to_find}'.")

                except KeyError:
                    st.error(f"Column '{col_to_check}' not found in the main JE dataset.")
                except ValueError as ve: # Catch specific conversion errors
                    st.error(f"Error during duplicate lookup: {ve}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during duplicate lookup: {e}")
                    st.exception(e)

            # Reset trigger after search attempt
            st.session_state.dup_search_triggered = False
            # Important: Rerun *after* resetting the trigger if the button caused the initial run
            # This prevents infinite loops if there was an error, but ensures display updates.
            # However, the rerun after button press already handles this. Avoid double rerun.


    else:
        st.warning("No suitable columns available for duplicate checking in the JE data.")
else:
    st.info("Select an Account/Period with Journal Entries displayed to enable duplicate lookup.")