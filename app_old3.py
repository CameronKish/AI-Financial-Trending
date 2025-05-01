import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- SET PAGE CONFIG FIRST ---
st.set_page_config(layout="wide", page_title="P&L Analyzer")

# --- Add Current Time Context & Location ---
current_time = datetime.now()
st.sidebar.write(f"Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
st.sidebar.write(f"Location Context: Denver, CO, USA") # Placeholder

# --- Define EY Parthenon Inspired Colors ---
EY_YELLOW = "#FFE600"
EY_DARK_BLUE_GREY = "#2E2E38"
EY_TEXT_ON_YELLOW = EY_DARK_BLUE_GREY

# --- Import data and functions ---
try:
    exec(open("data_processor.py").read(), globals())
    if not all(k in globals() for k in ['pl_flat_df', 'je_detail_df', 'get_journal_entries', 'PL_ID_COLUMN', 'PL_MAP_COLUMN', 'JE_ID_COLUMN', 'JE_DATE_COLUMN']):
        missing = [k for k in ['pl_flat_df', 'je_detail_df', 'get_journal_entries', 'PL_ID_COLUMN', 'PL_MAP_COLUMN', 'JE_ID_COLUMN', 'JE_DATE_COLUMN'] if k not in globals()]
        st.error(f"Could not load necessary data/functions from data_processor.py. Missing: {missing}")
        st.stop()
except FileNotFoundError:
    st.error("Error: `data_processor.py` not found.")
    st.stop()
except Exception as e:
    st.error(f"Error running `data_processor.py`: {e}")
    st.stop()

# --- Initialize Session State ---
if 'selected_account_id' not in st.session_state: st.session_state.selected_account_id = None
if 'selected_account_name' not in st.session_state: st.session_state.selected_account_name = None
if 'selected_period' not in st.session_state: st.session_state.selected_period = None
if 'prev_selected_account_id' not in st.session_state: st.session_state.prev_selected_account_id = None
if 'prev_selected_period' not in st.session_state: st.session_state.prev_selected_period = None
if 'related_jes_df' not in st.session_state: st.session_state.related_jes_df = pd.DataFrame()
if 'dup_col' not in st.session_state: st.session_state.dup_col = None
if 'dup_val' not in st.session_state: st.session_state.dup_val = None
if 'dup_search_triggered' not in st.session_state: st.session_state.dup_search_triggered = False


# --- Streamlit App Layout ---
st.markdown(f"<h1 style='color: {EY_DARK_BLUE_GREY};'>P&L Analyzer (Table Click & Sidebar Select)</h1>", unsafe_allow_html=True)

# --- Prepare P&L Wide View ---
try:
    pl_flat_df['Period_dt'] = pd.to_datetime(pl_flat_df['Period'], errors='coerce')
    valid_periods = pl_flat_df['Period_dt'].notna()
    if not valid_periods.all():
         pl_flat_df['Period_Str'] = pl_flat_df.apply(lambda row: row['Period_dt'].strftime('%Y-%m') if pd.notna(row['Period_dt']) else str(row['Period']), axis=1)
    else: pl_flat_df['Period_Str'] = pl_flat_df['Period_dt'].dt.strftime('%Y-%m')
    period_col_for_pivot = 'Period_Str'
    pnl_wide_view_df = pl_flat_df.pivot_table(index=[PL_ID_COLUMN, PL_MAP_COLUMN], columns=period_col_for_pivot, values='Amount').fillna(0).sort_index(axis=1)
    pnl_wide_view_df_reset = pnl_wide_view_df.reset_index()
    diff_df = pnl_wide_view_df.diff(axis=1)
    period_options = pnl_wide_view_df.columns.tolist()
    threshold_std_dev = 2.0 # Keep default, slider removed for now
    row_std_diff = pnl_wide_view_df.diff(axis=1).std(axis=1, skipna=True).fillna(0)
    outlier_threshold_values_mi = threshold_std_dev * row_std_diff
except Exception as e:
    st.error(f"Error preparing P&L data: {e}")
    st.exception(e); st.stop()

# --- Outlier Highlighting Function ---
def highlight_outliers_pandas(row, diffs_df, thresholds_series):
    try:
        row_diff = diffs_df.loc[row.name]
        threshold_value = thresholds_series.loc[row.name]
    except KeyError: return [''] * len(row)
    except Exception: return [''] * len(row)
    styles = [''] * len(row.index)
    for i, period in enumerate(row.index):
        if i == 0: continue
        diff_val = row_diff.get(period)
        if pd.notna(diff_val) and pd.notna(threshold_value) and threshold_value > 1e-6 and abs(diff_val) > threshold_value:
            if i < len(styles): styles[i] = f'background-color: {EY_YELLOW}; color: {EY_TEXT_ON_YELLOW};'
    return styles

# --- Sidebar Selection Widgets ---
st.sidebar.header("Select Data for JE Lookup")
# --- ADDED: Outlier Sensitivity Slider ---
threshold_std_dev = st.sidebar.slider( # Define slider here
    "Outlier Sensitivity (Std Deviations)",
    min_value=1.0, max_value=4.0, value=2.0, step=0.1,
    help="Lower value = More sensitive (more yellow highlights)."
)
st.sidebar.markdown("---") # Separator
# --- END: Outlier Sensitivity Slider ---
st.sidebar.caption("Use these OR click row/column in the table.")
account_options = pnl_wide_view_df.index.get_level_values(PL_MAP_COLUMN).unique().tolist()
account_name_to_id_map = {name: id_ for id_, name in pnl_wide_view_df.index.unique()}
account_id_to_name_map = {id_: name for id_, name in pnl_wide_view_df.index.unique()}

def get_index(options_list, value):
    try: return options_list.index(value)
    except ValueError: return 0

current_account_name = account_id_to_name_map.get(st.session_state.selected_account_id, account_options[0] if account_options else None)
account_index = get_index(sorted(account_options), current_account_name) if current_account_name else 0
period_index = get_index(period_options, st.session_state.selected_period) if st.session_state.selected_period else len(period_options) - 1

sb_selected_account_name = st.sidebar.selectbox("Select GL Account:", options=sorted(account_options), index=account_index, key="sb_account")
sb_selected_period = st.sidebar.selectbox("Select Period:", options=period_options, index=period_index, key="sb_period")

# --- Synchronize Sidebar selection back to central session state ---
sidebar_account_id = account_name_to_id_map.get(sb_selected_account_name)
sidebar_period = sb_selected_period
sidebar_changed = False
if sidebar_account_id != st.session_state.selected_account_id or sidebar_period != st.session_state.selected_period:
    st.session_state.selected_account_id = sidebar_account_id; st.session_state.selected_account_name = sb_selected_account_name; st.session_state.selected_period = sidebar_period
    st.session_state.selected_je_col = None; st.session_state.selected_je_val = None
    sidebar_changed = True

# --- Main Area ---
st.markdown(f"<h2 style='color: {EY_DARK_BLUE_GREY};'>Profit & Loss Overview (Monthly)</h2>", unsafe_allow_html=True)
st.caption("Highlighting indicates MoM change > selected Std Deviations. Click row index + column header to select via table.")
try:
    # Recalculate thresholds using the slider value *before* applying style
    outlier_threshold_values_mi = threshold_std_dev * row_std_diff
    styled_df = pnl_wide_view_df.style.apply(highlight_outliers_pandas, axis=1, diffs_df=diff_df, thresholds_series=outlier_threshold_values_mi).format("{:,.0f}")
    st.dataframe(styled_df, use_container_width=True, key="pnl_select_df", on_select="rerun", selection_mode=("single-row", "single-column"))
except Exception as e_style:
     st.error(f"Error applying styles or displaying styled dataframe: {e_style}"); st.exception(e_style)
     st.info("Displaying unstyled DataFrame as fallback:")
     st.dataframe(pnl_wide_view_df.style.format("{:,.0f}"), use_container_width=True)

# --- Process Table Selection -> Update Central State ---
df_selection_processed = False; table_changed = False
if "pnl_select_df" in st.session_state:
    selection_state = st.session_state.pnl_select_df.selection
    selected_rows_indices = selection_state.get('rows', []); selected_cols_names = selection_state.get('columns', [])
    if selected_rows_indices and selected_cols_names:
        try:
            selected_row_pos_index = selected_rows_indices[0]; selected_row_data = pnl_wide_view_df_reset.iloc[selected_row_pos_index]
            table_account_id = str(selected_row_data[PL_ID_COLUMN]); table_account_name = selected_row_data[PL_MAP_COLUMN]; table_period = selected_cols_names[0]
            if table_account_id != st.session_state.selected_account_id or table_period != st.session_state.selected_period:
                st.session_state.selected_account_id = table_account_id; st.session_state.selected_account_name = table_account_name; st.session_state.selected_period = table_period
                st.session_state.selected_je_col = None; st.session_state.selected_je_val = None
                df_selection_processed = True; table_changed = True
                st.rerun()
        except Exception as e_proc: st.warning(f"Could not process Table selection: {e_proc}")

# --- Fetch Journal Entries (Based on Central State) ---
st.markdown(f"<hr><h2 style='color: {EY_DARK_BLUE_GREY};'>Journal Entry Details</h2>", unsafe_allow_html=True)
should_fetch = False
if st.session_state.selected_account_id and st.session_state.selected_period:
     if (st.session_state.selected_account_id != st.session_state.prev_selected_account_id or
         st.session_state.selected_period != st.session_state.prev_selected_period or
         (st.session_state.related_jes_df.empty and (sidebar_changed or table_changed))):
            should_fetch = True
if should_fetch and not table_changed:
    try:
        st.session_state.related_jes_df = get_journal_entries(st.session_state.selected_account_id, st.session_state.selected_period, je_detail_df)
        st.session_state.prev_selected_account_id = st.session_state.selected_account_id; st.session_state.prev_selected_period = st.session_state.selected_period
        st.session_state.selected_je_col = None; st.session_state.selected_je_val = None
    except Exception as e_fetch:
        st.error(f"An error occurred while fetching Journal Entries: {e_fetch}")
        st.session_state.related_jes_df = pd.DataFrame()

# --- Display JE Table (Pre-formatting Amount Columns) ---
related_jes_to_display = st.session_state.related_jes_df
if st.session_state.selected_account_id and st.session_state.selected_period:
    st.write(f"Showing JEs for: **{st.session_state.selected_account_name} ({st.session_state.selected_account_id})** | Period: **{st.session_state.selected_period}**")
    if isinstance(related_jes_to_display, pd.DataFrame) and not related_jes_to_display.empty:
        # Create a copy for display formatting
        je_display_df = related_jes_to_display.copy()

        # Identify amount columns
        je_amount_cols = [col for col in je_display_df.columns if 'Amount' in col or 'Debit' in col or 'Credit' in col]

        # Apply string formatting with commas and 0 decimals to amount columns
        for col in je_amount_cols:
             # Handle potential errors if column isn't purely numeric
             try:
                  je_display_df[col] = je_display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
             except (TypeError, ValueError):
                  st.warning(f"Could not apply number format to column '{col}' in JE Details. Displaying as is.")

        # Define column config ONLY for non-amount columns (like Date)
        je_col_config = {}
        date_cols_to_format = []
        if JE_DATE_COLUMN in je_display_df.columns: date_cols_to_format.append(JE_DATE_COLUMN)
        else: date_cols_to_format = [col for col in je_display_df.columns if 'Date' in col]
        for col in date_cols_to_format:
             je_col_config[col] = st.column_config.DateColumn(format="YYYY-MM-DD")

        # Display the pre-formatted DataFrame
        st.dataframe(
            je_display_df, # Pass the dataframe with amount cols as strings
            use_container_width=True,
            column_config=je_col_config # Apply config only to remaining cols (Date)
            )
    elif isinstance(related_jes_to_display, pd.DataFrame) and related_jes_to_display.empty:
        st.info(f"No Journal Entries found for this account and period.")
    else: st.warning("JE data is in an unexpected state.")
else: st.info("Select an Account and Period to view Journal Entries.")


# --- Duplicate JE Finder (Pre-formatting Amount Columns) ---
st.markdown(f"<hr><h2 style='color: {EY_DARK_BLUE_GREY};'>Duplicate Value Lookup</h2>", unsafe_allow_html=True)
if not related_jes_to_display.empty: # Base this on whether JEs are currently shown
    potential_dup_cols = ['Customer', 'Memo', 'Transaction Id', 'Amount (Presentation Currency)']
    available_dup_cols = [col for col in potential_dup_cols if col in je_detail_df.columns]
    if available_dup_cols:
        col1, col2 = st.columns(2);
        with col1:
            last_dup_col_index = available_dup_cols.index(st.session_state.dup_col) if st.session_state.dup_col in available_dup_cols else 0
            selected_dup_col = st.selectbox("Select Column to Check:", options=available_dup_cols, index=last_dup_col_index, key='dup_col_select')
        with col2:
            if selected_dup_col:
                if selected_dup_col in related_jes_to_display.columns:
                     value_options = sorted(related_jes_to_display[selected_dup_col].dropna().unique())
                     if st.session_state.dup_val not in value_options: st.session_state.dup_val = value_options[0] if value_options else None
                     last_dup_val_index = value_options.index(st.session_state.dup_val) if st.session_state.dup_val in value_options else 0
                     selected_dup_val = st.selectbox(f"Select Value from Current JEs:", options=value_options, index=last_dup_val_index, key='dup_val_select')
                else: selected_dup_val = None; _ = st.selectbox(f"Select Value:", options=[], key='dup_val_select', disabled=True, index=0)
            else: selected_dup_val = None; _ = st.selectbox(f"Select Value:", options=[], key='dup_val_select', disabled=True, index=0)
        find_duplicates_button = st.button("Find All Duplicates for Selected Value")
        if find_duplicates_button:
            st.session_state.dup_col = selected_dup_col; st.session_state.dup_val = selected_dup_val
            st.session_state.dup_search_triggered = True

        # Display duplicate results
        if st.session_state.dup_search_triggered and st.session_state.dup_col and st.session_state.dup_val is not None:
            col_to_check = st.session_state.dup_col; val_to_find = st.session_state.dup_val
            st.write(f"Finding all JEs where **{col_to_check}** is **'{val_to_find}'**...")
            try: # Filter logic...
                if pd.api.types.is_numeric_dtype(je_detail_df[col_to_check].dtype) and pd.api.types.is_number(val_to_find): duplicate_jes_df = je_detail_df[np.isclose(je_detail_df[col_to_check].fillna(np.nan), val_to_find)]
                elif pd.api.types.is_datetime64_any_dtype(je_detail_df[col_to_check].dtype) and isinstance(val_to_find, (datetime, pd.Timestamp)): val_to_find_dt = pd.to_datetime(val_to_find, errors='coerce'); duplicate_jes_df = je_detail_df[je_detail_df[col_to_check] == val_to_find_dt] if pd.notna(val_to_find_dt) else je_detail_df[je_detail_df[col_to_check].astype(str).str.strip() == str(val_to_find).strip()]
                else: duplicate_jes_df = je_detail_df[je_detail_df[col_to_check].astype(str).str.strip() == str(val_to_find).strip()]

                if not duplicate_jes_df.empty:
                    st.write(f"Found {len(duplicate_jes_df)} entries:")
                    # Create a copy for display formatting
                    dup_df_display = duplicate_jes_df.copy()
                    # Identify amount columns
                    dup_amount_cols = [col for col in dup_df_display.columns if 'Amount' in col or 'Debit' in col or 'Credit' in col]
                    # Apply string formatting
                    for col in dup_amount_cols:
                         try:
                            dup_df_display[col] = dup_df_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
                         except (TypeError, ValueError):
                            st.warning(f"Could not apply number format to column '{col}' in Duplicate JEs. Displaying as is.")

                    # Define column config ONLY for non-amount columns (like Date)
                    dup_col_config = {}
                    dup_date_cols = [col for col in dup_df_display.columns if 'Date' in col]
                    for col in dup_date_cols: dup_col_config[col] = st.column_config.DateColumn(format="YYYY-MM-DD")

                    # Display the pre-formatted DataFrame
                    st.dataframe(dup_df_display, use_container_width=True, column_config=dup_col_config)
                else: st.info(f"No other JEs found where '{col_to_check}' is '{val_to_find}'.")
            except KeyError: st.error(f"Column '{col_to_check}' not found.")
            except Exception as e: st.error(f"An error occurred during duplicate lookup: {e}")
            # Reset trigger only after potential display attempt
            st.session_state.dup_search_triggered = False
    # else: st.warning("No suitable columns found for duplicate checking.")
else: st.info("Select Account/Period with Journal Entries displayed to enable duplicate lookup.")