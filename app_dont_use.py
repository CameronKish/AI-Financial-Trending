import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go

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
    JE_AMOUNT_COLUMN = 'Amount (Presentation Currency)'
    if 'pl_flat_df' not in globals() or not isinstance(pl_flat_df, pd.DataFrame): st.error("Failed 'pl_flat_df'."); st.stop()
    required_pl_cols = [PL_ID_COLUMN, PL_MAP_COLUMN] + [col for col in ['Mapping 1', 'Mapping 2', 'Mapping 3'] if col in locals() and col in pl_flat_df.columns]
    if not all(col in pl_flat_df.columns for col in required_pl_cols):
        missing_pl = [col for col in required_pl_cols if col not in pl_flat_df.columns]; st.error(f"P&L missing required columns: {missing_pl}"); st.stop()
    if 'je_detail_df' not in globals() or not isinstance(je_detail_df, pd.DataFrame): st.error("Failed 'je_detail_df'."); st.stop()
    required_je_cols = [JE_AMOUNT_COLUMN, JE_DATE_COLUMN, 'Account Name', JE_ID_COLUMN]
    if not all(col in je_detail_df.columns for col in required_je_cols):
        missing_je = [col for col in required_je_cols if col not in je_detail_df.columns]; st.error(f"JE missing columns: {missing_je}"); st.stop()
    if 'Account Name' not in je_detail_df.columns: JE_ACCOUNT_NAME_COL = None
    else: JE_ACCOUNT_NAME_COL = 'Account Name'
except FileNotFoundError: st.error("`data_processor.py` not found."); st.stop()
except NameError as e: st.error(f"Variable not defined in data_processor.py? {e}"); st.stop()
except Exception as e: st.error(f"Error loading data: {e}"); st.stop()

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
if 'chart_accounts_selection' not in st.session_state: st.session_state.chart_accounts_selection = []
# Find available mapping levels for the radio button selector default
available_mapping_levels_init = sorted([col for col in pl_flat_df.columns if 'Mapping' in col])
view_options_init = available_mapping_levels_init + ['Account ID Detail']
default_view_init = PL_MAP_COLUMN if PL_MAP_COLUMN in view_options_init else view_options_init[-1]
if 'pnl_agg_level' not in st.session_state: st.session_state.pnl_agg_level = default_view_init


# --- Robust Formatting Function (Corrected Indentation) ---
def format_amount_safely(value):
    """
    Formats numeric values as strings with commas and 0 decimals.
    Returns non-numeric values as strings, handles NaN/None.
    """
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float, np.number)):
        try:
            return f"{value:,.0f}"
        except (TypeError, ValueError):
            return str(value) # Fallback
    else:
        try:
            num = pd.to_numeric(value)
            if pd.isna(num):
                 return str(value)
            return f"{num:,.0f}"
        except (TypeError, ValueError):
            return str(value)

# --- Corrected get_index function definition (Corrected Indentation) ---
def get_index(options_list, value):
    """Safely finds the index of a value in a list, returning 0 if not found."""
    try:
        return options_list.index(value)
    except ValueError:
        return 0 # Default to first item

# --- Streamlit App Layout ---
st.markdown(f"<h1 style='color: {EY_DARK_BLUE_GREY};'>P&L Analyzer (Aggregated View)</h1>", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("Controls")
threshold_std_dev = st.sidebar.slider("Outlier Sensitivity (Std Deviations)", 1.0, 4.0, 2.0, 0.1, help="Lower = More sensitive")
st.sidebar.markdown("---")
st.sidebar.header("Date Range for P&L Table")
if 'Period_dt' not in pl_flat_df.columns: pl_flat_df['Period_dt'] = pd.to_datetime(pl_flat_df['Period'], errors='coerce')
min_date = pl_flat_df['Period_dt'].min(); max_date = pl_flat_df['Period_dt'].max()
default_end_date = max_date if pd.notna(max_date) else date.today(); default_start_date = min_date if pd.notna(min_date) else default_end_date - pd.DateOffset(years=1)
if pd.isna(default_start_date) or pd.isna(default_end_date): default_start_date = date.today() - pd.DateOffset(years=1); default_end_date = date.today()
start_date = st.sidebar.date_input("Start Date", value=default_start_date, min_value=min_date, max_value=max_date, key='start_date')
end_date = st.sidebar.date_input("End Date", value=default_end_date, min_value=start_date, max_value=max_date, key='end_date')
start_datetime = pd.to_datetime(start_date); end_datetime = pd.to_datetime(end_date)
st.sidebar.markdown("---")
st.sidebar.header("Select Account for JE Lookup")
st.sidebar.caption("JE lookup driven by this selection.")

# --- Prepare P&L Data ---
try:
    if 'Period_dt' in pl_flat_df.columns and pd.api.types.is_datetime64_any_dtype(pl_flat_df['Period_dt']):
         pl_flat_df_filtered = pl_flat_df[(pl_flat_df['Period_dt'] >= start_datetime) & (pl_flat_df['Period_dt'] <= end_datetime)].copy()
    else: st.warning("Date column missing/invalid."); pl_flat_df_filtered = pl_flat_df.copy()
    pnl_wide_view_df, pnl_wide_view_df_reset, diff_df, period_options, row_std_diff = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], pd.Series()
    x_axis_col_chart = 'Period_Str'; has_valid_dates = False # Defaults
    if not pl_flat_df_filtered.empty:
        pl_flat_df_filtered['Period_dt'] = pd.to_datetime(pl_flat_df_filtered['Period'], errors='coerce')
        pl_flat_df_filtered['Period_Str'] = pl_flat_df_filtered['Period_dt'].dt.strftime('%Y-%m')
        pl_flat_df_filtered.dropna(subset=['Period_Str'], inplace=True)
        has_valid_dates = pl_flat_df_filtered['Period_dt'].notna().all(); x_axis_col_chart = 'Period_dt' if has_valid_dates else 'Period_Str'
        period_col_for_pivot = 'Period_Str'
        all_mapping_cols = sorted([col for col in pl_flat_df_filtered.columns if 'Mapping' in col])
        pivot_index_cols = [PL_ID_COLUMN] + all_mapping_cols
        if not all(col in pl_flat_df_filtered.columns for col in pivot_index_cols):
            missing_pivot_idx = [col for col in pivot_index_cols if col not in pl_flat_df_filtered.columns]; st.error(f"P&L missing cols for pivot: {missing_pivot_idx}"); st.stop()
        pnl_wide_view_df = pl_flat_df_filtered.pivot_table(index=pivot_index_cols, columns=period_col_for_pivot, values='Amount').fillna(0).sort_index(axis=1)
        if not pnl_wide_view_df.empty:
            pnl_wide_view_df_reset = pnl_wide_view_df.reset_index(); diff_df = pnl_wide_view_df.diff(axis=1); period_options = pnl_wide_view_df.columns.tolist(); row_std_diff = pnl_wide_view_df.diff(axis=1).std(axis=1, skipna=True).fillna(0)
except Exception as e: st.error(f"Error preparing P&L data: {e}"); st.exception(e); st.stop()

# --- Outlier Highlighting Function ---
def highlight_outliers_pandas(row, diffs_df, thresholds_series):
    try: row_diff = diffs_df.loc[row.name]; threshold_value = thresholds_series.loc[row.name]
    except KeyError: return [''] * len(row); 
    except Exception: return [''] * len(row)
    styles = [''] * len(row.index);
    for i, period in enumerate(row.index):
        if i == 0: continue; diff_val = row_diff.get(period)
        if pd.notna(diff_val) and pd.notna(threshold_value) and threshold_value > 1e-6 and abs(diff_val) > threshold_value:
            if i < len(styles): styles[i] = f'background-color: {EY_YELLOW}; color: {EY_TEXT_ON_YELLOW};'
    return styles

# --- Sidebar Selection Widgets Logic ---
account_options = sorted(pnl_wide_view_df.index.get_level_values(PL_MAP_COLUMN).unique().tolist()) if not pnl_wide_view_df.empty else []
if not pnl_wide_view_df.empty:
     map_df = pnl_wide_view_df.index.to_frame(index=False); account_name_to_id_map = map_df.drop_duplicates(subset=[PL_MAP_COLUMN]).set_index(PL_MAP_COLUMN)[PL_ID_COLUMN].to_dict(); account_id_to_name_map = map_df.drop_duplicates(subset=[PL_ID_COLUMN]).set_index(PL_ID_COLUMN)[PL_MAP_COLUMN].to_dict()
else: account_name_to_id_map = {}; account_id_to_name_map = {}
current_account_name = account_id_to_name_map.get(st.session_state.selected_account_id, account_options[0] if account_options else None)
account_index = get_index(account_options, current_account_name) # Use unsorted options list from unique()
if st.session_state.selected_period not in period_options: st.session_state.selected_period = period_options[-1] if period_options else None
period_index = get_index(period_options, st.session_state.selected_period) if st.session_state.selected_period else 0
sb_selected_account_name = st.sidebar.selectbox("Select Account Name (for JE):", options=account_options, index=account_index, key="sb_account")
sb_selected_period = st.sidebar.selectbox("Select Period (for JE):", options=period_options, index=period_index, key="sb_period")

# --- Synchronize Sidebar selection ---
sidebar_account_id = account_name_to_id_map.get(sb_selected_account_name); sidebar_period = sb_selected_period; sidebar_changed = False
if sidebar_account_id != st.session_state.selected_account_id or sidebar_period != st.session_state.selected_period:
    st.session_state.selected_account_id = sidebar_account_id
    st.session_state.selected_account_name = sb_selected_account_name
    st.session_state.selected_period = sidebar_period
    st.session_state.selected_je_col = None
    st.session_state.selected_je_val = None
    sidebar_changed = True

# --- Fetch Journal Entries ---
should_fetch = False; table_changed_flag = False
if st.session_state.selected_account_id and st.session_state.selected_period:
     if (st.session_state.selected_account_id != st.session_state.prev_selected_account_id or
         st.session_state.selected_period != st.session_state.prev_selected_period or
         (st.session_state.related_jes_df.empty and (sidebar_changed))): should_fetch = True
if 'df_selection_processed' in locals() and locals().get('df_selection_processed'): table_changed_flag = True
if should_fetch and not table_changed_flag:
    try:
        st.session_state.related_jes_df = get_journal_entries(st.session_state.selected_account_id, st.session_state.selected_period, je_detail_df)
        st.session_state.prev_selected_account_id = st.session_state.selected_account_id
        st.session_state.prev_selected_period = st.session_state.selected_period
        st.session_state.selected_je_col = None
        st.session_state.selected_je_val = None
    except Exception as e_fetch: st.error(f"An error fetching JEs: {e_fetch}"); st.session_state.related_jes_df = pd.DataFrame()

# --- Define Tabs ---
tab1, tab2 = st.tabs([" P&L Analysis & Drilldown ", " Visualizations "])

# --- TAB 1: P&L ANALYSIS & DRILLDOWN ---
with tab1:
    st.markdown(f"<h2 style='color: {EY_DARK_BLUE_GREY};'>Profit & Loss Overview (Monthly)</h2>", unsafe_allow_html=True)
    # Aggregation Level Selector
    mapping_levels = sorted([col for col in pl_flat_df_filtered.columns if 'Mapping' in col])
    view_options = mapping_levels + ['Account ID Detail']
    default_view = st.session_state.pnl_agg_level if st.session_state.pnl_agg_level in view_options else view_options[-1]
    agg_level_index = view_options.index(default_view) if default_view in view_options else len(view_options)-1
    selected_level = st.radio("Select P&L View Level:", options=view_options, index=agg_level_index, horizontal=True, key="pnl_agg_level_selector")
    if selected_level != st.session_state.pnl_agg_level: st.session_state.pnl_agg_level = selected_level; st.rerun()

    # Conditionally Aggregate Data for Display
    display_df_agg = pd.DataFrame(); enable_selection = False; apply_highlighting = False
    if pl_flat_df_filtered.empty: st.warning("No P&L data available for selected date range.")
    else:
        try: # Aggregate based on selection
            current_agg_level = st.session_state.pnl_agg_level
            if current_agg_level == 'Account ID Detail': display_df_agg = pnl_wide_view_df; enable_selection = True; apply_highlighting = True
            elif current_agg_level in mapping_levels:
                level_index = mapping_levels.index(current_agg_level); grouping_cols_agg = mapping_levels[:level_index+1] + ['Period_Str']
                if all(col in pl_flat_df_filtered.columns for col in grouping_cols_agg): display_df_agg = pl_flat_df_filtered.groupby(grouping_cols_agg[:-1])['Amount'].sum().unstack(fill_value=0).reindex(columns=period_options, fill_value=0).sort_index(axis=1)
                else: st.error(f"Missing cols for grouping: {current_agg_level}."); display_df_agg = pnl_wide_view_df; enable_selection = True; apply_highlighting = True; st.session_state.pnl_agg_level = 'Account ID Detail'
            else: st.error("Invalid level."); display_df_agg = pnl_wide_view_df; enable_selection = True; apply_highlighting = True; st.session_state.pnl_agg_level = 'Account ID Detail'
        except Exception as agg_err: st.error(f"Error aggregating data: {agg_err}"); display_df_agg = pnl_wide_view_df; enable_selection = True; apply_highlighting = True; st.session_state.pnl_agg_level = 'Account ID Detail'

    # Display P&L Table
    st.caption(f"Displaying: {st.session_state.pnl_agg_level}. Dates: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
    if apply_highlighting: st.caption("Highlighting indicates MoM change > selected Std Deviations.")
    else: st.caption("Highlighting disabled for aggregated views.")
    if not display_df_agg.empty:
        try: # Display with conditional styling/selection
            if apply_highlighting: outlier_threshold_values_mi = threshold_std_dev * row_std_diff; styled_display_df = display_df_agg.style.apply(highlight_outliers_pandas, axis=1, diffs_df=diff_df, thresholds_series=outlier_threshold_values_mi).format("{:,.0f}")
            else: styled_display_df = display_df_agg.style.format("{:,.0f}")
            if enable_selection: st.dataframe(styled_display_df, use_container_width=True, key="pnl_select_df", on_select="rerun", selection_mode=("single-row", "single-column"))
            else: st.dataframe(styled_display_df, use_container_width=True, key="pnl_select_df")
        except Exception as e_style: st.error(f"Error displaying P&L table: {e_style}"); st.exception(e_style); st.dataframe(display_df_agg.style.format("{:,.0f}"), use_container_width=True)

    # Process Table Selection (only if enabled)
    df_selection_processed = False; table_changed = False
    if enable_selection and "pnl_select_df" in st.session_state and not pnl_wide_view_df.empty:
        selection_state = st.session_state.pnl_select_df.selection
        selected_rows_indices = selection_state.get('rows', []); selected_cols_names = selection_state.get('columns', [])
        if selected_rows_indices and selected_cols_names:
            try:
                selected_row_pos_index = selected_rows_indices[0]
                if selected_row_pos_index < len(pnl_wide_view_df_reset):
                    selected_row_data = pnl_wide_view_df_reset.iloc[selected_row_pos_index]
                    table_account_id = str(selected_row_data[PL_ID_COLUMN]); table_account_name = selected_row_data[PL_MAP_COLUMN]; table_period = selected_cols_names[0]
                    if table_account_id != st.session_state.selected_account_id or table_period != st.session_state.selected_period:
                        st.session_state.selected_account_id = table_account_id; st.session_state.selected_account_name = table_account_name; st.session_state.selected_period = table_period
                        st.session_state.selected_je_col = None; st.session_state.selected_je_val = None
                        df_selection_processed = True; table_changed = True
                        st.rerun()
                else: st.warning("Selected row index out of bounds.")
            except Exception as e_proc: st.warning(f"Could not process Table selection: {e_proc}")

    # Display Journal Entry Details
    st.markdown(f"<hr><h2 style='color: {EY_DARK_BLUE_GREY};'>Journal Entry Details</h2>", unsafe_allow_html=True)
    related_jes_to_display = st.session_state.related_jes_df
    if st.session_state.selected_account_id and st.session_state.selected_period:
        st.write(f"Showing JEs for Account ID: **{st.session_state.selected_account_id}** | Name: **{st.session_state.selected_account_name}** | Period: **{st.session_state.selected_period}**")
        if isinstance(related_jes_to_display, pd.DataFrame) and not related_jes_to_display.empty:
            je_display_df = related_jes_to_display.copy(); je_amount_cols = [col for col in je_display_df.columns if 'Amount' in col or 'Debit' in col or 'Credit' in col]
            # Use robust formatting function
            for col in je_amount_cols: je_display_df[col] = je_display_df[col].apply(format_amount_safely)
            je_col_config = {}; date_cols_to_format = [col for col in je_display_df.columns if 'Date' in col or JE_DATE_COLUMN in col]
            for col in date_cols_to_format: je_col_config[col] = st.column_config.DateColumn(format="YYYY-MM-DD", help="Transaction Date")
            st.dataframe(je_display_df, use_container_width=True, column_config=je_col_config)
        elif isinstance(related_jes_to_display, pd.DataFrame) and related_jes_to_display.empty: st.info(f"No Journal Entries found.")
        else: st.warning("JE data is in an unexpected state.")
    else: st.info("Select Account/Period.")

    # Duplicate JE Finder
    st.markdown(f"<hr><h2 style='color: {EY_DARK_BLUE_GREY};'>Duplicate Value Lookup</h2>", unsafe_allow_html=True)
    if not related_jes_to_display.empty:
        # ... (Duplicate finder logic remains the same) ...
        potential_dup_cols = ['Customer', 'Memo', 'Transaction Id', 'Amount (Presentation Currency)']; available_dup_cols = [col for col in potential_dup_cols if col in je_detail_df.columns]
        if available_dup_cols:
            col1, col2 = st.columns(2);
            with col1: last_dup_col_index = available_dup_cols.index(st.session_state.dup_col) if st.session_state.dup_col in available_dup_cols else 0; selected_dup_col = st.selectbox("Select Column to Check:", options=available_dup_cols, index=last_dup_col_index, key='dup_col_select')
            with col2:
                if selected_dup_col and selected_dup_col in related_jes_to_display.columns: value_options = sorted(related_jes_to_display[selected_dup_col].dropna().unique()); last_dup_val_index = value_options.index(st.session_state.dup_val) if st.session_state.dup_val in value_options else 0; selected_dup_val = st.selectbox(f"Select Value from Current JEs:", options=value_options, index=last_dup_val_index, key='dup_val_select')
                else: selected_dup_val = None; _ = st.selectbox(f"Select Value:", options=[], key='dup_val_select', disabled=True, index=0)
            find_duplicates_button = st.button("Find All Duplicates for Selected Value")
            if find_duplicates_button: st.session_state.dup_col = selected_dup_col; st.session_state.dup_val = selected_dup_val; st.session_state.dup_search_triggered = True
            if st.session_state.dup_search_triggered and st.session_state.dup_col and st.session_state.dup_val is not None:
                col_to_check = st.session_state.dup_col; val_to_find = st.session_state.dup_val; st.write(f"Finding all JEs where **{col_to_check}** is **'{val_to_find}'**...")
                try: # Filter logic...
                    if pd.api.types.is_numeric_dtype(je_detail_df[col_to_check].dtype) and pd.api.types.is_number(val_to_find): duplicate_jes_df = je_detail_df[np.isclose(je_detail_df[col_to_check].fillna(np.nan), val_to_find)]
                    elif pd.api.types.is_datetime64_any_dtype(je_detail_df[col_to_check].dtype) and isinstance(val_to_find, (datetime, pd.Timestamp)): val_to_find_dt = pd.to_datetime(val_to_find, errors='coerce'); duplicate_jes_df = je_detail_df[je_detail_df[col_to_check] == val_to_find_dt] if pd.notna(val_to_find_dt) else je_detail_df[je_detail_df[col_to_check].astype(str).str.strip() == str(val_to_find).strip()]
                    else: duplicate_jes_df = je_detail_df[je_detail_df[col_to_check].astype(str).str.strip() == str(val_to_find).strip()]
                    if not duplicate_jes_df.empty:
                        st.write(f"Found {len(duplicate_jes_df)} entries:")
                        dup_col_config = {}; dup_df_display = duplicate_jes_df.copy()
                        dup_amount_cols = [col for col in dup_df_display.columns if 'Amount' in col or 'Debit' in col or 'Credit' in col]; dup_date_cols = [col for col in dup_df_display.columns if 'Date' in col]
                        for col in dup_amount_cols: dup_df_display[col] = dup_df_display[col].apply(format_amount_safely) # Use robust formatting
                        for col in dup_date_cols: dup_col_config[col] = st.column_config.DateColumn(format="YYYY-MM-DD")
                        st.dataframe(dup_df_display, use_container_width=True, column_config=dup_col_config)
                    else: st.info(f"No other JEs found where '{col_to_check}' is '{val_to_find}'.")
                except KeyError: st.error(f"Column '{col_to_check}' not found.")
                except Exception as e: st.error(f"An error occurred during duplicate lookup: {e}")
                st.session_state.dup_search_triggered = False
        # else: st.warning("No suitable columns found for duplicate checking.")
    else: st.info("Select Account/Period with Journal Entries displayed to enable duplicate lookup.")


# --- TAB 2: VISUALIZATIONS ---
with tab2:
    st.markdown(f"<h2 style='color: {EY_DARK_BLUE_GREY};'>P&L Account Trends</h2>", unsafe_allow_html=True)
    # P&L Trend Chart code (using enhanced multiselect)
    # ... (Code remains the same) ...
    chart_account_options = sorted(pl_flat_df[PL_MAP_COLUMN].unique())
    if 'chart_accounts_selection' not in st.session_state or not st.session_state.chart_accounts_selection :
         default_chart_selection = []; potential_defaults = ["Total Net Sales", "Total COGS/COS", "Total Operating Expenses"]
         for acc in potential_defaults:
              if acc in chart_account_options: default_chart_selection.append(acc)
         if not default_chart_selection and chart_account_options: default_chart_selection = chart_account_options[:min(3, len(chart_account_options))]
         st.session_state.chart_accounts_selection = default_chart_selection
    c1, c2, c3 = st.columns([4, 1, 1])
    with c1: user_chart_selection = st.multiselect("Select Account(s) to Plot:", options=chart_account_options, default=st.session_state.chart_accounts_selection, key="chart_multiselect_widget")
    if user_chart_selection != st.session_state.chart_accounts_selection: st.session_state.chart_accounts_selection = user_chart_selection; st.rerun()
    with c2: st.markdown("<br>", unsafe_allow_html=True);
    if st.button("Select All", key='select_all_chart'): st.session_state.chart_accounts_selection = chart_account_options; st.rerun()
    with c3: st.markdown("<br>", unsafe_allow_html=True);
    if st.button("Clear Selection", key='clear_chart'): st.session_state.chart_accounts_selection = []; st.rerun()
    if st.session_state.chart_accounts_selection:
        chart_data = pl_flat_df[pl_flat_df[PL_MAP_COLUMN].isin(st.session_state.chart_accounts_selection)].copy()
        sort_col = x_axis_col_chart
        if sort_col in chart_data.columns: chart_data = chart_data.sort_values(by=sort_col); x_axis_label = "Period"
        else: chart_data = pd.DataFrame()
        if not chart_data.empty:
            fig = px.line(chart_data, x=sort_col, y='Amount', color=PL_MAP_COLUMN, markers=True, title="Monthly Trend for Selected Accounts")
            fig.update_layout(xaxis_title=x_axis_label, yaxis_title="Amount ($)", yaxis_tickformat=",.0f", hovermode="x unified")
            fig.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Period: %{x}<br>Amount: %{y:,.0f}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No data available.")
    else: st.info("Select accounts to plot.")

    # P&L Waterfall Chart
    st.markdown(f"<hr><h2 style='color: {EY_DARK_BLUE_GREY};'>P&L Waterfall Chart (Single Period)</h2>", unsafe_allow_html=True)
    # ... (Waterfall chart code remains the same) ...
    waterfall_period_options = period_options
    if waterfall_period_options:
        wf_period_index = len(waterfall_period_options) - 1
        wf_default_period = st.session_state.selected_period if st.session_state.selected_period in waterfall_period_options else waterfall_period_options[wf_period_index]
        waterfall_period = st.selectbox("Select Period for Waterfall:", options=waterfall_period_options, index=waterfall_period_options.index(wf_default_period))
        if waterfall_period:
            try:
                period_data = pnl_wide_view_df[waterfall_period]
                waterfall_categories, waterfall_values, waterfall_measures = [], [], []
                # --- Customize based on YOUR P&L ---
                rev = period_data[period_data.index.get_level_values(PL_MAP_COLUMN) == 'Total Net Sales'].sum() # Assumes PL_MAP_COLUMN is lowest level name
                cogs = period_data[period_data.index.get_level_values(PL_MAP_COLUMN) == 'Total COGS/COS'].sum()
                opex = period_data[period_data.index.get_level_values(PL_MAP_LEVEL_1) == 'Opex'].sum() # Assumes Mapping 1 used for Opex category
                if pd.notna(rev): waterfall_categories.append("Revenue"); waterfall_values.append(rev); waterfall_measures.append("absolute")
                if pd.notna(cogs) and abs(cogs)>1e-6 : waterfall_categories.append("COGS"); waterfall_values.append(cogs); waterfall_measures.append("relative")
                gross_profit = rev + cogs;
                if pd.notna(gross_profit): waterfall_categories.append("Gross Profit"); waterfall_values.append(gross_profit); waterfall_measures.append("total")
                if pd.notna(opex) and abs(opex)>1e-6 : waterfall_categories.append("Operating Expenses"); waterfall_values.append(opex); waterfall_measures.append("relative")
                operating_income = gross_profit + opex;
                if pd.notna(operating_income): waterfall_categories.append("Operating Income"); waterfall_values.append(operating_income); waterfall_measures.append("total")
                # --- End Customization ---
                if waterfall_categories:
                    fig_waterfall = go.Figure(go.Waterfall(name = "P&L", orientation = "v", measure = waterfall_measures, x = waterfall_categories, textposition = "outside", text = [f"{v:,.0f}" for v in waterfall_values], y = waterfall_values, connector = {"line":{"color":"rgb(63, 63, 63)"}}, totals = {"marker":{"color":"rgba(0,0,0,0)"}}, decreasing = {"marker":{"color":"#d9534f"}}, increasing = {"marker":{"color":"#5cb85c"}} ))
                    fig_waterfall.update_layout(title=f"P&L Waterfall for {waterfall_period}", showlegend=False, yaxis_title="Amount ($)", yaxis_tickformat=",.0f")
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                else: st.info("Could not construct waterfall.")
            except Exception as e_waterfall: st.error(f"Waterfall error: {e_waterfall}")
    else: st.info("Select a date range with data.")


    # JE Amount Distribution Histogram
    st.markdown(f"<hr><h2 style='color: {EY_DARK_BLUE_GREY};'>Journal Entry Amount Distribution</h2>", unsafe_allow_html=True)
    # ... [JE Histogram Code remains the same] ...
    st.caption("Shows the distribution of transaction amounts across ALL journal entries.")
    if 'je_detail_df' in globals() and isinstance(je_detail_df, pd.DataFrame) and not je_detail_df.empty:
        if JE_AMOUNT_COLUMN in je_detail_df.columns:
            try:
                hist_data = je_detail_df[je_detail_df[JE_AMOUNT_COLUMN].notna() & (je_detail_df[JE_AMOUNT_COLUMN] != 0)][JE_AMOUNT_COLUMN]
                if not hist_data.empty:
                     fig_hist = px.histogram(hist_data, x=JE_AMOUNT_COLUMN, title="Distribution of JE Amounts (Non-Zero)", nbins=50, log_y=True, labels={JE_AMOUNT_COLUMN: "Transaction Amount ($)"})
                     fig_hist.update_layout(yaxis_title="Frequency (Log Scale)", xaxis_tickformat=",.0f")
                     st.plotly_chart(fig_hist, use_container_width=True)
                else: st.info("No non-zero JE amounts found.")
            except Exception as e_hist: st.error(f"Could not generate JE Histogram: {e_hist}")
        else: st.warning(f"Amount column '{JE_AMOUNT_COLUMN}' not found.")
    else: st.warning("Full JE data (`je_detail_df`) is not available or empty.")


    # JE Analysis Scatter Plot
    st.markdown(f"<hr><h2 style='color: {EY_DARK_BLUE_GREY};'>Total JE Analysis (Amount vs. Frequency)</h2>", unsafe_allow_html=True)
    # ... [JE Scatter Plot code remains the same] ...
    st.caption("Analyzes ALL Journal Entries in the dataset by the selected category.")
    if 'je_detail_df' in globals() and isinstance(je_detail_df, pd.DataFrame) and not je_detail_df.empty:
        potential_analysis_cols = ['Customer', 'Memo', 'Account Name']; analysis_cols_options = [col for col in potential_analysis_cols if col in je_detail_df.columns and (pd.api.types.is_string_dtype(je_detail_df[col]) or pd.api.types.is_object_dtype(je_detail_df[col])) and je_detail_df[col].nunique() > 1]
        if JE_AMOUNT_COLUMN not in je_detail_df.columns: st.warning(f"Amount column '{JE_AMOUNT_COLUMN}' not found."); analysis_cols_options = []
        if analysis_cols_options:
            selected_analysis_col_all = st.selectbox("Analyze ALL Journal Entries by:", options=analysis_cols_options, index=0, key="je_analysis_col_tab2_all")
            if selected_analysis_col_all:
                st.write(f"Aggregating all {len(je_detail_df)} journal entries by **{selected_analysis_col_all}**...")
                try:
                    with st.spinner(f"Aggregating by {selected_analysis_col_all}..."):
                         agg_data_all = je_detail_df.groupby(selected_analysis_col_all).agg(Count=('Transaction Id', 'size'), Total_Amount=(JE_AMOUNT_COLUMN, 'sum')).reset_index()
                         agg_data_all['Abs_Total_Amount'] = agg_data_all['Total_Amount'].abs(); agg_data_all = agg_data_all.sort_values(by='Total_Amount', ascending=False)
                    if not agg_data_all.empty:
                        st.write(f"Relationship between Frequency and Total Amount for **{selected_analysis_col_all}** categories across all data:")
                        fig_agg_all = px.scatter(agg_data_all, x='Count', y='Total_Amount', size='Abs_Total_Amount', hover_name=selected_analysis_col_all, title=f'All JE Analysis by {selected_analysis_col_all} (Amount vs. Frequency)', labels={'Count': 'Number of Entries (Frequency)', 'Total_Amount': 'Net Total Amount ($)'})
                        fig_agg_all.update_layout(xaxis_title="Number of Entries (Frequency)", yaxis_title="Net Total Amount ($)", yaxis_tickformat=",.0f", xaxis_tickformat=",d")
                        fig_agg_all.update_traces(hovertemplate=f"<b>%{{hovertext}}</b><br>Net Total Amount: %{{y:,.0f}}<br>Count: %{{x}}<extra></extra>", hovertext=agg_data_all[selected_analysis_col_all])
                        st.plotly_chart(fig_agg_all, use_container_width=True)
                        N_TOP = 25; st.write(f"Top {min(N_TOP, len(agg_data_all))} categories by Net Amount (All JEs):")
                        st.dataframe(agg_data_all.head(N_TOP), use_container_width=True, column_config={selected_analysis_col_all : st.column_config.TextColumn(selected_analysis_col_all), "Count": st.column_config.NumberColumn("Count", format="%d"), "Total_Amount": st.column_config.NumberColumn("Net Total Amount ($)", format="%.0f"), "Abs_Total_Amount": st.column_config.NumberColumn("Abs Amount (for size)", format="%.0f", disabled=True) })
                    else: st.info(f"No aggregated data to display for '{selected_analysis_col_all}' across all JEs.")
                except KeyError as e_agg_key: st.error(f"Error during aggregation: Column mismatch? {e_agg_key}")
                except Exception as e_agg: st.error(f"An error occurred during full JE analysis: {e_agg}"); st.exception(e_agg)
        else: st.info("No suitable text columns found in the full JE dataset for analysis.")
    else: st.warning("Full JE data (`je_detail_df`) is not available or empty.")