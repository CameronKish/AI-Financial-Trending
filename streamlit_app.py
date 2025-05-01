import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

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
    if 'je_detail_df' not in globals() or not isinstance(je_detail_df, pd.DataFrame): st.error("Failed to load 'je_detail_df'."); st.stop()
    if JE_AMOUNT_COLUMN not in je_detail_df.columns: st.error(f"JE Amount column '{JE_AMOUNT_COLUMN}' not found."); st.stop()
    if JE_DATE_COLUMN not in je_detail_df.columns: st.error(f"JE Date column '{JE_DATE_COLUMN}' not found."); st.stop()
    if 'Account Name' not in je_detail_df.columns: JE_ACCOUNT_NAME_COL = None
    else: JE_ACCOUNT_NAME_COL = 'Account Name'
    if not all(k in globals() for k in ['pl_flat_df', 'get_journal_entries', 'PL_ID_COLUMN', 'PL_MAP_COLUMN', 'JE_ID_COLUMN']):
        missing = [k for k in ['pl_flat_df', 'je_detail_df', 'get_journal_entries', 'PL_ID_COLUMN', 'PL_MAP_COLUMN', 'JE_ID_COLUMN', 'JE_DATE_COLUMN'] if k not in globals()]
        st.error(f"Could not load necessary data/functions. Missing: {missing}"); st.stop()
except FileNotFoundError: st.error("Error: `data_processor.py` not found."); st.stop()
except Exception as e: st.error(f"Error running `data_processor.py` or loading data: {e}"); st.stop()

# --- Initialize Session State ---
# (Add state for chart selection)
if 'selected_account_id' not in st.session_state: st.session_state.selected_account_id = None
if 'selected_account_name' not in st.session_state: st.session_state.selected_account_name = None
if 'selected_period' not in st.session_state: st.session_state.selected_period = None
if 'prev_selected_account_id' not in st.session_state: st.session_state.prev_selected_account_id = None
if 'prev_selected_period' not in st.session_state: st.session_state.prev_selected_period = None
if 'related_jes_df' not in st.session_state: st.session_state.related_jes_df = pd.DataFrame()
if 'dup_col' not in st.session_state: st.session_state.dup_col = None
if 'dup_val' not in st.session_state: st.session_state.dup_val = None
if 'dup_search_triggered' not in st.session_state: st.session_state.dup_search_triggered = False
# Initialize chart selection state (using default logic)
if 'chart_accounts_selection' not in st.session_state:
    # Determine default selection for chart (run this logic only once)
    temp_account_options = pl_flat_df[PL_MAP_COLUMN].unique().tolist() # Need options list here
    default_chart_selection = []
    potential_defaults = ["Total Net Sales", "Total COGS/COS", "Total Operating Expenses"]
    for acc in potential_defaults:
        if acc in temp_account_options: default_chart_selection.append(acc)
    if not default_chart_selection and temp_account_options:
        default_chart_selection = temp_account_options[:min(3, len(temp_account_options))]
    st.session_state.chart_accounts_selection = default_chart_selection


# --- Robust Formatting Function ---
def format_amount_safely(value):
    if pd.isna(value): return ""
    if isinstance(value, (int, float, np.number)):
        try: return f"{value:,.0f}"
        except (TypeError, ValueError): return str(value)
    else:
        try: num = pd.to_numeric(value); return f"{num:,.0f}"
        except (TypeError, ValueError): return str(value)

# --- Corrected get_index function definition ---
def get_index(options_list, value):
    try: return options_list.index(value)
    except ValueError: return 0

# --- Streamlit App Layout ---
st.markdown(f"<h1 style='color: {EY_DARK_BLUE_GREY};'>P&L Analyzer (Adjustable Outliers)</h1>", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("Controls")
threshold_std_dev = st.sidebar.slider("Outlier Sensitivity (Std Deviations)", 1.0, 4.0, 2.0, 0.1, help="Lower = More sensitive")
st.sidebar.markdown("---")
st.sidebar.header("Select Data for JE Lookup")
st.sidebar.caption("Use these OR click row/column in the table.")

# --- Prepare P&L Data ---
# (P&L data prep code remains the same)
try:
    pl_flat_df['Period_dt'] = pd.to_datetime(pl_flat_df['Period'], errors='coerce')
    has_valid_dates = pl_flat_df['Period_dt'].notna().all()
    if not has_valid_dates: pl_flat_df['Period_Str'] = pl_flat_df.apply(lambda row: row['Period_dt'].strftime('%Y-%m') if pd.notna(row['Period_dt']) else str(row['Period']), axis=1); x_axis_col_chart = 'Period_Str'
    else: pl_flat_df['Period_Str'] = pl_flat_df['Period_dt'].dt.strftime('%Y-%m'); x_axis_col_chart = 'Period_dt'
    period_col_for_pivot = 'Period_Str'
    pnl_wide_view_df = pl_flat_df.pivot_table(index=[PL_ID_COLUMN, PL_MAP_COLUMN], columns=period_col_for_pivot, values='Amount').fillna(0).sort_index(axis=1)
    pnl_wide_view_df_reset = pnl_wide_view_df.reset_index()
    diff_df = pnl_wide_view_df.diff(axis=1)
    period_options = pnl_wide_view_df.columns.tolist()
    row_std_diff = pnl_wide_view_df.diff(axis=1).std(axis=1, skipna=True).fillna(0)
except Exception as e: st.error(f"Error preparing P&L data: {e}"); st.exception(e); st.stop()

# --- Outlier Highlighting Function ---
# (Definition remains the same)
def highlight_outliers_pandas(row, diffs_df, thresholds_series):
    try: row_diff = diffs_df.loc[row.name]; threshold_value = thresholds_series.loc[row.name]
    except KeyError: return [''] * len(row)
    except Exception: return [''] * len(row)
    styles = [''] * len(row.index);
    for i, period in enumerate(row.index):
        if i == 0: continue
        diff_val = row_diff.get(period)
        if pd.notna(diff_val) and pd.notna(threshold_value) and threshold_value > 1e-6 and abs(diff_val) > threshold_value:
            if i < len(styles): styles[i] = f'background-color: {EY_YELLOW}; color: {EY_TEXT_ON_YELLOW};'
    return styles

# --- Sidebar Selection Widgets Logic ---
# (Code remains the same)
account_options = pnl_wide_view_df.index.get_level_values(PL_MAP_COLUMN).unique().tolist()
account_name_to_id_map = {name: id_ for id_, name in pnl_wide_view_df.index.unique()}
account_id_to_name_map = {id_: name for id_, name in pnl_wide_view_df.index.unique()}
current_account_name = account_id_to_name_map.get(st.session_state.selected_account_id, account_options[0] if account_options else None)
sorted_account_options = sorted(account_options)
account_index = get_index(sorted_account_options, current_account_name) if current_account_name else 0
period_index = get_index(period_options, st.session_state.selected_period) if st.session_state.selected_period else len(period_options) - 1
sb_selected_account_name = st.sidebar.selectbox("Select GL Account:", options=sorted_account_options, index=account_index, key="sb_account")
sb_selected_period = st.sidebar.selectbox("Select Period:", options=period_options, index=period_index, key="sb_period")

# --- Synchronize Sidebar selection ---
# (Code remains the same)
sidebar_account_id = account_name_to_id_map.get(sb_selected_account_name)
sidebar_period = sb_selected_period
sidebar_changed = False
if sidebar_account_id != st.session_state.selected_account_id or sidebar_period != st.session_state.selected_period:
    st.session_state.selected_account_id = sidebar_account_id; st.session_state.selected_account_name = sb_selected_account_name; st.session_state.selected_period = sidebar_period
    st.session_state.selected_je_col = None; st.session_state.selected_je_val = None
    sidebar_changed = True

# --- Fetch Journal Entries ---
# (Fetch logic remains the same)
should_fetch = False; table_changed_flag = False
if st.session_state.selected_account_id and st.session_state.selected_period:
     if (st.session_state.selected_account_id != st.session_state.prev_selected_account_id or
         st.session_state.selected_period != st.session_state.prev_selected_period or
         (st.session_state.related_jes_df.empty and (sidebar_changed))):
            should_fetch = True
if 'df_selection_processed' in locals() and df_selection_processed: table_changed_flag = True
if should_fetch and not table_changed_flag:
    try:
        st.session_state.related_jes_df = get_journal_entries(st.session_state.selected_account_id, st.session_state.selected_period, je_detail_df)
        st.session_state.prev_selected_account_id = st.session_state.selected_account_id; st.session_state.prev_selected_period = st.session_state.selected_period
        st.session_state.selected_je_col = None; st.session_state.selected_je_val = None
    except Exception as e_fetch: st.error(f"An error fetching JEs: {e_fetch}"); st.session_state.related_jes_df = pd.DataFrame()

# --- Define Tabs ---
tab1, tab2 = st.tabs([" P&L Analysis & Drilldown ", " Visualizations "])

# --- TAB 1: P&L ANALYSIS & DRILLDOWN ---
with tab1:
    # (Code for P&L Table, Table Selection, JE Details, Duplicate Finder remains the same as Response #39)
    # ... [P&L Table Display code] ...
    st.markdown(f"<h2 style='color: {EY_DARK_BLUE_GREY};'>Profit & Loss Overview (Monthly)</h2>", unsafe_allow_html=True)
    st.caption("Highlighting indicates MoM change > selected Std Deviations. Click row index + column header to select via table.")
    try:
        outlier_threshold_values_mi = threshold_std_dev * row_std_diff
        styled_df = pnl_wide_view_df.style.apply(highlight_outliers_pandas, axis=1, diffs_df=diff_df, thresholds_series=outlier_threshold_values_mi).format("{:,.0f}")
        st.dataframe(styled_df, use_container_width=True, key="pnl_select_df", on_select="rerun", selection_mode=("single-row", "single-column"))
    except Exception as e_style: st.error(f"Error displaying P&L table: {e_style}"); st.exception(e_style); st.dataframe(pnl_wide_view_df.style.format("{:,.0f}"), use_container_width=True)

    # Process Table Selection
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

    # Display Journal Entry Details
    st.markdown(f"<hr><h2 style='color: {EY_DARK_BLUE_GREY};'>Journal Entry Details</h2>", unsafe_allow_html=True)
    related_jes_to_display = st.session_state.related_jes_df
    if st.session_state.selected_account_id and st.session_state.selected_period:
        st.write(f"Showing JEs for: **{st.session_state.selected_account_name} ({st.session_state.selected_account_id})** | Period: **{st.session_state.selected_period}**")
        if isinstance(related_jes_to_display, pd.DataFrame) and not related_jes_to_display.empty:
            je_display_df = related_jes_to_display.copy(); je_amount_cols = [col for col in je_display_df.columns if 'Amount' in col or 'Debit' in col or 'Credit' in col]
            for col in je_amount_cols: je_display_df[col] = je_display_df[col].apply(format_amount_safely) # Use robust formatting
            je_col_config = {}; date_cols_to_format = [col for col in je_display_df.columns if 'Date' in col or JE_DATE_COLUMN in col]
            for col in date_cols_to_format: je_col_config[col] = st.column_config.DateColumn(format="YYYY-MM-DD", help="Transaction Date")
            st.dataframe(je_display_df, use_container_width=True, column_config=je_col_config)
        elif isinstance(related_jes_to_display, pd.DataFrame) and related_jes_to_display.empty: st.info(f"No Journal Entries found for this account and period.")
        else: st.warning("JE data is in an unexpected state.")
    else: st.info("Select an Account and Period to view Journal Entries.")

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

    # --- MODIFIED: Enhanced Multiselect for Chart ---
    chart_account_options = sorted(pl_flat_df[PL_MAP_COLUMN].unique())

    # Layout for multiselect and buttons
    c1, c2, c3 = st.columns([4, 1, 1]) # Adjust ratios as needed

    with c1:
        # Use session state for default, assign key
        # The return value `user_chart_selection` captures direct interaction
        user_chart_selection = st.multiselect(
            "Select Account(s) to Plot:",
            options=chart_account_options,
            default=st.session_state.chart_accounts_selection, # Controlled by state
            key="chart_multiselect_widget" # Assign key
        )
        # Synchronize direct widget interaction back to session state if it differs
        if user_chart_selection != st.session_state.chart_accounts_selection:
             st.session_state.chart_accounts_selection = user_chart_selection
             st.rerun() # Rerun if user changed selection directly

    with c2:
        st.markdown("<br>", unsafe_allow_html=True) # Add space for alignment
        if st.button("Select All", key='select_all_chart'):
            st.session_state.chart_accounts_selection = chart_account_options # Select all options
            st.rerun() # Rerun to update multiselect display

    with c3:
        st.markdown("<br>", unsafe_allow_html=True) # Add space for alignment
        if st.button("Clear Selection", key='clear_chart'):
            st.session_state.chart_accounts_selection = [] # Clear selection
            st.rerun() # Rerun to update multiselect display
    # --- END: Enhanced Multiselect ---


    # Plotting logic now uses st.session_state.chart_accounts_selection
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
        else: st.info("No data available for the selected accounts to plot.")
    else:
        st.info("Select one or more accounts using the controls above to display trend chart.")


    # --- JE Analysis Scatter Plot (Using ALL JEs) ---
    # (Code remains the same as Response #39)
    st.markdown(f"<hr><h2 style='color: {EY_DARK_BLUE_GREY};'>Total JE Analysis (Amount vs. Frequency)</h2>", unsafe_allow_html=True)
    st.caption("Analyzes ALL Journal Entries in the dataset by the selected category.")
    if 'je_detail_df' in globals() and isinstance(je_detail_df, pd.DataFrame) and not je_detail_df.empty:
        potential_analysis_cols = ['Customer', 'Memo', 'Account Name']
        analysis_cols_options = [col for col in potential_analysis_cols if col in je_detail_df.columns and (pd.api.types.is_string_dtype(je_detail_df[col]) or pd.api.types.is_object_dtype(je_detail_df[col])) and je_detail_df[col].nunique() > 1]
        if JE_AMOUNT_COLUMN not in je_detail_df.columns: st.warning(f"Amount column '{JE_AMOUNT_COLUMN}' not found."); analysis_cols_options = []
        if analysis_cols_options:
            selected_analysis_col_all = st.selectbox("Analyze ALL Journal Entries by:", options=analysis_cols_options, index=0, key="je_analysis_col_tab2_all")
            if selected_analysis_col_all:
                st.write(f"Aggregating all {len(je_detail_df)} journal entries by **{selected_analysis_col_all}**...")
                try:
                    with st.spinner(f"Aggregating by {selected_analysis_col_all}..."):
                         agg_data_all = je_detail_df.groupby(selected_analysis_col_all).agg(Count=('Transaction Id', 'size'), Total_Amount=(JE_AMOUNT_COLUMN, 'sum')).reset_index()
                         agg_data_all['Abs_Total_Amount'] = agg_data_all['Total_Amount'].abs() # Use abs amount for size
                         agg_data_all = agg_data_all.sort_values(by='Total_Amount', ascending=False)
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

    st.markdown(f"<hr><h2 style='color: {EY_DARK_BLUE_GREY};'>Other Visualizations</h2>", unsafe_allow_html=True)
    st.info("More charts can be added here.")