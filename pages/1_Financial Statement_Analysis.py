# pages/1_Financial Statement_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time

import utils # Includes ensure_settings_loaded
import data_processor # For get_journal_entries
from prompts import get_pnl_analysis_prompt 

# --- Helper Function Definitions --- (Keep these at the top)
def get_account_category_map(je_df, je_id_col, statement_categories_list):
    account_map = {}
    if je_df is not None and not je_df.empty and je_id_col in je_df.columns:
        temp_map_df = je_df[[je_id_col]].drop_duplicates().copy()
        temp_map_df['_StdAccID_'] = temp_map_df[je_id_col].astype(str).str.strip()
        use_custom = st.session_state.get('use_custom_ranges', False)
        parsed_ranges = st.session_state.get('parsed_gl_ranges', {})
        temp_map_df['Assigned_Category'] = temp_map_df['_StdAccID_'].apply(
            lambda x: utils.assign_category_using_rules(x, use_custom, parsed_ranges, utils.get_prefix_based_category)
        )
        temp_map_df = temp_map_df[temp_map_df['Assigned_Category'].isin(statement_categories_list)]
        for _, row in temp_map_df.iterrows():
            account_map[str(row[je_id_col]).strip()] = row['Assigned_Category']
    return account_map

def prepare_statement_flat_data_for_pivot(flat_df_source, id_col, map_col, type_col, 
                                     account_category_map, selected_categories_list, 
                                     valid_statement_categories_list):
    if flat_df_source is None or flat_df_source.empty: return pd.DataFrame(), {}
    df_processed = flat_df_source.copy()
    if type_col in df_processed.columns:
        df_processed = df_processed[df_processed[type_col] == "Individual Account"].copy()
    
    name_to_id_map = {}
    if account_category_map:
        df_processed['Assigned_Category'] = df_processed[id_col].astype(str).str.strip().map(account_category_map)
        df_processed.dropna(subset=['Assigned_Category'], inplace=True) 
        df_processed = df_processed[df_processed['Assigned_Category'].isin(valid_statement_categories_list)].copy()
    else: df_processed['Assigned_Category'] = "Unknown"

    if selected_categories_list and 'Assigned_Category' in df_processed.columns:
        df_processed = df_processed[df_processed['Assigned_Category'].isin(selected_categories_list)]
    
    if not df_processed.empty and id_col in df_processed.columns and map_col in df_processed.columns:
        for _, row in df_processed[[id_col, map_col]].drop_duplicates().iterrows():
            name_to_id_map[str(row[map_col])] = str(row[id_col]) 
            
    return df_processed, name_to_id_map

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Financial Statement Analysis")
st.markdown(f"<style> h1 {{ color: {utils.EY_DARK_BLUE_GREY}; }} </style>", unsafe_allow_html=True)
st.markdown(f"<h1>Financial Statement Analysis & Drilldown</h1>", unsafe_allow_html=True)

# --- Ensure LLM Settings are Loaded ---
utils.ensure_settings_loaded()

# --- Check if data has been uploaded and processed ---
if not st.session_state.get('uploaded_data_ready', False):
    st.error("Data has not been uploaded and processed. Please go to the '📄 Data Upload & Validation' page first.")
    st.stop()

# --- Initialize Unified Page-Specific Session State ---
if 'fsa_selected_account_id' not in st.session_state: st.session_state.fsa_selected_account_id = None
if 'fsa_selected_account_name' not in st.session_state: st.session_state.fsa_selected_account_name = None
if 'fsa_selected_period' not in st.session_state: st.session_state.fsa_selected_period = None
if 'fsa_related_jes_df' not in st.session_state: st.session_state.fsa_related_jes_df = pd.DataFrame()
if 'fsa_llm_analyses' not in st.session_state: st.session_state.fsa_llm_analyses = {} 
if 'fsa_llm_streaming_key' not in st.session_state: st.session_state.fsa_llm_streaming_key = None
if 'fsa_llm_prompt_for_display' not in st.session_state: st.session_state.fsa_llm_prompt_for_display = ""
if 'fsa_needs_je_refetch' not in st.session_state: st.session_state.fsa_needs_je_refetch = True
if 'fsa_outlier_threshold' not in st.session_state: st.session_state.fsa_outlier_threshold = 2.0
if 'fsa_dup_col' not in st.session_state: st.session_state.fsa_dup_col = None
if 'fsa_dup_val' not in st.session_state: st.session_state.fsa_dup_val = None
if 'fsa_dup_search_triggered' not in st.session_state: st.session_state.fsa_dup_search_triggered = False
if 'fsa_prev_selected_account_id' not in st.session_state: st.session_state.fsa_prev_selected_account_id = None
if 'fsa_prev_selected_period' not in st.session_state: st.session_state.fsa_prev_selected_period = None
if 'fsa_category_filter_selection' not in st.session_state: st.session_state.fsa_category_filter_selection = []
if 'fsa_show_highlighted_only' not in st.session_state: st.session_state.fsa_show_highlighted_only = False
if 'fsa_statement_type' not in st.session_state: st.session_state.fsa_statement_type = "P&L" # Default to P&L

if 'fsa_page_filter_je_amount_value' not in st.session_state: st.session_state.fsa_page_filter_je_amount_value = 0.0
if 'fsa_page_filter_je_amount_operator' not in st.session_state: st.session_state.fsa_page_filter_je_amount_operator = "Off"

try:
    pl_flat_df_source_global = st.session_state.get('active_pl_flat_df')
    bs_flat_df_source_global = st.session_state.get('active_bs_flat_df') 
    je_detail_df_global = st.session_state.get('compiled_je_df')
    column_mappings_global = st.session_state.get('column_mappings')

    if je_detail_df_global is None or je_detail_df_global.empty:
        st.error("JE data ('compiled_je_df') is not available. Please process data."); st.stop()
    if not column_mappings_global:
        st.error("Column mappings are not available. Please process data."); st.stop()
    if (pl_flat_df_source_global is None or pl_flat_df_source_global.empty) and \
       (bs_flat_df_source_global is None or bs_flat_df_source_global.empty):
        st.error("Neither P&L nor BS data is available. Please process data in 'Data Upload & Validation'."); st.stop()

except KeyError as e: st.error(f"Required data key missing: {e}. Ensure data is processed."); st.stop()
except AttributeError as e: st.error(f"Data not in expected format: {e}. Re-upload/process."); st.stop()

STMT_ID_COLUMN = 'Account ID'       
STMT_MAP_COLUMN = 'Account Name'    
STMT_TYPE_COLUMN = '_Account_Type_' 
STMT_PERIOD_COLUMN = 'Period'       
STMT_AMOUNT_COLUMN = 'Amount'       

JE_ID_COLUMN_GLOBAL = column_mappings_global.get("account_id")
JE_ACCOUNT_NAME_COL_GLOBAL = column_mappings_global.get("mapping")
JE_DATE_COLUMN_GLOBAL = column_mappings_global.get("date")
JE_AMOUNT_COLUMN_GLOBAL = column_mappings_global.get("amount")
ALL_JE_COLUMNS_FROM_SOURCE_GLOBAL = je_detail_df_global.columns.tolist()

st.sidebar.header("Page Filters & Controls")
sidebar_filter_changed = False 

st.sidebar.subheader("Statement Table Outlier Sensitivity")
current_outlier_thresh = st.session_state.get('fsa_outlier_threshold', 2.0)
new_outlier_thresh = st.sidebar.slider("Sensitivity:", 1.0, 5.0, current_outlier_thresh, 0.1, 
                                        key='fsa_outlier_slider', 
                                        help="Lower value = more sensitive to changes")
if not np.isclose(new_outlier_thresh, current_outlier_thresh):
    st.session_state.fsa_outlier_threshold = new_outlier_thresh
    sidebar_filter_changed = True

st.sidebar.subheader("Statement Table View Mode") 
current_show_highlighted = st.session_state.get('fsa_show_highlighted_only', False)
new_show_highlighted = st.sidebar.toggle("Show Only Highlighted Accounts", 
                                          value=current_show_highlighted, 
                                          key="fsa_view_mode_toggle",
                                          help="Filters the statement table for accounts meeting outlier criteria.")
if new_show_highlighted != current_show_highlighted:
    st.session_state.fsa_show_highlighted_only = new_show_highlighted
    sidebar_filter_changed = True

st.sidebar.markdown("---")
st.sidebar.subheader("JE Amount Filter")
st.sidebar.caption("Applies to JE drilldown and duplicate search.")
amount_ops = ["Off", "Greater than (>)", "Less than (<)", "Absolute value greater than (|x| >)"]
sel_op = st.sidebar.radio("Operator:", options=amount_ops, 
                          index=amount_ops.index(st.session_state.fsa_page_filter_je_amount_operator), 
                          key="fsa_je_op_radio")
amount_disabled = (sel_op == "Off")
new_je_val = st.sidebar.number_input("Amount Value:", 
                                   value=float(st.session_state.fsa_page_filter_je_amount_value or 0.0), 
                                   disabled=amount_disabled, step=100.0, format="%.2f", 
                                   key="fsa_je_val_input")

if sel_op != st.session_state.fsa_page_filter_je_amount_operator:
    st.session_state.fsa_page_filter_je_amount_operator = sel_op; sidebar_filter_changed = True
if not amount_disabled and new_je_val != st.session_state.fsa_page_filter_je_amount_value:
    st.session_state.fsa_page_filter_je_amount_value = new_je_val; sidebar_filter_changed = True

if sidebar_filter_changed: 
    st.session_state.fsa_needs_je_refetch = True; st.rerun()

# --- Statement Type Selection using Buttons ---
st.markdown("#### Select Statement Type for Analysis")
col_stmt_btn1, col_stmt_btn2 = st.columns(2)

active_statement_type = st.session_state.fsa_statement_type

with col_stmt_btn1:
    if st.button("Profit & Loss Analysis", 
                 type="primary" if active_statement_type == "P&L" else "secondary", 
                 use_container_width=True, key="fsa_select_pl_btn"):
        if active_statement_type != "P&L":
            st.session_state.fsa_statement_type = "P&L"
            st.session_state.fsa_selected_account_id = None; st.session_state.fsa_selected_account_name = None
            st.session_state.fsa_selected_period = None; st.session_state.fsa_related_jes_df = pd.DataFrame()
            st.session_state.fsa_category_filter_selection = [] 
            st.session_state.fsa_llm_prompt_for_display = ""; st.session_state.fsa_needs_je_refetch = True
            st.rerun()
with col_stmt_btn2:
    if st.button("Balance Sheet Analysis", 
                 type="primary" if active_statement_type == "Balance Sheet" else "secondary", 
                 use_container_width=True, key="fsa_select_bs_btn", 
                 disabled=(bs_flat_df_source_global is None or bs_flat_df_source_global.empty)): # Disable if no BS data
        if active_statement_type != "Balance Sheet":
            st.session_state.fsa_statement_type = "Balance Sheet"
            st.session_state.fsa_selected_account_id = None; st.session_state.fsa_selected_account_name = None
            st.session_state.fsa_selected_period = None; st.session_state.fsa_related_jes_df = pd.DataFrame()
            st.session_state.fsa_category_filter_selection = [] 
            st.session_state.fsa_llm_prompt_for_display = ""; st.session_state.fsa_needs_je_refetch = True
            st.rerun()

# --- Set active data and configurations based on selected_statement_type ---
active_flat_df, active_categories_list, active_statement_name = None, [], ""
if st.session_state.fsa_statement_type == "P&L":
    active_flat_df = pl_flat_df_source_global
    active_categories_list = utils.PNL_CATEGORIES_LIST
    active_statement_name = "Profit & Loss"
elif st.session_state.fsa_statement_type == "Balance Sheet":
    active_flat_df = bs_flat_df_source_global
    active_categories_list = utils.BS_CATEGORIES_LIST
    active_statement_name = "Balance Sheet"

if active_flat_df is None or active_flat_df.empty: # This check ensures data is available for chosen type
    st.info(f"{active_statement_name} data is not available. Select a different statement type or ensure data was processed in 'Data Upload & Validation'.")
    st.stop()

st.header(f"{active_statement_name} Drilldown") 

account_category_map = get_account_category_map(je_detail_df_global, JE_ID_COLUMN_GLOBAL, active_categories_list)

st.markdown(f"#### Filter {active_statement_name} Table by Category")
category_options = active_categories_list[:] 
if account_category_map:
    actual_cats_in_stmt = sorted([cat for cat in list(set(account_category_map.values())) if cat in active_categories_list])
    if actual_cats_in_stmt: category_options = actual_cats_in_stmt

if not st.session_state.fsa_category_filter_selection and category_options:
    st.session_state.fsa_category_filter_selection = category_options[:]
valid_current_cat_selection = [cat for cat in st.session_state.fsa_category_filter_selection if cat in category_options]
if not valid_current_cat_selection and category_options:
    valid_current_cat_selection = category_options[:]
st.session_state.fsa_category_filter_selection = valid_current_cat_selection

selected_stmt_categories = st.multiselect(
    f"Select {active_statement_name} Categories:", options=category_options,
    default=st.session_state.fsa_category_filter_selection,
    key="fsa_category_multiselect_widget" 
)
if selected_stmt_categories != st.session_state.fsa_category_filter_selection:
    st.session_state.fsa_category_filter_selection = selected_stmt_categories; st.rerun()

flat_df_for_pivot, name_to_id_map_for_dropdowns = prepare_statement_flat_data_for_pivot(
    active_flat_df, STMT_ID_COLUMN, STMT_MAP_COLUMN, STMT_TYPE_COLUMN,
    account_category_map, st.session_state.fsa_category_filter_selection, active_categories_list
)

wide_df, diff_df, row_std_diff, wide_reset = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.DataFrame()
if not flat_df_for_pivot.empty:
    try:
        wide_df = flat_df_for_pivot.pivot_table(index=[STMT_ID_COLUMN, STMT_MAP_COLUMN], columns=STMT_PERIOD_COLUMN, values=STMT_AMOUNT_COLUMN).fillna(0)
        if not wide_df.empty:
            wide_df = wide_df.sort_index(axis=1)
            diff_df = wide_df.diff(axis=1).fillna(0)
            row_std_diff = wide_df.diff(axis=1).std(axis=1, skipna=True).fillna(0)
        wide_reset = wide_df.reset_index()
    except Exception as e: st.error(f"Error pivoting {active_statement_name} data: {e}")

wide_df_display = wide_df.copy()
if st.session_state.fsa_show_highlighted_only: 
    highlighted_indices = []
    if not wide_df.empty and not diff_df.empty and not row_std_diff.empty:
        thresholds = st.session_state.fsa_outlier_threshold * row_std_diff
        for idx_tuple in wide_df.index:
            if idx_tuple in diff_df.index and idx_tuple in thresholds.index: 
                if pd.notna(thresholds.loc[idx_tuple]) and thresholds.loc[idx_tuple] > 1e-6:
                    if (diff_df.loc[idx_tuple].abs() > thresholds.loc[idx_tuple]).any():
                        highlighted_indices.append(idx_tuple)
    if highlighted_indices: wide_df_display = wide_df.loc[highlighted_indices]
    else: wide_df_display = pd.DataFrame(index=wide_df.index, columns=wide_df.columns).iloc[0:0]

st.markdown("---"); st.markdown(f"#### {active_statement_name} Data Overview")
st.caption(f"Highlighting: MoM change > {st.session_state.fsa_outlier_threshold:.1f} Std Dev. Click cell for JE/LLM.")

def handle_statement_table_click(): 
    raw_sel = st.session_state.get("fsa_select_df_key", {}).get('selection', {})
    rows, cols = raw_sel.get('rows', []), raw_sel.get('columns', [])
    if rows and cols:
        try:
            pos_idx = rows[0]
            if not wide_df_display.empty and pos_idx < len(wide_df_display): 
                acc_multi_idx = wide_df_display.index[pos_idx]
                acc_id, acc_name, period = str(acc_multi_idx[0]), str(acc_multi_idx[1]), cols[0]
                if acc_id != st.session_state.fsa_selected_account_id or \
                   acc_name != st.session_state.fsa_selected_account_name or \
                   period != st.session_state.fsa_selected_period:
                    st.session_state.fsa_selected_account_id = acc_id
                    st.session_state.fsa_selected_account_name = acc_name
                    st.session_state.fsa_selected_period = period
                    st.session_state.fsa_needs_je_refetch = True; st.rerun()
        except Exception as e: st.warning(f"{active_statement_name} table click error: {e}")

if not wide_df_display.empty:
    current_outlier_thresh_for_style = st.session_state.fsa_outlier_threshold
    styled_df = wide_df_display.style.apply(utils.highlight_outliers_pandas, axis=1, diffs_df=diff_df, thresholds_series=(current_outlier_thresh_for_style * row_std_diff), color=utils.EY_YELLOW, text_color=utils.EY_TEXT_ON_YELLOW).format("{:,.0f}")
    st.dataframe(styled_df, use_container_width=True, height=400, key="fsa_select_df_key", on_select="rerun", selection_mode=("single-row", "single-column"))
    if st.session_state.get("fsa_select_df_key"): handle_statement_table_click()
else: st.info(f"{active_statement_name} table empty based on current filters.")

st.markdown("---"); st.markdown(f"#### Select Account & Period for {active_statement_name} Drilldown")
st.caption("*(Populated from table above. Click table or use dropdowns.)*")
col1_sel, col2_sel = st.columns(2)
acc_opts_dd = sorted(wide_df_display.index.get_level_values(STMT_MAP_COLUMN).unique().tolist()) if not wide_df_display.empty else []
per_opts_dd = wide_df_display.columns.tolist() if not wide_df_display.empty else []

with col1_sel: 
    widget_acc_name = st.selectbox(f"GL Account:", acc_opts_dd, 
                                   index=utils.get_index(acc_opts_dd, st.session_state.fsa_selected_account_name), 
                                   key="fsa_widget_acc_select", disabled=not acc_opts_dd)
with col2_sel: 
    widget_per = st.selectbox(f"Period:", per_opts_dd, 
                              index=utils.get_index(per_opts_dd, st.session_state.fsa_selected_period), 
                              key="fsa_widget_per_select", disabled=not per_opts_dd)

widget_acc_id = name_to_id_map_for_dropdowns.get(widget_acc_name) if widget_acc_name else None
if (widget_acc_id != st.session_state.fsa_selected_account_id or \
    widget_per != st.session_state.fsa_selected_period or \
    widget_acc_name != st.session_state.fsa_selected_account_name):
    if widget_acc_id and widget_per and widget_acc_name:
        st.session_state.fsa_selected_account_id = widget_acc_id
        st.session_state.fsa_selected_account_name = widget_acc_name
        st.session_state.fsa_selected_period = widget_per
        st.session_state.fsa_needs_je_refetch = True; st.rerun()

if st.session_state.fsa_needs_je_refetch:
    sel_id, sel_per = st.session_state.fsa_selected_account_id, st.session_state.fsa_selected_period
    if sel_id and sel_per:
        try:
            fetched_jes = data_processor.get_journal_entries(sel_id, sel_per, je_detail_df_global, JE_ID_COLUMN_GLOBAL, JE_DATE_COLUMN_GLOBAL, ALL_JE_COLUMNS_FROM_SOURCE_GLOBAL)
            temp_df = fetched_jes.copy() if isinstance(fetched_jes, pd.DataFrame) else pd.DataFrame()
            if JE_AMOUNT_COLUMN_GLOBAL in temp_df.columns and st.session_state.fsa_page_filter_je_amount_operator != "Off":
                op_je, val_je = st.session_state.fsa_page_filter_je_amount_operator, float(st.session_state.fsa_page_filter_je_amount_value)
                if not pd.api.types.is_numeric_dtype(temp_df[JE_AMOUNT_COLUMN_GLOBAL]):
                    temp_df[JE_AMOUNT_COLUMN_GLOBAL] = pd.to_numeric(temp_df[JE_AMOUNT_COLUMN_GLOBAL], errors='coerce')
                temp_df.dropna(subset=[JE_AMOUNT_COLUMN_GLOBAL], inplace=True)
                if op_je == "Greater than (>)": temp_df = temp_df[temp_df[JE_AMOUNT_COLUMN_GLOBAL] > val_je]
                elif op_je == "Less than (<)": temp_df = temp_df[temp_df[JE_AMOUNT_COLUMN_GLOBAL] < val_je]
                elif op_je == "Absolute value greater than (|x| >)": temp_df = temp_df[temp_df[JE_AMOUNT_COLUMN_GLOBAL].abs() > val_je]
            st.session_state.fsa_related_jes_df = temp_df
            st.session_state.fsa_prev_selected_account_id, st.session_state.fsa_prev_selected_period = sel_id, sel_per
        except Exception as e: st.error(f"Error fetching JEs: {e}"); st.session_state.fsa_related_jes_df = pd.DataFrame()
        finally: st.session_state.fsa_needs_je_refetch = False

st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>Journal Entry Details for Selected {active_statement_name} Account</h2>", unsafe_allow_html=True)
related_jes_to_display = st.session_state.get('fsa_related_jes_df', pd.DataFrame())
sel_id_disp, sel_name_disp, sel_per_disp = st.session_state.fsa_selected_account_id, st.session_state.fsa_selected_account_name, st.session_state.fsa_selected_period

token_str, est_tokens = "", 0
if isinstance(related_jes_to_display, pd.DataFrame) and not related_jes_to_display.empty:
    try:
        fmt_je_str = utils.format_je_for_llm(related_jes_to_display, JE_AMOUNT_COLUMN_GLOBAL)
        est_tokens = utils.estimate_token_count(fmt_je_str)
        token_str = f"(Est. Tokens for LLM: {est_tokens:,})"
    except: token_str = "(Token est. error)"
elif isinstance(related_jes_to_display, pd.DataFrame): token_str = "(Est. Tokens for LLM: 0)"

if sel_id_disp and sel_per_disp:
    st.write(f"Showing JEs for {active_statement_name} Account: **{sel_name_disp} ({sel_id_disp})** | Period: **{sel_per_disp}** {token_str}")
    st.caption(f"JE table filtered by Account/Period, and by page-level JE Amount filter: '{st.session_state.fsa_page_filter_je_amount_operator}' '{st.session_state.fsa_page_filter_je_amount_value if st.session_state.fsa_page_filter_je_amount_operator != 'Off' else ''}'.")
    if isinstance(related_jes_to_display, pd.DataFrame) and not related_jes_to_display.empty:
        disp_df = related_jes_to_display.copy()
        if JE_AMOUNT_COLUMN_GLOBAL in disp_df.columns and pd.api.types.is_numeric_dtype(disp_df[JE_AMOUNT_COLUMN_GLOBAL]):
             disp_df[JE_AMOUNT_COLUMN_GLOBAL] = disp_df[JE_AMOUNT_COLUMN_GLOBAL].apply(utils.format_amount_safely)
        col_cfg = {}
        if JE_DATE_COLUMN_GLOBAL in disp_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(disp_df[JE_DATE_COLUMN_GLOBAL]):
                disp_df[JE_DATE_COLUMN_GLOBAL] = pd.to_datetime(disp_df[JE_DATE_COLUMN_GLOBAL], errors='coerce')
            col_cfg[JE_DATE_COLUMN_GLOBAL] = st.column_config.DateColumn(label=JE_DATE_COLUMN_GLOBAL, format="YYYY-MM-DD")
        st.dataframe(disp_df, use_container_width=True, column_config=col_cfg, hide_index=True)
    elif isinstance(related_jes_to_display, pd.DataFrame) and related_jes_to_display.empty:
        if (sel_id_disp == st.session_state.fsa_prev_selected_account_id and sel_per_disp == st.session_state.fsa_prev_selected_period):
            st.info(f"No JEs found for {sel_name_disp} in {sel_per_disp} matching current JE amount filters.")
        else: st.info(f"Awaiting JE data or none available for this {active_statement_name} selection and JE filters.")
else: st.info(f"Select {active_statement_name} Account/Period to view Journal Entries.")

if st.session_state.fsa_llm_prompt_for_display:
    with st.expander(f"View Last Generated LLM Prompt For {active_statement_name} Analysis", expanded=False):
        st.text_area("LLM Prompt:", value=st.session_state.fsa_llm_prompt_for_display, height=300, disabled=True, key="fsa_llm_prompt_display_area")

st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>LLM Period Analysis for {active_statement_name}</h2>", unsafe_allow_html=True)
sel_prov = st.session_state.get('llm_provider')
model_id, llm_ok, llm_cfg = None, False, {}
if sel_prov == "Ollama":
    model_id = st.session_state.get('chosen_ollama_model')
    if model_id: llm_ok = True; llm_cfg['model_name'] = model_id
elif sel_prov == "OpenAI":
    model_id, api_key = st.session_state.get('openai_model_name'), st.session_state.get('openai_api_key')
    if model_id and api_key: llm_ok = True; llm_cfg.update({'model_name': model_id, 'api_key': api_key})
elif sel_prov == "Azure OpenAI":
    cfg_keys = ['azure_deployment_name', 'azure_endpoint', 'azure_api_key', 'azure_api_version']
    cfg_vals = {k: st.session_state.get(k) for k in cfg_keys}
    if all(cfg_vals.values()):
        model_id, llm_ok = cfg_vals['azure_deployment_name'], True
        llm_cfg['deployment_name'] = cfg_vals['azure_deployment_name']
        llm_cfg['endpoint'] = cfg_vals['azure_endpoint']
        llm_cfg['api_key'] = cfg_vals['azure_api_key']
        llm_cfg['api_version'] = cfg_vals['azure_api_version']

if not llm_ok: st.warning(f"LLM provider '{sel_prov}' not configured. LLM analysis disabled.")
token_limit = st.session_state.get('llm_context_limit', utils.DEFAULT_SETTINGS['llm_context_limit'])
je_data_ok = isinstance(related_jes_to_display, pd.DataFrame) and not related_jes_to_display.empty
can_analyze = (sel_id_disp and sel_name_disp and sel_per_disp and llm_ok and je_data_ok)

if can_analyze and est_tokens + 250 > token_limit :
    st.warning(f"⚠️ JE data for LLM (~{est_tokens:,} tokens) + overhead may exceed context limit ({token_limit:,} tokens).", icon="⚠️")

analysis_key, analysis_pressed = None, False
if can_analyze:
    analysis_key = (sel_id_disp, sel_name_disp, sel_per_disp, st.session_state.fsa_statement_type) 
    btn_label = f"🤖 Analyze {active_statement_name} Period ({sel_prov}: {model_id or 'Default'})"
    is_streaming = (st.session_state.fsa_llm_streaming_key == analysis_key)
    if analysis_key in st.session_state.fsa_llm_analyses and not is_streaming: btn_label = f"🔄 Re-analyze {active_statement_name} Period"
    if st.button(btn_label, key=f"fsa_llm_btn_{analysis_key}", disabled=is_streaming): 
        analysis_pressed, st.session_state.fsa_llm_streaming_key = True, analysis_key
elif sel_id_disp and sel_per_disp:
    if not llm_ok: pass
    elif not je_data_ok: st.warning(f"No JE data for this {active_statement_name} selection to send to LLM.")
else: st.info(f"Select {active_statement_name} Account/Period with JEs to enable LLM analysis.")

current_streaming_key, output_placeholder = st.session_state.fsa_llm_streaming_key, st.empty()
should_stream_now = current_streaming_key and current_streaming_key == (sel_id_disp, sel_name_disp, sel_per_disp, st.session_state.fsa_statement_type)

if should_stream_now or (analysis_pressed and can_analyze):
    if not llm_ok: output_placeholder.error("LLM config invalid."); st.session_state.fsa_llm_streaming_key = None
    else:
        try:
            with output_placeholder.container(): st.info(f"🤖 Contacting {sel_prov} ({model_id or 'Default'})...")
            acc_id, acc_name, curr_per = sel_id_disp, sel_name_disp, sel_per_disp
            curr_amt = wide_df.loc[(acc_id, acc_name), curr_per] if not wide_df.empty and (acc_id, acc_name) in wide_df.index and curr_per in wide_df.columns else None
            periods_ctx = wide_df.columns.tolist() if not wide_df.empty else []
            prev_per, prev_amt = (periods_ctx[periods_ctx.index(curr_per)-1] if periods_ctx and curr_per in periods_ctx and periods_ctx.index(curr_per)>0 else None), None
            if prev_per and not wide_df.empty and (acc_id, acc_name) in wide_df.index and prev_per in wide_df.columns: prev_amt = wide_df.loc[(acc_id, acc_name), prev_per]
            is_outlier_ctx = ""
            if not diff_df.empty and not row_std_diff.empty and (acc_id, acc_name) in diff_df.index and curr_per in diff_df.columns and (acc_id, acc_name) in row_std_diff.index:
                diff_v, std_v, thresh_m = diff_df.loc[(acc_id, acc_name), curr_per], row_std_diff.loc[(acc_id, acc_name)], st.session_state.fsa_outlier_threshold
                if pd.notna(diff_v) and pd.notna(std_v) and np.isfinite(diff_v) and std_v > 1e-6 and abs(diff_v) > (thresh_m * std_v): is_outlier_ctx = f"\nNote: {active_statement_name} value met outlier criteria (MoM change > {thresh_m:.1f} Std Dev)."
            
            prompt_text = get_pnl_analysis_prompt(acc_name, acc_id, curr_per, utils.format_amount_safely(curr_amt), prev_per, utils.format_amount_safely(prev_amt), utils.format_je_for_llm(st.session_state.fsa_related_jes_df, JE_AMOUNT_COLUMN_GLOBAL), is_outlier_ctx)
            if st.session_state.fsa_statement_type == "Balance Sheet":
                prompt_text = prompt_text.replace("P&L account activity", "Balance Sheet account activity")
            st.session_state.fsa_llm_prompt_for_display = prompt_text
            
            final_resp = ""
            with output_placeholder.container():
                st.markdown("---")
                try: final_resp = st.write_stream(utils.call_llm_stream(sel_prov, llm_cfg, st.session_state.fsa_llm_prompt_for_display)) or "*(No content)*"
                except Exception as stream_err: st.error(f"LLM Stream Error: {stream_err}"); final_resp = f"**Stream Error:** {str(stream_err)}"
            st.session_state.fsa_llm_analyses[current_streaming_key] = final_resp
        except Exception as e_llm_prep: st.error(f"LLM Prep Error: {e_llm_prep}"); st.session_state.fsa_llm_analyses[current_streaming_key] = f"**Prep Error:** {str(e_llm_prep)}"
        finally: st.session_state.fsa_llm_streaming_key = None; st.rerun()
elif analysis_key and analysis_key in st.session_state.fsa_llm_analyses and not should_stream_now:
    with output_placeholder.container(): st.markdown("---"); st.markdown(st.session_state.fsa_llm_analyses[analysis_key])

st.sidebar.markdown("---"); st.sidebar.header(f"LLM Analysis History (Current Page)")
if st.sidebar.button("Clear LLM History for this Page", key='fsa_clear_llm_hist_btn'): 
    st.session_state.fsa_llm_analyses = {}; st.session_state.fsa_llm_streaming_key = None
    st.session_state.fsa_llm_prompt_for_display = "" 
    st.rerun()
if st.session_state.fsa_llm_analyses:
    hist_keys = list(st.session_state.fsa_llm_analyses.keys())
    st.sidebar.caption(f"Showing latest {min(len(hist_keys), 10)} analyses:")
    for k_hist in reversed(hist_keys[-10:]):
        try: 
            with st.sidebar.expander(f"{k_hist[1]} ({k_hist[3]}) - {k_hist[2]}"): st.markdown(st.session_state.fsa_llm_analyses[k_hist] or "_Initiated..._")
        except: st.sidebar.warning(f"Err display hist {k_hist}")
else: st.sidebar.caption("No LLM analyses run yet on this page.")

st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>Duplicate Value Lookup</h2>", unsafe_allow_html=True)
st.caption("Searches JE data. Filtered by page-level JE Amount filter (if active).")
current_related_jes = st.session_state.get('fsa_related_jes_df', pd.DataFrame())

je_detail_df_for_dup_search = je_detail_df_global.copy()
if JE_AMOUNT_COLUMN_GLOBAL in je_detail_df_for_dup_search.columns and st.session_state.fsa_page_filter_je_amount_operator != "Off":
    op_dup_main, val_dup_main = st.session_state.fsa_page_filter_je_amount_operator, float(st.session_state.fsa_page_filter_je_amount_value)
    je_detail_df_for_dup_search[JE_AMOUNT_COLUMN_GLOBAL] = pd.to_numeric(je_detail_df_for_dup_search[JE_AMOUNT_COLUMN_GLOBAL], errors='coerce')
    je_detail_df_for_dup_search.dropna(subset=[JE_AMOUNT_COLUMN_GLOBAL], inplace=True)
    if op_dup_main == "Greater than (>)": dup_search_base_df = je_detail_df_for_dup_search[je_detail_df_for_dup_search[JE_AMOUNT_COLUMN_GLOBAL] > val_dup_main]
    elif op_dup_main == "Less than (<)": dup_search_base_df = je_detail_df_for_dup_search[je_detail_df_for_dup_search[JE_AMOUNT_COLUMN_GLOBAL] < val_dup_main]
    elif op_dup_main == "Absolute value greater than (|x| >)": dup_search_base_df = je_detail_df_for_dup_search[je_detail_df_for_dup_search[JE_AMOUNT_COLUMN_GLOBAL].abs() > val_dup_main]
else: # If filter is off, use the full je_detail_df_for_dup_search
    dup_search_base_df = je_detail_df_for_dup_search


source_for_dup_options_ui = current_related_jes if not current_related_jes.empty else dup_search_base_df

available_dup_cols = []
if not source_for_dup_options_ui.empty:
    priority_cols = [JE_AMOUNT_COLUMN_GLOBAL, 'Memo', 'Customer', 'Transaction Id', 'Description', 'Line Description']
    priority_cols = [col for col in priority_cols if col in source_for_dup_options_ui.columns and source_for_dup_options_ui[col].nunique(dropna=True) > 0]
    other_cols = [col for col in source_for_dup_options_ui.columns if col not in priority_cols and source_for_dup_options_ui[col].nunique(dropna=True) > 0 and col not in [JE_ID_COLUMN_GLOBAL, JE_ACCOUNT_NAME_COL_GLOBAL, JE_DATE_COLUMN_GLOBAL]]
    available_dup_cols = sorted(list(set(priority_cols + other_cols)))
elif not dup_search_base_df.empty :
     available_dup_cols = [col for col in dup_search_base_df.columns if dup_search_base_df[col].nunique(dropna=True) > 0 and col not in [JE_ID_COLUMN_GLOBAL, JE_ACCOUNT_NAME_COL_GLOBAL, JE_DATE_COLUMN_GLOBAL]][:10]
     available_dup_cols = sorted(list(set(available_dup_cols)))

if available_dup_cols:
    col1_d, col2_d = st.columns(2)
    with col1_d: sel_dup_col = st.selectbox("Check JE Column:", available_dup_cols, index=utils.get_index(available_dup_cols, st.session_state.fsa_dup_col), key='fsa_dup_col_select')
    val_opts, last_dup_val, dup_val_idx = [], st.session_state.fsa_dup_val, 0
    if sel_dup_col and sel_dup_col in source_for_dup_options_ui.columns:
        try: val_opts = sorted(list(source_for_dup_options_ui[sel_dup_col].dropna().astype(str).unique()))
        except: val_opts = list(source_for_dup_options_ui[sel_dup_col].dropna().astype(str).unique())
    if val_opts and last_dup_val is not None:
        try: dup_val_idx = val_opts.index(str(last_dup_val))
        except ValueError: dup_val_idx = 0
    with col2_d: sel_dup_val = st.selectbox(f"Value from JEs (current context if available):", options=val_opts, index=dup_val_idx, key='fsa_dup_val_select', disabled=(not val_opts))
    if st.button("Find All Occurrences", key='fsa_find_dup_btn') and sel_dup_col and sel_dup_val is not None:
        st.session_state.fsa_dup_col, st.session_state.fsa_dup_val, st.session_state.fsa_dup_search_triggered = sel_dup_col, sel_dup_val, True; st.rerun()

    if st.session_state.fsa_dup_search_triggered:
        col_c, val_f_str = st.session_state.fsa_dup_col, str(st.session_state.fsa_dup_val)
        st.write(f"Finding JEs where **{col_c}** is **'{val_f_str}'** (searches across JEs filtered by sidebar JE Amount filter)...")
        if col_c and val_f_str is not None and col_c in dup_search_base_df.columns:
            try:
                mask = dup_search_base_df[col_c].astype(str).fillna("").str.strip() == val_f_str.strip()
                duplicate_jes = dup_search_base_df[mask]
                if not duplicate_jes.empty:
                    st.write(f"Found {len(duplicate_jes)} matching entries:")
                    disp_df_d = duplicate_jes.copy()
                    if JE_AMOUNT_COLUMN_GLOBAL in disp_df_d.columns and pd.api.types.is_numeric_dtype(disp_df_d[JE_AMOUNT_COLUMN_GLOBAL]):
                         disp_df_d[JE_AMOUNT_COLUMN_GLOBAL] = disp_df_d[JE_AMOUNT_COLUMN_GLOBAL].apply(utils.format_amount_safely)
                    cfg_d = {JE_DATE_COLUMN_GLOBAL: st.column_config.DateColumn(format="YYYY-MM-DD")} if JE_DATE_COLUMN_GLOBAL in disp_df_d.columns else {}
                    st.dataframe(disp_df_d, use_container_width=True, column_config=cfg_d, hide_index=True)
                else: st.info(f"No JEs found where '{col_c}' is '{val_f_str}'.")
            except Exception as e_d: st.error(f"Dup search error: {e_d}")
        elif not col_c or col_c not in dup_search_base_df.columns:
             st.error(f"Column '{col_c}' not available for searching.")
        st.session_state.fsa_dup_search_triggered = False
elif je_detail_df_global.empty: st.info("No JE data for duplicate search.")
else: st.info("No suitable columns/JE data after filters for duplicate search options.")