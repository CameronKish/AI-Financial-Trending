# pages/1_P&L_Analysis_&_Drilldown.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time

import utils # Includes ensure_settings_loaded
import data_processor # For get_journal_entries
from prompts import get_pnl_analysis_prompt # Reused for BS for now

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

# --- Initialize Page-Specific Session State ---
# For P&L tab
if 'pnl_selected_account_id' not in st.session_state: st.session_state.pnl_selected_account_id = None
if 'pnl_selected_account_name' not in st.session_state: st.session_state.pnl_selected_account_name = None
if 'pnl_selected_period' not in st.session_state: st.session_state.pnl_selected_period = None
if 'pnl_related_jes_df' not in st.session_state: st.session_state.pnl_related_jes_df = pd.DataFrame()
if 'pnl_llm_analyses' not in st.session_state: st.session_state.pnl_llm_analyses = {}
if 'pnl_llm_streaming_key' not in st.session_state: st.session_state.pnl_llm_streaming_key = None
if 'pnl_llm_prompt_for_display' not in st.session_state: st.session_state.pnl_llm_prompt_for_display = ""
if 'pnl_needs_je_refetch' not in st.session_state: st.session_state.pnl_needs_je_refetch = True
if 'pnl_outlier_threshold' not in st.session_state: st.session_state.pnl_outlier_threshold = 2.0
if 'pnl_dup_col' not in st.session_state: st.session_state.pnl_dup_col = None
if 'pnl_dup_val' not in st.session_state: st.session_state.pnl_dup_val = None
if 'pnl_dup_search_triggered' not in st.session_state: st.session_state.pnl_dup_search_triggered = False
if 'pnl_prev_selected_account_id' not in st.session_state: st.session_state.pnl_prev_selected_account_id = None
if 'pnl_prev_selected_period' not in st.session_state: st.session_state.pnl_prev_selected_period = None
if 'pnl_category_filter_selection' not in st.session_state: st.session_state.pnl_category_filter_selection = []
if 'pnl_show_highlighted_only' not in st.session_state: st.session_state.pnl_show_highlighted_only = False

# For BS tab
if 'bs_selected_account_id' not in st.session_state: st.session_state.bs_selected_account_id = None
if 'bs_selected_account_name' not in st.session_state: st.session_state.bs_selected_account_name = None
if 'bs_selected_period' not in st.session_state: st.session_state.bs_selected_period = None
if 'bs_related_jes_df' not in st.session_state: st.session_state.bs_related_jes_df = pd.DataFrame()
if 'bs_llm_analyses' not in st.session_state: st.session_state.bs_llm_analyses = {}
if 'bs_llm_streaming_key' not in st.session_state: st.session_state.bs_llm_streaming_key = None
if 'bs_llm_prompt_for_display' not in st.session_state: st.session_state.bs_llm_prompt_for_display = ""
if 'bs_needs_je_refetch' not in st.session_state: st.session_state.bs_needs_je_refetch = True
if 'bs_outlier_threshold' not in st.session_state: st.session_state.bs_outlier_threshold = 2.0
if 'bs_dup_col' not in st.session_state: st.session_state.bs_dup_col = None
if 'bs_dup_val' not in st.session_state: st.session_state.bs_dup_val = None
if 'bs_dup_search_triggered' not in st.session_state: st.session_state.bs_dup_search_triggered = False
if 'bs_prev_selected_account_id' not in st.session_state: st.session_state.bs_prev_selected_account_id = None
if 'bs_prev_selected_period' not in st.session_state: st.session_state.bs_prev_selected_period = None
if 'bs_category_filter_selection' not in st.session_state: st.session_state.bs_category_filter_selection = []
if 'bs_show_highlighted_only' not in st.session_state: st.session_state.bs_show_highlighted_only = False


# Sidebar Filters state
if 'fsa_filter_je_amount_value' not in st.session_state: st.session_state.fsa_filter_je_amount_value = 0.0
if 'fsa_filter_je_amount_operator' not in st.session_state: st.session_state.fsa_filter_je_amount_operator = "Off"

try:
    pl_flat_df_source_global = st.session_state.get('active_pl_flat_df')
    bs_flat_df_source_global = st.session_state.get('active_bs_flat_df') 
    je_detail_df_global = st.session_state.get('compiled_je_df')
    column_mappings_global = st.session_state.get('column_mappings')

    if je_detail_df_global is None or je_detail_df_global.empty:
        st.error("JE data ('compiled_je_df') is not available. Please process data."); st.stop()
    if not column_mappings_global:
        st.error("Column mappings are not available. Please process data."); st.stop()

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
filter_changed_in_sidebar = False

st.sidebar.subheader("Statement Table Outlier Sensitivity")
common_outlier_threshold = st.session_state.get('pnl_outlier_threshold', 2.0) 
slider_visual_value = st.sidebar.slider("Sensitivity:", 1.0, 5.0, common_outlier_threshold, 0.1, 
                                        key='fsa_outlier_threshold_widget', 
                                        help="Lower value = more sensitive to changes")
if not np.isclose(slider_visual_value, common_outlier_threshold):
    st.session_state.pnl_outlier_threshold = slider_visual_value
    st.session_state.bs_outlier_threshold = slider_visual_value 
    filter_changed_in_sidebar = True

st.sidebar.subheader("P&L Table View Mode") 
show_highlighted_pnl_sidebar = st.sidebar.toggle("Show Only Highlighted P&L Accounts", 
                                          value=st.session_state.pnl_show_highlighted_only, 
                                          key="sidebar_pnl_view_mode_toggle",
                                          help="Filters the P&L table for accounts meeting outlier criteria.")
if show_highlighted_pnl_sidebar != st.session_state.pnl_show_highlighted_only:
    st.session_state.pnl_show_highlighted_only = show_highlighted_pnl_sidebar; filter_changed_in_sidebar = True

st.sidebar.subheader("BS Table View Mode") 
show_highlighted_bs_sidebar = st.sidebar.toggle("Show Only Highlighted BS Accounts", 
                                          value=st.session_state.bs_show_highlighted_only, 
                                          key="sidebar_bs_view_mode_toggle",
                                          help="Filters the BS table for accounts meeting outlier criteria.")
if show_highlighted_bs_sidebar != st.session_state.bs_show_highlighted_only:
    st.session_state.bs_show_highlighted_only = show_highlighted_bs_sidebar; filter_changed_in_sidebar = True


st.sidebar.markdown("---")
st.sidebar.subheader("JE Amount Filter")
st.sidebar.caption("Applies to JE drilldown and duplicate search across all statements.")
amount_operator_options = ["Off", "Greater than (>)", "Less than (<)", "Absolute value greater than (|x| >)"]
selected_amount_operator = st.sidebar.radio("Operator:", options=amount_operator_options, index=amount_operator_options.index(st.session_state.fsa_filter_je_amount_operator), key="fsa_je_amount_op_radio")
amount_value_disabled = (selected_amount_operator == "Off")
new_amount_value = st.sidebar.number_input("Amount Value:", value=float(st.session_state.fsa_filter_je_amount_value or 0.0), disabled=amount_value_disabled, step=100.0, format="%.2f", key="fsa_je_amount_val_input")

if selected_amount_operator != st.session_state.fsa_filter_je_amount_operator:
    st.session_state.fsa_filter_je_amount_operator = selected_amount_operator; filter_changed_in_sidebar = True
if not amount_value_disabled and new_amount_value != st.session_state.fsa_filter_je_amount_value:
    st.session_state.fsa_filter_je_amount_value = new_amount_value; filter_changed_in_sidebar = True

if filter_changed_in_sidebar: 
    st.session_state.pnl_needs_je_refetch = True 
    st.session_state.bs_needs_je_refetch = True 
    st.rerun()

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

# --- Main Tab Content ---
tab_pnl, tab_bs = st.tabs(["P&L Analysis", "BS Analysis"])

# --- P&L Tab ---
with tab_pnl:
    prefix = "pnl"
    st.header("Profit & Loss Drilldown")

    st.markdown("#### Filter P&L Table by Category")
    pnl_account_cat_map = get_account_category_map(je_detail_df_global, JE_ID_COLUMN_GLOBAL, utils.PNL_CATEGORIES_LIST)
    pnl_cat_options = utils.PNL_CATEGORIES_LIST
    if pnl_account_cat_map:
        actual_cats = sorted(list(set(pnl_account_cat_map.values())))
        if actual_cats : pnl_cat_options = actual_cats
    if not st.session_state[f'{prefix}_category_filter_selection'] and pnl_cat_options:
        st.session_state[f'{prefix}_category_filter_selection'] = pnl_cat_options[:]
    
    sel_pnl_cats = st.multiselect("Select P&L Categories:", pnl_cat_options, default=st.session_state[f'{prefix}_category_filter_selection'], key=f"{prefix}_cat_multi")
    if sel_pnl_cats != st.session_state[f'{prefix}_category_filter_selection']:
        st.session_state[f'{prefix}_category_filter_selection'] = sel_pnl_cats; st.rerun()

    pl_flat_for_pivot, pnl_name_to_id_map_dd = prepare_statement_flat_data_for_pivot(
        pl_flat_df_source_global, STMT_ID_COLUMN, STMT_MAP_COLUMN, STMT_TYPE_COLUMN,
        pnl_account_cat_map, st.session_state[f'{prefix}_category_filter_selection'], utils.PNL_CATEGORIES_LIST
    )

    pnl_wide_df, pnl_diff_df, pnl_row_std_diff, pnl_wide_reset = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.DataFrame()
    if not pl_flat_for_pivot.empty:
        try:
            pnl_wide_df = pl_flat_for_pivot.pivot_table(index=[STMT_ID_COLUMN, STMT_MAP_COLUMN], columns=STMT_PERIOD_COLUMN, values=STMT_AMOUNT_COLUMN).fillna(0)
            if not pnl_wide_df.empty:
                pnl_wide_df = pnl_wide_df.sort_index(axis=1)
                pnl_diff_df = pnl_wide_df.diff(axis=1).fillna(0)
                pnl_row_std_diff = pnl_wide_df.diff(axis=1).std(axis=1, skipna=True).fillna(0)
            pnl_wide_reset = pnl_wide_df.reset_index()
        except Exception as e: st.error(f"Error pivoting P&L data: {e}")
    
    pnl_wide_df_display = pnl_wide_df.copy()
    if st.session_state.pnl_show_highlighted_only: 
        highlighted_indices = []
        if not pnl_wide_df.empty and not pnl_diff_df.empty and not pnl_row_std_diff.empty:
            thresholds = st.session_state.pnl_outlier_threshold * pnl_row_std_diff
            for idx_tuple in pnl_wide_df.index:
                if idx_tuple in pnl_diff_df.index and idx_tuple in thresholds.index: # Check if index exists in series
                    if pd.notna(thresholds.loc[idx_tuple]) and thresholds.loc[idx_tuple] > 1e-6: # Check if threshold is valid
                        if (pnl_diff_df.loc[idx_tuple].abs() > thresholds.loc[idx_tuple]).any():
                            highlighted_indices.append(idx_tuple)
        if highlighted_indices: pnl_wide_df_display = pnl_wide_df.loc[highlighted_indices]
        else: pnl_wide_df_display = pd.DataFrame(index=pnl_wide_df.index, columns=pnl_wide_df.columns).iloc[0:0]
    
    st.markdown("---"); st.markdown("#### P&L Data Overview")
    st.caption(f"Highlighting: MoM change > {st.session_state.pnl_outlier_threshold:.1f} Std Dev. Click cell for JE/LLM.")

    def handle_pnl_table_click_for_tab(): 
        raw_sel = st.session_state.get(f"{prefix}_select_df_key", {}).get('selection', {})
        rows, cols = raw_sel.get('rows', []), raw_sel.get('columns', [])
        if rows and cols:
            try:
                pos_idx = rows[0]
                if not pnl_wide_df_display.empty and pos_idx < len(pnl_wide_df_display): 
                    acc_multi_idx = pnl_wide_df_display.index[pos_idx]
                    acc_id, acc_name, period = str(acc_multi_idx[0]), str(acc_multi_idx[1]), cols[0]
                    if acc_id != st.session_state[f'{prefix}_selected_account_id'] or \
                       acc_name != st.session_state[f'{prefix}_selected_account_name'] or \
                       period != st.session_state[f'{prefix}_selected_period']:
                        st.session_state[f'{prefix}_selected_account_id'] = acc_id
                        st.session_state[f'{prefix}_selected_account_name'] = acc_name
                        st.session_state[f'{prefix}_selected_period'] = period
                        st.session_state[f'{prefix}_needs_je_refetch'] = True; st.rerun()
            except Exception as e: st.warning(f"P&L table click error: {e}")

    if not pnl_wide_df_display.empty:
        styled_df = pnl_wide_df_display.style.apply(utils.highlight_outliers_pandas, axis=1, diffs_df=pnl_diff_df, thresholds_series=(st.session_state.pnl_outlier_threshold * pnl_row_std_diff), color=utils.EY_YELLOW, text_color=utils.EY_TEXT_ON_YELLOW).format("{:,.0f}")
        st.dataframe(styled_df, use_container_width=True, height=400, key=f"{prefix}_select_df_key", on_select="rerun", selection_mode=("single-row", "single-column"))
        if st.session_state.get(f"{prefix}_select_df_key"): handle_pnl_table_click_for_tab()
    else: st.info("P&L table empty based on current filters (Category or Highlighted Only).")

    st.markdown("---"); st.markdown("#### Select Account & Period for P&L Drilldown")
    st.caption("*(Populated from P&L table. Click table or use dropdowns.)*")
    col1_pnl_sel, col2_pnl_sel = st.columns(2)
    acc_opts_dd = sorted(pnl_wide_df_display.index.get_level_values(STMT_MAP_COLUMN).unique().tolist()) if not pnl_wide_df_display.empty else []
    per_opts_dd = pnl_wide_df_display.columns.tolist() if not pnl_wide_df_display.empty else []
    
    with col1_pnl_sel: 
        widget_acc_name = st.selectbox(f"{prefix.upper()} GL Account:", acc_opts_dd, 
                                       index=utils.get_index(acc_opts_dd, st.session_state[f'{prefix}_selected_account_name']), 
                                       key=f"{prefix}_widget_acc_select", disabled=not acc_opts_dd)
    with col2_pnl_sel: 
        widget_per = st.selectbox(f"{prefix.upper()} Period:", per_opts_dd, 
                                  index=utils.get_index(per_opts_dd, st.session_state[f'{prefix}_selected_period']), 
                                  key=f"{prefix}_widget_per_select", disabled=not per_opts_dd)

    widget_acc_id = pnl_name_to_id_map_dd.get(widget_acc_name) if widget_acc_name else None # Use map from category-filtered data
    if (widget_acc_id != st.session_state[f'{prefix}_selected_account_id'] or \
        widget_per != st.session_state[f'{prefix}_selected_period'] or \
        widget_acc_name != st.session_state[f'{prefix}_selected_account_name']):
        if widget_acc_id and widget_per and widget_acc_name:
            st.session_state[f'{prefix}_selected_account_id'] = widget_acc_id
            st.session_state[f'{prefix}_selected_account_name'] = widget_acc_name
            st.session_state[f'{prefix}_selected_period'] = widget_per
            st.session_state[f'{prefix}_needs_je_refetch'] = True; st.rerun()

    if st.session_state[f'{prefix}_needs_je_refetch']:
        sel_id, sel_per = st.session_state[f'{prefix}_selected_account_id'], st.session_state[f'{prefix}_selected_period']
        if sel_id and sel_per:
            try:
                fetched_jes = data_processor.get_journal_entries(sel_id, sel_per, je_detail_df_global, JE_ID_COLUMN_GLOBAL, JE_DATE_COLUMN_GLOBAL, ALL_JE_COLUMNS_FROM_SOURCE_GLOBAL)
                temp_df = fetched_jes.copy() if isinstance(fetched_jes, pd.DataFrame) else pd.DataFrame()
                if JE_AMOUNT_COLUMN_GLOBAL in temp_df.columns and st.session_state.fsa_filter_je_amount_operator != "Off":
                    op_je, val_je = st.session_state.fsa_filter_je_amount_operator, float(st.session_state.fsa_filter_je_amount_value)
                    if not pd.api.types.is_numeric_dtype(temp_df[JE_AMOUNT_COLUMN_GLOBAL]):
                        temp_df[JE_AMOUNT_COLUMN_GLOBAL] = pd.to_numeric(temp_df[JE_AMOUNT_COLUMN_GLOBAL], errors='coerce')
                    temp_df.dropna(subset=[JE_AMOUNT_COLUMN_GLOBAL], inplace=True)
                    if op_je == "Greater than (>)": temp_df = temp_df[temp_df[JE_AMOUNT_COLUMN_GLOBAL] > val_je]
                    elif op_je == "Less than (<)": temp_df = temp_df[temp_df[JE_AMOUNT_COLUMN_GLOBAL] < val_je]
                    elif op_je == "Absolute value greater than (|x| >)": temp_df = temp_df[temp_df[JE_AMOUNT_COLUMN_GLOBAL].abs() > val_je]
                st.session_state[f'{prefix}_related_jes_df'] = temp_df
                st.session_state[f'{prefix}_prev_selected_account_id'], st.session_state[f'{prefix}_prev_selected_period'] = sel_id, sel_per
            except Exception as e: st.error(f"Error fetching {prefix.upper()} JEs: {e}"); st.session_state[f'{prefix}_related_jes_df'] = pd.DataFrame()
            finally: st.session_state[f'{prefix}_needs_je_refetch'] = False
    
    st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>Journal Entry Details for Selected P&L Account</h2>", unsafe_allow_html=True)
    # ... (P&L JE Display - same as before, uses pnl_related_jes_df)
    related_jes_to_display = st.session_state[f'{prefix}_related_jes_df']
    # ... (rest of JE display logic)

    if st.session_state[f'{prefix}_llm_prompt_for_display']:
        with st.expander(f"View Last Generated LLM Prompt For {prefix.upper()} Analysis", expanded=False):
            st.text_area("LLM Prompt:", value=st.session_state[f'{prefix}_llm_prompt_for_display'], height=300, disabled=True, key=f"{prefix}_llm_prompt_display_area")
    
    st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>LLM Period Analysis for P&L</h2>", unsafe_allow_html=True)
    # ... (P&L LLM Analysis - same as before, uses pnl_related_jes_df, pnl_wide_view_df for context, pnl_llm_analyses etc.)
    
    st.sidebar.markdown("---"); st.sidebar.header(f"P&L LLM Analysis History") # PNL Specific
    # ... (P&L LLM History Display)

    st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>Duplicate Value Lookup (P&L Context)</h2>", unsafe_allow_html=True)
    # ... (P&L Duplicate Finder - uses pnl_related_jes_df for options, je_detail_df_for_dup_search for searching)


# --- BS Tab ---
with tab_bs:
    prefix = "bs"
    st.header("Balance Sheet Drilldown")

    if bs_flat_df_source_global is None or bs_flat_df_source_global.empty:
        st.info("Balance Sheet data not available from upload. Please ensure 'active_bs_flat_df' is generated.")
        st.stop()

    st.markdown("#### Filter BS Table by Category")
    bs_account_cat_map = get_account_category_map(je_detail_df_global, JE_ID_COLUMN_GLOBAL, utils.BS_CATEGORIES_LIST)
    bs_cat_options = utils.BS_CATEGORIES_LIST
    if bs_account_cat_map:
        actual_bs_cats = sorted(list(set(bs_account_cat_map.values())))
        if actual_bs_cats: bs_cat_options = actual_bs_cats
    if not st.session_state[f'{prefix}_category_filter_selection'] and bs_cat_options:
        st.session_state[f'{prefix}_category_filter_selection'] = bs_cat_options[:]

    sel_bs_cats = st.multiselect("Select BS Categories:", bs_cat_options, default=st.session_state[f'{prefix}_category_filter_selection'], key=f"{prefix}_cat_multi")
    if sel_bs_cats != st.session_state[f'{prefix}_category_filter_selection']:
        st.session_state[f'{prefix}_category_filter_selection'] = sel_bs_cats; st.rerun()

    bs_flat_for_pivot, bs_name_to_id_map_dd = prepare_statement_flat_data_for_pivot(
        bs_flat_df_source_global, STMT_ID_COLUMN, STMT_MAP_COLUMN, STMT_TYPE_COLUMN,
        bs_account_cat_map, st.session_state[f'{prefix}_category_filter_selection'], utils.BS_CATEGORIES_LIST
    )

    bs_wide_df, bs_diff_df, bs_row_std_diff, bs_wide_reset = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.DataFrame()
    if not bs_flat_for_pivot.empty:
        try:
            bs_wide_df = bs_flat_df_for_pivot.pivot_table(index=[STMT_ID_COLUMN, STMT_MAP_COLUMN], columns=STMT_PERIOD_COLUMN, values=STMT_AMOUNT_COLUMN).fillna(0)
            if not bs_wide_df.empty:
                bs_wide_df = bs_wide_df.sort_index(axis=1)
                bs_diff_df = bs_wide_df.diff(axis=1).fillna(0) # MoM change for BS balances
                bs_row_std_diff = bs_wide_df.diff(axis=1).std(axis=1, skipna=True).fillna(0)
            bs_wide_reset = bs_wide_df.reset_index()
        except Exception as e: st.error(f"Error pivoting BS data: {e}")

    bs_wide_df_display = bs_wide_df.copy()
    if st.session_state.bs_show_highlighted_only: # Using BS-specific state
        highlighted_indices_bs = []
        if not bs_wide_df.empty and not bs_diff_df.empty and not bs_row_std_diff.empty:
            thresholds_bs = st.session_state.bs_outlier_threshold * bs_row_std_diff
            for idx_tuple in bs_wide_df.index:
                if idx_tuple in bs_diff_df.index and idx_tuple in thresholds_bs.index:
                    if pd.notna(thresholds_bs.loc[idx_tuple]) and thresholds_bs.loc[idx_tuple] > 1e-6:
                        if (bs_diff_df.loc[idx_tuple].abs() > thresholds_bs.loc[idx_tuple]).any():
                            highlighted_indices_bs.append(idx_tuple)
        if highlighted_indices_bs: bs_wide_df_display = bs_wide_df.loc[highlighted_indices_bs]
        else: bs_wide_df_display = pd.DataFrame(index=bs_wide_df.index, columns=bs_wide_df.columns).iloc[0:0]

    st.markdown("---"); st.markdown("#### Balance Sheet Data Overview")
    st.caption(f"Highlighting: MoM change > {st.session_state.bs_outlier_threshold:.1f} Std Dev. Click cell for JE/LLM.")

    def handle_bs_table_click_for_tab():
        raw_sel = st.session_state.get(f"{prefix}_select_df_key", {}).get('selection', {})
        rows, cols = raw_sel.get('rows', []), raw_sel.get('columns', [])
        if rows and cols:
            try:
                pos_idx = rows[0]
                if not bs_wide_df_display.empty and pos_idx < len(bs_wide_df_display):
                    acc_multi_idx = bs_wide_df_display.index[pos_idx]
                    acc_id, acc_name, period = str(acc_multi_idx[0]), str(acc_multi_idx[1]), cols[0]
                    if acc_id != st.session_state[f'{prefix}_selected_account_id'] or \
                       acc_name != st.session_state[f'{prefix}_selected_account_name'] or \
                       period != st.session_state[f'{prefix}_selected_period']:
                        st.session_state[f'{prefix}_selected_account_id'] = acc_id
                        st.session_state[f'{prefix}_selected_account_name'] = acc_name
                        st.session_state[f'{prefix}_selected_period'] = period
                        st.session_state[f'{prefix}_needs_je_refetch'] = True; st.rerun()
            except Exception as e: st.warning(f"BS table click error: {e}")
            
    if not bs_wide_df_display.empty:
        styled_df_bs = bs_wide_df_display.style.apply(utils.highlight_outliers_pandas, axis=1, diffs_df=bs_diff_df, thresholds_series=(st.session_state.bs_outlier_threshold * bs_row_std_diff), color=utils.EY_YELLOW, text_color=utils.EY_TEXT_ON_YELLOW).format("{:,.0f}")
        st.dataframe(styled_df_bs, use_container_width=True, height=400, key=f"{prefix}_select_df_key", on_select="rerun", selection_mode=("single-row", "single-column"))
        if st.session_state.get(f"{prefix}_select_df_key"): handle_bs_table_click_for_tab()
    else: st.info("BS table empty based on current filters.")

    st.markdown("---"); st.markdown("#### Select Account & Period for BS Drilldown")
    st.caption("*(Populated from BS table. Click table or use dropdowns.)*")
    col1_bs_sel, col2_bs_sel = st.columns(2)
    acc_opts_dd_bs = sorted(bs_wide_df_display.index.get_level_values(STMT_MAP_COLUMN).unique().tolist()) if not bs_wide_df_display.empty else []
    per_opts_dd_bs = bs_wide_df_display.columns.tolist() if not bs_wide_df_display.empty else []

    with col1_bs_sel: 
        widget_acc_name_bs = st.selectbox(f"{prefix.upper()} GL Account:", acc_opts_dd_bs, 
                                       index=utils.get_index(acc_opts_dd_bs, st.session_state[f'{prefix}_selected_account_name']), 
                                       key=f"{prefix}_widget_acc_select", disabled=not acc_opts_dd_bs)
    with col2_bs_sel: 
        widget_per_bs = st.selectbox(f"{prefix.upper()} Period:", per_opts_dd_bs, 
                                  index=utils.get_index(per_opts_dd_bs, st.session_state[f'{prefix}_selected_period']), 
                                  key=f"{prefix}_widget_per_select", disabled=not per_opts_dd_bs)

    widget_acc_id_bs = bs_name_to_id_map_dd.get(widget_acc_name_bs) if widget_acc_name_bs else None
    if (widget_acc_id_bs != st.session_state[f'{prefix}_selected_account_id'] or \
        widget_per_bs != st.session_state[f'{prefix}_selected_period'] or \
        widget_acc_name_bs != st.session_state[f'{prefix}_selected_account_name']):
        if widget_acc_id_bs and widget_per_bs and widget_acc_name_bs:
            st.session_state[f'{prefix}_selected_account_id'] = widget_acc_id_bs
            st.session_state[f'{prefix}_selected_account_name'] = widget_acc_name_bs
            st.session_state[f'{prefix}_selected_period'] = widget_per_bs
            st.session_state[f'{prefix}_needs_je_refetch'] = True; st.rerun()

    if st.session_state[f'{prefix}_needs_je_refetch']:
        sel_id, sel_per = st.session_state[f'{prefix}_selected_account_id'], st.session_state[f'{prefix}_selected_period']
        if sel_id and sel_per:
            try:
                fetched_jes = data_processor.get_journal_entries(sel_id, sel_per, je_detail_df_global, JE_ID_COLUMN_GLOBAL, JE_DATE_COLUMN_GLOBAL, ALL_JE_COLUMNS_FROM_SOURCE_GLOBAL)
                temp_df = fetched_jes.copy() if isinstance(fetched_jes, pd.DataFrame) else pd.DataFrame()
                # Apply global JE amount filter
                if JE_AMOUNT_COLUMN_GLOBAL in temp_df.columns and st.session_state.fsa_filter_je_amount_operator != "Off":
                    op_je, val_je = st.session_state.fsa_filter_je_amount_operator, float(st.session_state.fsa_filter_je_amount_value)
                    if not pd.api.types.is_numeric_dtype(temp_df[JE_AMOUNT_COLUMN_GLOBAL]):
                        temp_df[JE_AMOUNT_COLUMN_GLOBAL] = pd.to_numeric(temp_df[JE_AMOUNT_COLUMN_GLOBAL], errors='coerce')
                    temp_df.dropna(subset=[JE_AMOUNT_COLUMN_GLOBAL], inplace=True)
                    if op_je == "Greater than (>)": temp_df = temp_df[temp_df[JE_AMOUNT_COLUMN_GLOBAL] > val_je]
                    # ... other operators
                    elif op_je == "Less than (<)": temp_df = temp_df[temp_df[JE_AMOUNT_COLUMN_GLOBAL] < val_je]
                    elif op_je == "Absolute value greater than (|x| >)": temp_df = temp_df[temp_df[JE_AMOUNT_COLUMN_GLOBAL].abs() > val_je]
                st.session_state[f'{prefix}_related_jes_df'] = temp_df
                st.session_state[f'{prefix}_prev_selected_account_id'], st.session_state[f'{prefix}_prev_selected_period'] = sel_id, sel_per
            except Exception as e: st.error(f"Error fetching {prefix.upper()} JEs: {e}"); st.session_state[f'{prefix}_related_jes_df'] = pd.DataFrame()
            finally: st.session_state[f'{prefix}_needs_je_refetch'] = False
    
    st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>Journal Entry Details for Selected BS Account</h2>", unsafe_allow_html=True)
    # ... (BS JE Display, reusing PNL logic but with "bs_" prefixed state and bs_wide_df for amounts) ...
    # ... Example: st.dataframe(st.session_state[f'{prefix}_related_jes_df'], ...)

    if st.session_state.get(f'{prefix}_llm_prompt_for_display'):
        with st.expander(f"View Last Generated LLM Prompt For {prefix.upper()} Analysis", expanded=False):
            st.text_area("LLM Prompt:", value=st.session_state[f'{prefix}_llm_prompt_for_display'], height=300, disabled=True, key=f"{prefix}_llm_prompt_display_area")

    st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>LLM Period Analysis for BS</h2>", unsafe_allow_html=True)
    # ... (BS LLM Analysis - using "bs_" prefixed state, bs_wide_df for context. Prompt content needs care for BS.) ...

    st.sidebar.markdown("---"); st.sidebar.header(f"BS LLM Analysis History") # BS Specific
    # ... (BS LLM History Display - using "bs_" prefixed state)

    st.markdown(f"<hr><h2 style='color: {utils.EY_DARK_BLUE_GREY};'>Duplicate Value Lookup (BS Context)</h2>", unsafe_allow_html=True)
    # ... (BS Duplicate Finder - uses bs_related_jes_df for options, je_detail_df_for_dup_search for searching)
    st.info("Full JE Display, LLM Analysis, and Duplicate Finder for BS to be implemented by adapting P&L tab's logic.")