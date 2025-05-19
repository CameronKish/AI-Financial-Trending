# pages/2_Visualizations.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import utils 
from datetime import date, timedelta

st.set_page_config(layout="wide", page_title="Visualizations")
st.markdown(f"<style> h2 {{ color: {utils.EY_DARK_BLUE_GREY}; }} </style>", unsafe_allow_html=True)

# --- Data Loading and Initial Column Definition ---
pl_df_for_viz = pd.DataFrame()
bs_df_for_viz = pd.DataFrame() 
je_df_for_viz_source = pd.DataFrame() 
using_uploaded_data = False

PL_ACCOUNT_NAME_COL, PL_AMOUNT_COL, PL_PERIOD_DT_COL, PL_ACCOUNT_TYPE_COL = None, None, None, None
BS_ACCOUNT_NAME_COL, BS_AMOUNT_COL, BS_PERIOD_DT_COL, BS_ACCOUNT_TYPE_COL = None, None, None, None
JE_AMOUNT_COL_VIZ, JE_ACCOUNT_NAME_COL_VIZ, JE_ACCOUNT_ID_COL_VIZ, ACTUAL_JE_DATE_COL = None, None, None, None

if st.session_state.get('uploaded_data_ready', False):
    temp_pl_flat_df = st.session_state.get('active_pl_flat_df')
    if temp_pl_flat_df is not None and not temp_pl_flat_df.empty and \
       all(col in temp_pl_flat_df.columns for col in ['Account Name', 'Amount', 'Period_dt', '_Account_Type_']):
        pl_df_for_viz = temp_pl_flat_df.copy() 
        PL_ACCOUNT_NAME_COL, PL_AMOUNT_COL, PL_PERIOD_DT_COL, PL_ACCOUNT_TYPE_COL = 'Account Name', 'Amount', 'Period_dt', '_Account_Type_'
        using_uploaded_data = True
    else: pl_df_for_viz = pd.DataFrame()

    temp_bs_flat_df = st.session_state.get('active_bs_flat_df')
    if temp_bs_flat_df is not None and not temp_bs_flat_df.empty and \
       all(col in temp_bs_flat_df.columns for col in ['Account Name', 'Amount', 'Period_dt', '_Account_Type_']):
        bs_df_for_viz = temp_bs_flat_df.copy() 
        BS_ACCOUNT_NAME_COL, BS_AMOUNT_COL, BS_PERIOD_DT_COL, BS_ACCOUNT_TYPE_COL = 'Account Name', 'Amount', 'Period_dt', '_Account_Type_'
        if not using_uploaded_data: using_uploaded_data = True 
    else: bs_df_for_viz = pd.DataFrame()

    temp_je_df = st.session_state.get('compiled_je_df')
    if temp_je_df is not None and not temp_je_df.empty:
        je_df_for_viz_source = temp_je_df.copy() 
        uploaded_mappings = st.session_state.get('column_mappings', {})
        JE_AMOUNT_COL_VIZ = uploaded_mappings.get('amount')
        JE_ACCOUNT_NAME_COL_VIZ = uploaded_mappings.get('mapping') 
        JE_ACCOUNT_ID_COL_VIZ = uploaded_mappings.get('account_id') 
        ACTUAL_JE_DATE_COL = uploaded_mappings.get('date')
        if not using_uploaded_data : using_uploaded_data = True 
    else: je_df_for_viz_source = pd.DataFrame()
        
    if using_uploaded_data:
        st.success("Displaying visualizations from uploaded data.", icon="✅")
    
elif st.session_state.get('data_loaded', False): 
    pl_df_for_viz = st.session_state.get('pl_flat_df', pd.DataFrame()).copy()
    bs_df_for_viz = pd.DataFrame() 
    je_df_for_viz_source = st.session_state.get('je_detail_df', pd.DataFrame()).copy()
    original_col_config = st.session_state.get('column_config', {})
    PL_ACCOUNT_NAME_COL, PL_AMOUNT_COL, PL_PERIOD_DT_COL = original_col_config.get("PL_MAP_DISPLAY"), 'Amount', 'Period_dt' 
    JE_AMOUNT_COL_VIZ, JE_ACCOUNT_NAME_COL_VIZ, JE_ACCOUNT_ID_COL_VIZ, ACTUAL_JE_DATE_COL = \
        original_col_config.get("JE_AMOUNT"), original_col_config.get("JE_ACCOUNT_NAME"), \
        original_col_config.get("JE_ID"), original_col_config.get("JE_DATE")
    st.info("Displaying visualizations from sample data. For your own data, use 'Data Upload & Validation'.", icon="ℹ️")
else:
    st.error("Data not available. Load data or use 'Data Upload & Validation'."); st.stop()

# --- Initialize filter states ---
if 'viz_filter_start_date' not in st.session_state: st.session_state.viz_filter_start_date = None
if 'viz_filter_end_date' not in st.session_state: st.session_state.viz_filter_end_date = None
if 'viz_filter_amount_value' not in st.session_state: st.session_state.viz_filter_amount_value = 0.0
if 'viz_filter_amount_operator' not in st.session_state: st.session_state.viz_filter_amount_operator = "Off"
pl_type_options_init = ["Individual Accounts Only", "Calculated Groupings Only"]
if 'pl_trend_account_type_filter' not in st.session_state or st.session_state.pl_trend_account_type_filter not in pl_type_options_init:
    st.session_state.pl_trend_account_type_filter = pl_type_options_init[0] 
if 'bs_trend_account_type_filter' not in st.session_state or st.session_state.bs_trend_account_type_filter not in pl_type_options_init:
    st.session_state.bs_trend_account_type_filter = pl_type_options_init[0]
if 'je_scope_filter_radio_state' not in st.session_state: st.session_state.je_scope_filter_radio_state = "All JEs"
if 'chart_accounts_selection_pl' not in st.session_state: st.session_state.chart_accounts_selection_pl = [] 
if 'chart_accounts_selection_bs' not in st.session_state: st.session_state.chart_accounts_selection_bs = [] 


# --- Sidebar Filters ---
st.sidebar.header("Visualization Filters")
all_dates = []
if not pl_df_for_viz.empty and PL_PERIOD_DT_COL and pd.api.types.is_datetime64_any_dtype(pl_df_for_viz[PL_PERIOD_DT_COL]):
    all_dates.extend(pl_df_for_viz[PL_PERIOD_DT_COL].dropna())
if not bs_df_for_viz.empty and BS_PERIOD_DT_COL and pd.api.types.is_datetime64_any_dtype(bs_df_for_viz[BS_PERIOD_DT_COL]):
    all_dates.extend(bs_df_for_viz[BS_PERIOD_DT_COL].dropna())
if not je_df_for_viz_source.empty and isinstance(ACTUAL_JE_DATE_COL, str) and ACTUAL_JE_DATE_COL in je_df_for_viz_source.columns:
    try: 
        je_dates = pd.to_datetime(je_df_for_viz_source[ACTUAL_JE_DATE_COL], errors='coerce').dropna()
        if not je_dates.empty: all_dates.extend(je_dates)
    except Exception: pass 
min_overall_date_dt, max_overall_date_dt = (min(all_dates).date(), max(all_dates).date()) if all_dates else (date.today() - timedelta(days=365), date.today())
if pd.isna(min_overall_date_dt) or min_overall_date_dt is pd.Timestamp.min : min_overall_date_dt = date.today() - timedelta(days=365)
if pd.isna(max_overall_date_dt) or max_overall_date_dt is pd.Timestamp.max : max_overall_date_dt = date.today()

st.sidebar.subheader("Date Range Filter")
default_start_date = st.session_state.viz_filter_start_date if st.session_state.viz_filter_start_date else min_overall_date_dt
default_end_date = st.session_state.viz_filter_end_date if st.session_state.viz_filter_end_date else max_overall_date_dt
col_start, col_end = st.sidebar.columns(2)
new_start_date = col_start.date_input("Start Date", value=default_start_date, min_value=min_overall_date_dt, max_value=max_overall_date_dt, key="viz_start_date_picker_v5_errfix") # incremented key
new_end_date = col_end.date_input("End Date", value=default_end_date, min_value=min_overall_date_dt, max_value=max_overall_date_dt, key="viz_end_date_picker_v5_errfix") # incremented key

reset_date_button_key = "reset_date_range_viz_v2"
if st.sidebar.button("Reset Date Range to Max", key=reset_date_button_key):
    st.session_state.viz_filter_start_date = min_overall_date_dt 
    st.session_state.viz_filter_end_date = max_overall_date_dt   
    st.rerun()

if new_start_date != st.session_state.viz_filter_start_date or new_end_date != st.session_state.viz_filter_end_date:
    # Only update and rerun if not triggered by the reset button itself (to avoid double rerun)
    # This check is a bit tricky due to Streamlit's execution. Simpler to always set and rerun if different.
    st.session_state.viz_filter_start_date = new_start_date
    st.session_state.viz_filter_end_date = new_end_date
    st.rerun()

st.sidebar.subheader("JE Amount Filter")
st.sidebar.caption("This filter applies to individual Journal Entry amounts.")
amount_operator_options = ["Off", "Greater than (>)", "Less than (<)", "Absolute value greater than (|x| >)"]
selected_amount_operator = st.sidebar.radio("Operator:", options=amount_operator_options, 
    index=amount_operator_options.index(st.session_state.viz_filter_amount_operator), 
    key="viz_amount_op_radio_v5_errfix") # incremented key
amount_value_disabled = (selected_amount_operator == "Off")
new_amount_value = st.sidebar.number_input("Amount Value:", 
    value=float(st.session_state.viz_filter_amount_value or 0.0), 
    disabled=amount_value_disabled, step=100.0, format="%.2f", key="viz_amount_val_input_v5_errfix") # incremented key
if selected_amount_operator != st.session_state.viz_filter_amount_operator or \
   (not amount_value_disabled and new_amount_value != st.session_state.viz_filter_amount_value):
    st.session_state.viz_filter_amount_operator = selected_amount_operator
    st.session_state.viz_filter_amount_value = new_amount_value if not amount_value_disabled else 0.0
    st.rerun()

if st.sidebar.button("Clear All Viz Filters", key="clear_viz_filters_btn_v7_errfix"): # incremented key
    st.session_state.viz_filter_start_date = min_overall_date_dt 
    st.session_state.viz_filter_end_date = max_overall_date_dt   
    st.session_state.viz_filter_amount_operator = "Off"
    st.session_state.viz_filter_amount_value = 0.0
    st.rerun()

# --- Apply Filters to DataFrames for Visualization ---
pl_df_filtered = pl_df_for_viz.copy()
if not pl_df_filtered.empty and PL_PERIOD_DT_COL and pd.api.types.is_datetime64_any_dtype(pl_df_filtered[PL_PERIOD_DT_COL]):
    start_date_filter_dt = pd.to_datetime(st.session_state.viz_filter_start_date).normalize() if st.session_state.viz_filter_start_date else None
    end_date_filter_dt = pd.to_datetime(st.session_state.viz_filter_end_date).normalize() if st.session_state.viz_filter_end_date else None
    if start_date_filter_dt: pl_df_filtered = pl_df_filtered[pd.to_datetime(pl_df_filtered[PL_PERIOD_DT_COL]).dt.normalize() >= start_date_filter_dt]
    if end_date_filter_dt: pl_df_filtered = pl_df_filtered[pd.to_datetime(pl_df_filtered[PL_PERIOD_DT_COL]).dt.normalize() <= end_date_filter_dt]

bs_df_filtered = bs_df_for_viz.copy()
if not bs_df_filtered.empty and BS_PERIOD_DT_COL and pd.api.types.is_datetime64_any_dtype(bs_df_filtered[BS_PERIOD_DT_COL]):
    start_date_filter_dt = pd.to_datetime(st.session_state.viz_filter_start_date).normalize() if st.session_state.viz_filter_start_date else None
    end_date_filter_dt = pd.to_datetime(st.session_state.viz_filter_end_date).normalize() if st.session_state.viz_filter_end_date else None
    if start_date_filter_dt: bs_df_filtered = bs_df_filtered[pd.to_datetime(bs_df_filtered[BS_PERIOD_DT_COL]).dt.normalize() >= start_date_filter_dt]
    if end_date_filter_dt: bs_df_filtered = bs_df_filtered[pd.to_datetime(bs_df_filtered[BS_PERIOD_DT_COL]).dt.normalize() <= end_date_filter_dt]

je_df_for_current_viz_filtered = je_df_for_viz_source.copy()
if not je_df_for_current_viz_filtered.empty:
    if isinstance(ACTUAL_JE_DATE_COL, str) and ACTUAL_JE_DATE_COL in je_df_for_current_viz_filtered.columns:
        try:
            je_df_for_current_viz_filtered['_Filter_Date_dt_'] = pd.to_datetime(je_df_for_current_viz_filtered[ACTUAL_JE_DATE_COL], errors='coerce').dt.normalize()
            je_df_for_current_viz_filtered.dropna(subset=['_Filter_Date_dt_'], inplace=True)
            start_date_filter_dt = pd.to_datetime(st.session_state.viz_filter_start_date).normalize() if st.session_state.viz_filter_start_date else None
            end_date_filter_dt = pd.to_datetime(st.session_state.viz_filter_end_date).normalize() if st.session_state.viz_filter_end_date else None
            if start_date_filter_dt: je_df_for_current_viz_filtered = je_df_for_current_viz_filtered[je_df_for_current_viz_filtered['_Filter_Date_dt_'] >= start_date_filter_dt]
            if end_date_filter_dt: je_df_for_current_viz_filtered = je_df_for_current_viz_filtered[je_df_for_current_viz_filtered['_Filter_Date_dt_'] <= end_date_filter_dt]
            je_df_for_current_viz_filtered = je_df_for_current_viz_filtered.drop(columns=['_Filter_Date_dt_'], errors='ignore')
        except Exception: pass 
    if isinstance(JE_AMOUNT_COL_VIZ, str) and JE_AMOUNT_COL_VIZ in je_df_for_current_viz_filtered.columns and st.session_state.viz_filter_amount_operator != "Off" and st.session_state.viz_filter_amount_value is not None:
        try:
            je_df_for_current_viz_filtered[JE_AMOUNT_COL_VIZ] = pd.to_numeric(je_df_for_current_viz_filtered[JE_AMOUNT_COL_VIZ], errors='coerce')
            je_df_for_current_viz_filtered.dropna(subset=[JE_AMOUNT_COL_VIZ], inplace=True)
            op, val = st.session_state.viz_filter_amount_operator, float(st.session_state.viz_filter_amount_value)
            if op == "Greater than (>)": je_df_for_current_viz_filtered = je_df_for_current_viz_filtered[je_df_for_current_viz_filtered[JE_AMOUNT_COL_VIZ] > val]
            elif op == "Less than (<)": je_df_for_current_viz_filtered = je_df_for_current_viz_filtered[je_df_for_current_viz_filtered[JE_AMOUNT_COL_VIZ] < val]
            elif op == "Absolute value greater than (|x| >)": je_df_for_current_viz_filtered = je_df_for_current_viz_filtered[je_df_for_current_viz_filtered[JE_AMOUNT_COL_VIZ].abs() > val]
        except Exception as e_amt_filter: st.sidebar.warning(f"Amt filter err: {e_amt_filter}")

st.title("Data Visualizations")

# --- CORRECTED: Define tabs at the correct scope ---
tab1, tab2, tab3 = st.tabs(["P&L Account Trends", "Balance Sheet Account Trends", "JE Analysis"])

with tab1: 
    st.header("P&L Account Trends")
    if not pl_df_filtered.empty and PL_ACCOUNT_NAME_COL and PL_AMOUNT_COL and PL_PERIOD_DT_COL:
        pl_df_for_type_filter_tab1 = pl_df_filtered.copy()
        pl_type_options_tab1 = ["Individual Accounts Only", "Calculated Groupings Only"]
        current_pl_type_filter_state_tab1 = st.session_state.pl_trend_account_type_filter
        selected_pl_type_filter_tab1 = st.radio("Show P&L Accounts:", options=pl_type_options_tab1,
                                      index=pl_type_options_tab1.index(current_pl_type_filter_state_tab1),
                                      horizontal=True, key="pl_type_filter_radio_widget_v6_errfix") 
        if selected_pl_type_filter_tab1 != current_pl_type_filter_state_tab1:
            st.session_state.pl_trend_account_type_filter = selected_pl_type_filter_tab1
            st.session_state.chart_accounts_selection_pl = [] 
            st.rerun()
        if using_uploaded_data and PL_ACCOUNT_TYPE_COL and PL_ACCOUNT_TYPE_COL in pl_df_for_type_filter_tab1.columns:
            if st.session_state.pl_trend_account_type_filter == pl_type_options_tab1[0]:
                pl_df_for_type_filter_tab1 = pl_df_for_type_filter_tab1[pl_df_for_type_filter_tab1[PL_ACCOUNT_TYPE_COL] == "Individual Account"]
            else: 
                pl_df_for_type_filter_tab1 = pl_df_for_type_filter_tab1[pl_df_for_type_filter_tab1[PL_ACCOUNT_TYPE_COL] == "Calculated Grouping"]
        elif not using_uploaded_data: pass
        
        pl_account_options_tab1 = sorted(pl_df_for_type_filter_tab1[PL_ACCOUNT_NAME_COL].astype(str).unique().tolist()) if PL_ACCOUNT_NAME_COL in pl_df_for_type_filter_tab1 and not pl_df_for_type_filter_tab1.empty else []
        current_multiselect_selection_pl_tab1 = st.session_state.get('chart_accounts_selection_pl', [])
        valid_defaults_for_multiselect_pl_tab1 = [item for item in current_multiselect_selection_pl_tab1 if item in pl_account_options_tab1]
        if not valid_defaults_for_multiselect_pl_tab1 and pl_account_options_tab1: 
            potential_defaults_tab1 = ["Net Income", "Gross Profit", "Total Revenue", "Total COGS", "Total Operating Expenses", "Total Other P&L Items"]
            valid_defaults_for_multiselect_pl_tab1 = [opt for opt in potential_defaults_tab1 if opt in pl_account_options_tab1][:3] 
            if not valid_defaults_for_multiselect_pl_tab1: valid_defaults_for_multiselect_pl_tab1 = pl_account_options_tab1[:min(1, len(pl_account_options_tab1))]
            st.session_state.chart_accounts_selection_pl = valid_defaults_for_multiselect_pl_tab1
        
        if not pl_account_options_tab1: st.info(f"No P&L accounts of type '{st.session_state.pl_trend_account_type_filter}' available after filters.")
        else:
            c1_pl_tab1, c2_pl_tab1, c3_pl_tab1 = st.columns([4,1,1])
            with c1_pl_tab1:
                user_pl_selection_tab1 = st.multiselect("Select P&L Account(s) to Plot:", options=pl_account_options_tab1, 
                                                        default=st.session_state.chart_accounts_selection_pl, 
                                                        key="pl_trend_multiselect_widget_v7_errfix") 
                if user_pl_selection_tab1 != st.session_state.chart_accounts_selection_pl: 
                    st.session_state.chart_accounts_selection_pl = user_pl_selection_tab1; st.rerun()
            with c2_pl_tab1: st.markdown("<br>", unsafe_allow_html=True); 
            if st.button("All Accounts of Type", key="pl_all_btn_v7_errfix", use_container_width=True): st.session_state.chart_accounts_selection_pl = pl_account_options_tab1; st.rerun()
            with c3_pl_tab1: st.markdown("<br>", unsafe_allow_html=True); 
            if st.button("Clear Selection", key="pl_clear_btn_v7_errfix", use_container_width=True): st.session_state.chart_accounts_selection_pl = []; st.rerun()
            selected_pl_accounts_tab1 = st.session_state.get('chart_accounts_selection_pl', [])
            if selected_pl_accounts_tab1:
                chart_data_pl_tab1 = pl_df_for_type_filter_tab1[pl_df_for_type_filter_tab1[PL_ACCOUNT_NAME_COL].isin(selected_pl_accounts_tab1)].copy()
                if not chart_data_pl_tab1.empty:
                    chart_data_pl_tab1 = chart_data_pl_tab1.sort_values(by=PL_PERIOD_DT_COL)
                    fig_pl_tab1 = px.line(chart_data_pl_tab1, x=PL_PERIOD_DT_COL, y=PL_AMOUNT_COL, color=PL_ACCOUNT_NAME_COL, markers=True, title=f"P&L Trends ({st.session_state.pl_trend_account_type_filter})")
                    fig_pl_tab1.update_layout(xaxis_title="Period", yaxis_title="Amount ($)", yaxis_tickformat=",.0f", hovermode="x unified")
                    fig_pl_tab1.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Period: %{x|%Y-%m-%d}<br>Amount: %{y:,.0f}<extra></extra>")
                    st.plotly_chart(fig_pl_tab1, use_container_width=True)
                else: st.info("No data to plot for selected P&L accounts, type, and filters.")
            else: st.info("Select P&L account(s) to display trends.")
    else: st.info("P&L data not available or not correctly formatted for trend visualization (after filters).")

with tab2:
    st.header("Balance Sheet Account Trends")
    if not bs_df_filtered.empty and BS_ACCOUNT_NAME_COL and BS_AMOUNT_COL and BS_PERIOD_DT_COL:
        bs_df_for_type_filter_tab2 = bs_df_filtered.copy()
        bs_type_options_tab2 = ["Individual Accounts Only", "Calculated Groupings Only"]
        current_bs_type_filter_state_tab2 = st.session_state.bs_trend_account_type_filter
        selected_bs_type_filter_tab2 = st.radio("Show BS Accounts:", options=bs_type_options_tab2,
                                      index=bs_type_options_tab2.index(current_bs_type_filter_state_tab2),
                                      horizontal=True, key="bs_type_filter_radio_widget_v6_errfix") 
        if selected_bs_type_filter_tab2 != current_bs_type_filter_state_tab2:
            st.session_state.bs_trend_account_type_filter = selected_bs_type_filter_tab2
            st.session_state.chart_accounts_selection_bs = []
            st.rerun()
        if using_uploaded_data and BS_ACCOUNT_TYPE_COL and BS_ACCOUNT_TYPE_COL in bs_df_for_type_filter_tab2.columns:
            if st.session_state.bs_trend_account_type_filter == bs_type_options_tab2[0]: 
                bs_df_for_type_filter_tab2 = bs_df_for_type_filter_tab2[bs_df_for_type_filter_tab2[BS_ACCOUNT_TYPE_COL] == "Individual Account"]
            else: 
                bs_df_for_type_filter_tab2 = bs_df_for_type_filter_tab2[bs_df_for_type_filter_tab2[BS_ACCOUNT_TYPE_COL] == "Calculated Grouping"]
        elif not using_uploaded_data: pass
        
        bs_account_options_tab2 = sorted(bs_df_for_type_filter_tab2[BS_ACCOUNT_NAME_COL].astype(str).unique().tolist()) if BS_ACCOUNT_NAME_COL in bs_df_for_type_filter_tab2 and not bs_df_for_type_filter_tab2.empty else []
        current_multiselect_selection_bs_tab2 = st.session_state.get('chart_accounts_selection_bs', [])
        valid_defaults_for_multiselect_bs_tab2 = [item for item in current_multiselect_selection_bs_tab2 if item in bs_account_options_tab2]
        if not valid_defaults_for_multiselect_bs_tab2 and bs_account_options_tab2:
            potential_defaults_bs_tab2 = ["Total Assets", "Total Liabilities", "Total Equity", "Cash", "Calculated Retained Earnings"]
            valid_defaults_for_multiselect_bs_tab2 = [opt for opt in potential_defaults_bs_tab2 if opt in bs_account_options_tab2][:3]
            if not valid_defaults_for_multiselect_bs_tab2: valid_defaults_for_multiselect_bs_tab2 = bs_account_options_tab2[:min(1, len(bs_account_options_tab2))]
            st.session_state.chart_accounts_selection_bs = valid_defaults_for_multiselect_bs_tab2
        
        if not bs_account_options_tab2: st.info(f"No BS accounts of type '{st.session_state.bs_trend_account_type_filter}' available after filters.")
        else:
            c1_bs_tab2, c2_bs_tab2, c3_bs_tab2 = st.columns([4,1,1])
            with c1_bs_tab2:
                user_bs_selection_tab2 = st.multiselect("Select BS Account(s) to Plot:", options=bs_account_options_tab2, 
                                                        default=st.session_state.chart_accounts_selection_bs, 
                                                        key="bs_trend_multiselect_widget_v7_errfix") 
                if user_bs_selection_tab2 != st.session_state.chart_accounts_selection_bs: 
                    st.session_state.chart_accounts_selection_bs = user_bs_selection_tab2; st.rerun()
            with c2_bs_tab2: st.markdown("<br>", unsafe_allow_html=True); 
            if st.button("All Accounts of Type", key="bs_all_btn_v7_errfix", use_container_width=True): st.session_state.chart_accounts_selection_bs = bs_account_options_tab2; st.rerun()
            with c3_bs_tab2: st.markdown("<br>", unsafe_allow_html=True); 
            if st.button("Clear Selection", key="bs_clear_btn_v7_errfix", use_container_width=True): st.session_state.chart_accounts_selection_bs = []; st.rerun()
            selected_bs_accounts_tab2 = st.session_state.get('chart_accounts_selection_bs', [])
            if selected_bs_accounts_tab2:
                chart_data_bs_tab2 = bs_df_for_type_filter_tab2[bs_df_for_type_filter_tab2[BS_ACCOUNT_NAME_COL].isin(selected_bs_accounts_tab2)].copy()
                if not chart_data_bs_tab2.empty:
                    chart_data_bs_tab2 = chart_data_bs_tab2.sort_values(by=BS_PERIOD_DT_COL)
                    fig_bs_tab2 = px.line(chart_data_bs_tab2, x=BS_PERIOD_DT_COL, y=BS_AMOUNT_COL, color=BS_ACCOUNT_NAME_COL, markers=True, title=f"BS Trends ({st.session_state.bs_trend_account_type_filter})")
                    fig_bs_tab2.update_layout(xaxis_title="Period", yaxis_title="Amount ($)", yaxis_tickformat=",.0f", hovermode="x unified")
                    fig_bs_tab2.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Period: %{x|%Y-%m-%d}<br>Amount: %{y:,.0f}<extra></extra>")
                    st.plotly_chart(fig_bs_tab2, use_container_width=True)
                else: st.info("No data to plot for selected BS accounts, type, and filters.")
            else: st.info("Select BS account(s) to display trends.")
    else: st.info("BS data not available or not correctly formatted for trend visualization (after filters).")

with tab3: 
    st.header("JE Analysis")
    je_filter_scope_tab3 = st.session_state.get('je_scope_filter_radio_state', "All JEs")
    can_filter_je_scope_tab3 = False
    if using_uploaded_data and JE_ACCOUNT_ID_COL_VIZ and isinstance(JE_ACCOUNT_ID_COL_VIZ, str) and JE_ACCOUNT_ID_COL_VIZ in je_df_for_viz_source.columns:
        can_filter_je_scope_tab3 = True
        selected_je_scope_option_tab3 = st.radio( "Filter JEs by Statement Section:",
            options=["All JEs", "P&L Accounts Only", "BS Accounts Only"],
            index=["All JEs", "P&L Accounts Only", "BS Accounts Only"].index(je_filter_scope_tab3),
            horizontal=True, key="je_scope_filter_radio_widget_v7_errfix") 
        if selected_je_scope_option_tab3 != je_filter_scope_tab3:
            st.session_state.je_scope_filter_radio_state = selected_je_scope_option_tab3; st.rerun()
        je_filter_scope_tab3 = selected_je_scope_option_tab3
    elif using_uploaded_data: st.caption("JE Account ID for P&L/BS scope filtering unavailable.")

    je_df_scoped_tab3 = je_df_for_current_viz_filtered.copy() 
    if can_filter_je_scope_tab3 and je_filter_scope_tab3 != "All JEs":
        if JE_ACCOUNT_ID_COL_VIZ and JE_ACCOUNT_ID_COL_VIZ in je_df_scoped_tab3.columns:
            with st.spinner("Categorizing JEs for scope filter..."):
                try:
                    use_custom_ranges_rules_tab3 = st.session_state.get('use_custom_ranges', False)
                    parsed_gl_ranges_rules_tab3 = st.session_state.get('parsed_gl_ranges', {})
                    je_df_scoped_tab3['_StdAccID_Temp_'] = je_df_scoped_tab3[JE_ACCOUNT_ID_COL_VIZ].astype(str).str.strip()
                    je_df_scoped_tab3['_AssignedCat_Temp_'] = je_df_scoped_tab3['_StdAccID_Temp_'].apply(
                        lambda x: utils.assign_category_using_rules(x, use_custom_ranges_rules_tab3, parsed_gl_ranges_rules_tab3, utils.get_prefix_based_category)
                    )
                    je_df_scoped_tab3['_StmtSec_Temp_'] = je_df_scoped_tab3['_AssignedCat_Temp_'].apply(utils.get_statement_section)
                    if je_filter_scope_tab3 == "P&L Accounts Only": je_df_scoped_tab3 = je_df_scoped_tab3[je_df_scoped_tab3['_StmtSec_Temp_'] == 'P&L']
                    elif je_filter_scope_tab3 == "BS Accounts Only": je_df_scoped_tab3 = je_df_scoped_tab3[je_df_scoped_tab3['_StmtSec_Temp_'] == 'BS']
                except Exception as e_cat_je_tab3: st.error(f"Error during JE categorization: {e_cat_je_tab3}. Using JEs from global filters."); je_df_scoped_tab3 = je_df_for_current_viz_filtered.copy()
        else: st.warning("JE Account ID for P&L/BS scope filtering unavailable. Using JEs from global filters."); je_df_scoped_tab3 = je_df_for_current_viz_filtered.copy()

    st.subheader("Amount vs. Frequency Scatter Plot")
    if not je_df_scoped_tab3.empty and JE_AMOUNT_COL_VIZ and isinstance(JE_AMOUNT_COL_VIZ, str) and JE_AMOUNT_COL_VIZ in je_df_scoped_tab3.columns:
        analysis_cols_options_je_tab3 = [col for col in je_df_scoped_tab3.columns if (pd.api.types.is_string_dtype(je_df_scoped_tab3[col]) or pd.api.types.is_object_dtype(je_df_scoped_tab3[col])) and not col.startswith('_') and (1 < je_df_scoped_tab3[col].nunique(dropna=True) < min(len(je_df_scoped_tab3) * 0.9, 2000))]
        analysis_cols_options_je_tab3 = sorted(list(set(analysis_cols_options_je_tab3)))
        if analysis_cols_options_je_tab3:
            default_je_scatter_idx = 0; current_selected_analysis_col = st.session_state.get('selected_analysis_col_je_state')
            if current_selected_analysis_col and current_selected_analysis_col in analysis_cols_options_je_tab3: default_je_scatter_idx = analysis_cols_options_je_tab3.index(current_selected_analysis_col)
            elif JE_ACCOUNT_NAME_COL_VIZ and JE_ACCOUNT_NAME_COL_VIZ in analysis_cols_options_je_tab3: default_je_scatter_idx = analysis_cols_options_je_tab3.index(JE_ACCOUNT_NAME_COL_VIZ)
            elif 'Memo' in analysis_cols_options_je_tab3: default_je_scatter_idx = analysis_cols_options_je_tab3.index('Memo')
            elif not analysis_cols_options_je_tab3 : default_je_scatter_idx = 0 
            selected_analysis_col_je_tab3 = st.selectbox("Analyze JEs by:", options=analysis_cols_options_je_tab3, index=default_je_scatter_idx, key="je_scatter_select_v7_errfix")
            if selected_analysis_col_je_tab3 != current_selected_analysis_col: st.session_state.selected_analysis_col_je_state = selected_analysis_col_je_tab3; st.rerun()

            if selected_analysis_col_je_tab3: 
                agg_data_je_scatter_tab3 = pd.DataFrame()
                try:
                    with st.spinner(f"Aggregating for scatter by {selected_analysis_col_je_tab3}..."):
                        _je_df_for_scatter_agg_tab3 = je_df_scoped_tab3.copy()
                        _je_df_for_scatter_agg_tab3[JE_AMOUNT_COL_VIZ] = pd.to_numeric(_je_df_for_scatter_agg_tab3[JE_AMOUNT_COL_VIZ], errors='coerce').fillna(0)
                        count_col_je_scatter_tab3 = JE_ACCOUNT_ID_COL_VIZ if JE_ACCOUNT_ID_COL_VIZ and JE_ACCOUNT_ID_COL_VIZ in _je_df_for_scatter_agg_tab3.columns and _je_df_for_scatter_agg_tab3[JE_ACCOUNT_ID_COL_VIZ].nunique() > len(_je_df_for_scatter_agg_tab3) * 0.9 else ('Transaction Id' if 'Transaction Id' in _je_df_for_scatter_agg_tab3.columns else None)
                        if count_col_je_scatter_tab3: agg_data_je_scatter_tab3 = _je_df_for_scatter_agg_tab3.groupby(selected_analysis_col_je_tab3, dropna=False).agg(Count=(count_col_je_scatter_tab3, 'nunique'), Total_Amount=(JE_AMOUNT_COL_VIZ, 'sum')).reset_index()
                        else: agg_data_je_scatter_tab3 = _je_df_for_scatter_agg_tab3.groupby(selected_analysis_col_je_tab3, dropna=False).agg(Count=(JE_AMOUNT_COL_VIZ, 'size'), Total_Amount=(JE_AMOUNT_COL_VIZ, 'sum')).reset_index()
                        agg_data_je_scatter_tab3['Abs_Total_Amount'] = agg_data_je_scatter_tab3['Total_Amount'].abs()
                        agg_data_je_scatter_tab3[selected_analysis_col_je_tab3] = agg_data_je_scatter_tab3[selected_analysis_col_je_tab3].astype(str).fillna("N/A")
                        agg_data_je_scatter_tab3 = agg_data_je_scatter_tab3.sort_values(by='Abs_Total_Amount', ascending=False) 
                        agg_data_je_scatter_tab3['Total_Amount'] = agg_data_je_scatter_tab3['Total_Amount'].astype(float)
                    if not agg_data_je_scatter_tab3.empty:
                        min_x_tab3, max_x_tab3 = agg_data_je_scatter_tab3['Count'].min(), agg_data_je_scatter_tab3['Count'].max(); x_pad_tab3 = (max_x_tab3 - min_x_tab3) * 0.05 if max_x_tab3 > min_x_tab3 else 1 
                        fig_scatter_tab3 = px.scatter(agg_data_je_scatter_tab3, x='Count', y='Total_Amount', size='Abs_Total_Amount', hover_name=selected_analysis_col_je_tab3, color='Total_Amount', color_continuous_scale=px.colors.diverging.Picnic, title=f'JE Analysis by {selected_analysis_col_je_tab3} ({je_filter_scope_tab3})', labels={'Count': 'Frequency', 'Total_Amount': 'Net Amount ($)'}, size_max=60)
                        fig_scatter_tab3.update_layout(xaxis_title="Frequency", yaxis_title="Net Amount ($)", yaxis_tickformat=",.0f", xaxis_tickformat=",d", xaxis_range=[min_x_tab3 - x_pad_tab3 if min_x_tab3 > 0 else 0, max_x_tab3 + x_pad_tab3], xaxis_autorange=False )
                        fig_scatter_tab3.update_traces(hovertemplate=f"<b>%{{customdata[0]}}</b><br>Net Amount: %{{y:,.0f}}<br>Count: %{{x}}<extra></extra>", customdata=agg_data_je_scatter_tab3[[selected_analysis_col_je_tab3]].values)
                        st.plotly_chart(fig_scatter_tab3, use_container_width=True)
                        with st.expander(f"View All Aggregated Data for '{selected_analysis_col_je_tab3}' ({je_filter_scope_tab3})"):
                             display_df_je_scatter_tab3 = agg_data_je_scatter_tab3.copy(); display_df_je_scatter_tab3['Net Amount ($)'] = display_df_je_scatter_tab3['Total_Amount'].apply(lambda x: f"${x:,.0f}")
                             cols_to_display_je_scatter_tab3 = [selected_analysis_col_je_tab3, 'Count', 'Net Amount ($)']
                             st.dataframe(display_df_je_scatter_tab3[cols_to_display_je_scatter_tab3],key="je_scatter_detail_table_v4_errfix", use_container_width=True, column_config={selected_analysis_col_je_tab3: st.column_config.TextColumn(selected_analysis_col_je_tab3)}, hide_index=True )
                    else: st.info(f"No aggregated JE data for scatter plot '{selected_analysis_col_je_tab3}' ({je_filter_scope_tab3}).")
                except Exception as e_scatter_tab3: st.error(f"JE Scatter Plot Error: {e_scatter_tab3}")
            
            st.subheader(f"Trend of Top Categories in '{selected_analysis_col_je_tab3}' over Time")
            if selected_analysis_col_je_tab3 and ACTUAL_JE_DATE_COL and isinstance(ACTUAL_JE_DATE_COL, str) and ACTUAL_JE_DATE_COL in je_df_scoped_tab3.columns:
                trend_agg_df_filtered_tab3 = pd.DataFrame() 
                try:
                    with st.spinner(f"Aggregating for trend by {selected_analysis_col_je_tab3}..."):
                        if 'agg_data_je_scatter_tab3' in locals() and not agg_data_je_scatter_tab3.empty: # Check if scatter data was generated
                            trend_df_source_tab3 = je_df_scoped_tab3.copy() 
                            trend_df_source_tab3['_Date_dt_'] = pd.to_datetime(trend_df_source_tab3[ACTUAL_JE_DATE_COL], errors='coerce')
                            trend_df_source_tab3.dropna(subset=['_Date_dt_'], inplace=True)
                            trend_df_source_tab3['_YearMonth_'] = trend_df_source_tab3['_Date_dt_'].dt.to_period('M').astype(str)
                            trend_df_source_tab3[JE_AMOUNT_COL_VIZ] = pd.to_numeric(trend_df_source_tab3[JE_AMOUNT_COL_VIZ], errors='coerce').fillna(0)
                            trend_agg_df_tab3 = trend_df_source_tab3.groupby(['_YearMonth_', selected_analysis_col_je_tab3])[JE_AMOUNT_COL_VIZ].sum().reset_index()
                            trend_agg_df_tab3 = trend_agg_df_tab3.sort_values(by='_YearMonth_')
                            N_TOP_FOR_TREND_tab3 = 7 
                            top_categories_for_trend_tab3 = agg_data_je_scatter_tab3.nlargest(N_TOP_FOR_TREND_tab3, 'Abs_Total_Amount')[selected_analysis_col_je_tab3].tolist()
                            trend_agg_df_filtered_tab3 = trend_agg_df_tab3[trend_agg_df_tab3[selected_analysis_col_je_tab3].isin(top_categories_for_trend_tab3)]
                    if not trend_agg_df_filtered_tab3.empty:
                        fig_trend_je_tab3 = px.line(trend_agg_df_filtered_tab3, x='_YearMonth_', y=JE_AMOUNT_COL_VIZ, color=selected_analysis_col_je_tab3, markers=True, title=f"Monthly Trend for Top {N_TOP_FOR_TREND_tab3} '{selected_analysis_col_je_tab3}' ({je_filter_scope_tab3})", labels={'_YearMonth_': 'Month', JE_AMOUNT_COL_VIZ: 'Net Amount ($)'})
                        fig_trend_je_tab3.update_layout(xaxis_title="Month", yaxis_title="Net Amount ($)", yaxis_tickformat=",.0f", hovermode="x unified")
                        fig_trend_je_tab3.update_traces(hovertemplate="<b>%{fullData.name}</b><br>Month: %{x}<br>Amount: %{y:,.0f}<extra></extra>")
                        st.plotly_chart(fig_trend_je_tab3, use_container_width=True)
                        with st.expander(f"View Trend Data for Top {N_TOP_FOR_TREND_tab3} '{selected_analysis_col_je_tab3}'"):
                            display_trend_df_tab3 = trend_agg_df_filtered_tab3.copy(); display_trend_df_tab3['Net Amount ($)'] = display_trend_df_tab3[JE_AMOUNT_COL_VIZ].apply(lambda x: f"${x:,.0f}")
                            st.dataframe(display_trend_df_tab3[['_YearMonth_', selected_analysis_col_je_tab3, 'Net Amount ($)']],key="je_trend_detail_table_v4_errfix", use_container_width=True, hide_index=True)
                    else: st.info(f"No trend data for '{selected_analysis_col_je_tab3}' ({je_filter_scope_tab3}). May require scatter data or category has no time series data.")
                except Exception as e_trend_je_tab3: st.error(f"JE Trend Chart Error: {e_trend_je_tab3}")
            else: st.info("Select category & ensure date column mapped for JE trends.")
        else: st.info("No suitable text columns for JE aggregation after filters.")
    elif not je_df_scoped_tab3.empty: st.warning(f"JE Amount column missing. Cannot perform JE analysis.")
    else: st.info(f"No JE data for scope ('{je_filter_scope_tab3}') and global filters.")