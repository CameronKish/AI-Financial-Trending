# pages/0_Data_Upload.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re 
import utils 

st.set_page_config(layout="wide", page_title="Data Upload & Validation")
st.title("📄 Data Upload & Initial Validation")
st.caption("Upload your financial data (Journal Entries) in an Excel file, map the necessary columns, and view a generated P&L and Balance Sheet for validation.")

DEFAULT_PREFIX_MAP = { 
    "Assets": "1xxxx", "Liabilities": "2xxxx", "Equity": "3xxxx",
    "Revenue": "4xxxx", "COGS": "5xxxx", "Operating Expenses": "6xxxx, 7xxxx",
    "Other P&L Items": "8xxxx, 9xxxx"
}
CONFIG_SHEET_NAME = "_FINANCIAL_APP_CONFIG_" 
CALCULATED_AMOUNT_COL_NAME = "Total_Amount_(calc)" 

REV_TOTAL_ID = "REVENUE_TOTAL_CALC"
COGS_TOTAL_ID = "COGS_TOTAL_CALC"
OPEX_TOTAL_ID = "OPEX_TOTAL_CALC"
OTHER_PL_TOTAL_ID = "OTHER_PL_TOTAL_CALC"
GP_ID_CONST = "GP_CALC" 
NI_ID_CONST = "NET_INCOME_CALC"

PL_CALCULATED_IDS = [GP_ID_CONST, NI_ID_CONST, REV_TOTAL_ID, COGS_TOTAL_ID, OPEX_TOTAL_ID, OTHER_PL_TOTAL_ID]
BS_CALCULATED_IDS = ["ASSETS_TOTAL", "LIAB_TOTAL", "EQUITY_TOTAL", "RECALC"] 

def safe_read_excel_sheet(excel_file, sheet_name):
    try: return pd.read_excel(excel_file, sheet_name=sheet_name)
    except Exception as e: st.error(f"Error reading sheet '{sheet_name}': {e}"); return None

def style_financial_table(df_to_style, calculated_row_ids_list):
    df_copy = df_to_style.copy()
    def highlight_specific_rows(row):
        account_id_from_row = row.name[0] if isinstance(row.name, tuple) else row.name
        if str(account_id_from_row) in calculated_row_ids_list: 
            return ['font-weight: bold; background-color: #e6f3ff'] * len(row)
        return [''] * len(row)
    
    float_cols = df_copy.select_dtypes(include=[np.number]).columns
    format_dict = {col: "{:,.0f}" for col in float_cols}
    
    return df_copy.style.apply(highlight_specific_rows, axis=1).format(format_dict, na_rep="0")

def try_create_total_amount_column(debit_col_name, credit_col_name):
    if st.session_state.compiled_je_df is None or st.session_state.compiled_je_df.empty: return False
    if CALCULATED_AMOUNT_COL_NAME in st.session_state.compiled_je_df.columns:
        if pd.api.types.is_numeric_dtype(st.session_state.compiled_je_df[CALCULATED_AMOUNT_COL_NAME]): return True 
        else: st.warning(f"Column '{CALCULATED_AMOUNT_COL_NAME}' exists but is not numeric. Will attempt to recreate.")
    
    if debit_col_name and credit_col_name and \
       debit_col_name in st.session_state.compiled_je_df.columns and \
       credit_col_name in st.session_state.compiled_je_df.columns:
        try:
            temp_df = st.session_state.compiled_je_df.copy()
            temp_df[debit_col_name] = pd.to_numeric(temp_df[debit_col_name], errors='coerce').fillna(0)
            temp_df[credit_col_name] = pd.to_numeric(temp_df[credit_col_name], errors='coerce').fillna(0)
            temp_df[CALCULATED_AMOUNT_COL_NAME] = temp_df[debit_col_name] - temp_df[credit_col_name]
            st.session_state.compiled_je_df = temp_df
            return True
        except Exception as e: st.error(f"Error creating '{CALCULATED_AMOUNT_COL_NAME}': {e}"); return False
    else:
        msg = []
        if not debit_col_name: msg.append("Debit column not specified.")
        elif debit_col_name not in st.session_state.compiled_je_df.columns: msg.append(f"Specified Debit column ('{debit_col_name}') not found.")
        if not credit_col_name: msg.append("Credit column not specified.")
        elif credit_col_name not in st.session_state.compiled_je_df.columns: msg.append(f"Specified Credit column ('{credit_col_name}') not found.")
        if msg: st.warning(f"Could not create '{CALCULATED_AMOUNT_COL_NAME}': " + " ".join(msg))
        return False

def process_financial_data(df, account_id_col, mapping_col, amount_col, date_col,
                           use_custom_ranges_flag, parsed_gl_ranges_from_state, 
                           roll_equity_flag=False, roll_bs_overall_flag=False,
                           pnl_sign_convention="RevNeg_ExpPos"):
    processed_df = df.copy()
    if not all([account_id_col, mapping_col, amount_col, date_col]):
        st.error("Core column mappings incomplete for financial processing."); return None, None, None, None, None, None
    core_cols_map_check = {"Account ID": account_id_col, "Mapping": mapping_col, "Amount": amount_col, "Date": date_col}
    for dn, acn in core_cols_map_check.items():
        if acn not in processed_df.columns:
            st.error(f"Mapped '{dn}' col ('{acn}') not found in JE data."); return None, None, None, None, None, None
    try: processed_df[amount_col] = pd.to_numeric(processed_df[amount_col], errors='coerce').fillna(0)
    except Exception as e: st.error(f"Could not convert Amount col '{amount_col}' to numeric: {e}"); return None, None, None, None, None, None
    try:
        processed_df['_Standard_Date_'] = pd.to_datetime(processed_df[date_col], errors='coerce')
        if processed_df['_Standard_Date_'].isnull().any(): 
            st.warning(f"Some dates in '{date_col}' were unparseable; rows with invalid dates dropped for statement generation.")
            processed_df.dropna(subset=['_Standard_Date_'], inplace=True)
        if processed_df.empty: st.error("No valid dates found in data. Cannot proceed."); return None, None, None, None, None, None
        processed_df['_YearMonth_'] = processed_df['_Standard_Date_'].dt.strftime('%Y-%m')
    except Exception as e: st.error(f"Could not process Date col '{date_col}': {e}"); return None, None, None, None, None, None
    
    processed_df['_Standard_Account_ID_'] = processed_df[account_id_col].astype(str).str.strip()
    processed_df['_Standard_Mapping_'] = processed_df[mapping_col].astype(str).str.strip()
    
    processed_df['_Assigned_Category_'] = processed_df['_Standard_Account_ID_'].apply(
        lambda x: utils.assign_category_using_rules(x, use_custom_ranges_flag, parsed_gl_ranges_from_state, utils.get_prefix_based_category)
    )
    if use_custom_ranges_flag and any(bool(v) for v in parsed_gl_ranges_from_state.values() if v): 
        st.caption("Using custom GL ranges for categorization.")
    else: st.caption("Using default prefix-based GL classification.")
    
    processed_df['_Statement_Section_'] = processed_df['_Assigned_Category_'].apply(utils.get_statement_section)
    
    pl_data = processed_df[processed_df['_Statement_Section_'] == 'P&L']
    bs_data_activity = processed_df[processed_df['_Statement_Section_'] == 'BS'] 
    uncat_rows = processed_df[processed_df['_Statement_Section_'] == 'Uncategorized']
    uncat_sum_df = pd.DataFrame()
    if not uncat_rows.empty:
        uncat_sum_df = uncat_rows.groupby(['_Standard_Account_ID_', '_Standard_Mapping_'])[amount_col].sum().reset_index()
        uncat_sum_df.columns = ['Account ID', 'Account Name', 'Total Amount']
    
    pl_disp_df, ni_series_recalc = pd.DataFrame(), pd.Series(dtype=float)
    gp_name, ni_name = "Gross Profit", "Net Income" 
    pl_idx_names = ['Account ID', 'Account Name'] 
    empty_pl_idx = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=pl_idx_names)
    periods = sorted(processed_df['_YearMonth_'].unique()) if '_YearMonth_' in processed_df and not processed_df['_YearMonth_'].empty else []

    if not pl_data.empty:
        pl_grp = pl_data.groupby(['_Standard_Account_ID_', '_Standard_Mapping_', '_Assigned_Category_', '_YearMonth_'])[amount_col].sum().unstack(fill_value=0)
        if not pl_grp.empty:
            pl_grp = pl_grp.reindex(columns=periods, fill_value=0).sort_index(axis=1).sort_index(level=0)
            pl_item_df_for_ni_calc = pl_grp.reset_index(level='_Assigned_Category_').drop(columns='_Assigned_Category_')
            pl_item_df_for_ni_calc.index.names = pl_idx_names
            
            rev_df = pl_grp[pl_grp.index.get_level_values('_Assigned_Category_') == "Revenue"].droplevel('_Assigned_Category_')
            cogs_df = pl_grp[pl_grp.index.get_level_values('_Assigned_Category_') == "COGS"].droplevel('_Assigned_Category_')
            opex_df = pl_grp[pl_grp.index.get_level_values('_Assigned_Category_') == "Operating Expenses"].droplevel('_Assigned_Category_')
            other_pl_df = pl_grp[pl_grp.index.get_level_values('_Assigned_Category_') == "Other P&L Items"].droplevel('_Assigned_Category_')

            pl_construction_list = []
            sum_rev = rev_df.sum(axis=0).reindex(periods, fill_value=0) if not rev_df.empty else pd.Series(0, index=periods, dtype=float)
            sum_cogs = cogs_df.sum(axis=0).reindex(periods, fill_value=0) if not cogs_df.empty else pd.Series(0, index=periods, dtype=float)
            sum_opex = opex_df.sum(axis=0).reindex(periods, fill_value=0) if not opex_df.empty else pd.Series(0, index=periods, dtype=float)
            sum_other_pl = other_pl_df.sum(axis=0).reindex(periods, fill_value=0) if not other_pl_df.empty else pd.Series(0, index=periods, dtype=float)

            if not rev_df.empty:
                pl_construction_list.append(rev_df)
                pl_construction_list.append(pd.DataFrame([sum_rev.values], index=pd.MultiIndex.from_tuples([(REV_TOTAL_ID, "Total Revenue")], names=pl_idx_names), columns=sum_rev.index))
            if not cogs_df.empty:
                pl_construction_list.append(cogs_df)
                pl_construction_list.append(pd.DataFrame([sum_cogs.values], index=pd.MultiIndex.from_tuples([(COGS_TOTAL_ID, "Total COGS")], names=pl_idx_names), columns=sum_cogs.index))
            if not rev_df.empty or not cogs_df.empty:
                gp_series = sum_rev + sum_cogs 
                gp_row = pd.DataFrame([gp_series.values], index=pd.MultiIndex.from_tuples([(GP_ID_CONST, gp_name)], names=pl_idx_names), columns=gp_series.index)
                pl_construction_list.append(gp_row)
            if not opex_df.empty:
                pl_construction_list.append(opex_df)
                pl_construction_list.append(pd.DataFrame([sum_opex.values], index=pd.MultiIndex.from_tuples([(OPEX_TOTAL_ID, "Total Operating Expenses")], names=pl_idx_names), columns=sum_opex.index))
            if not other_pl_df.empty:
                pl_construction_list.append(other_pl_df)
                pl_construction_list.append(pd.DataFrame([sum_other_pl.values], index=pd.MultiIndex.from_tuples([(OTHER_PL_TOTAL_ID, "Total Other P&L Items")], names=pl_idx_names), columns=sum_other_pl.index))

            pl_disp_df = pd.concat(pl_construction_list) if pl_construction_list else pd.DataFrame(index=empty_pl_idx, columns=periods)
            ni_series_recalc = pl_item_df_for_ni_calc.sum(axis=0) if not pl_item_df_for_ni_calc.empty else pd.Series(0, index=periods, dtype=float)
            ni_row_cols = ni_series_recalc.index if not ni_series_recalc.empty else periods 
            ni_row = pd.DataFrame([ni_series_recalc.values.copy()], index=pd.MultiIndex.from_tuples([(NI_ID_CONST, ni_name)], names=pl_idx_names), columns=ni_row_cols)
            if not pl_disp_df.empty or not ni_row.empty: 
                pl_disp_df = pd.concat([pl_disp_df, ni_row] if not pl_disp_df.empty else [ni_row])
            if not pl_disp_df.empty: pl_disp_df = pl_disp_df.reindex(columns=periods, fill_value=0).sort_index(axis=1)
        else: 
            pl_disp_df = pd.DataFrame(index=empty_pl_idx, columns=periods)
            ni_series_recalc = pd.Series(0, index=periods, dtype=float) if periods else pd.Series(dtype=float)
    else: 
        pl_disp_df = pd.DataFrame(index=empty_pl_idx, columns=periods)
        ni_series_recalc = pd.Series(0, index=periods, dtype=float) if periods else pd.Series(dtype=float)

    bs_idx_names = ['Account ID', 'Account Name']; empty_bs_idx = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=bs_idx_names)
    bs_period_activity_df = pd.DataFrame(index=empty_bs_idx, columns=periods) 
    re_id, re_name = "RECALC", "Calculated Retained Earnings"
    ast_id, ast_name, lia_id, lia_name, eq_id, eq_name = "ASSETS_TOTAL", "Total Assets", "LIAB_TOTAL", "Total Liabilities", "EQUITY_TOTAL", "Total Equity"
    
    if not bs_data_activity.empty:
        bs_grp_act = bs_data_activity.groupby(['_Standard_Account_ID_', '_Standard_Mapping_', '_YearMonth_'])[amount_col].sum().unstack(fill_value=0)
        if not bs_grp_act.empty:
            bs_period_activity_df = bs_grp_act.reindex(columns=periods, fill_value=0).sort_index(axis=1).sort_index(level=0)
            bs_period_activity_df.index.names = bs_idx_names
            
    bs_balances_df = bs_period_activity_df.copy() 
    if roll_bs_overall_flag and not bs_balances_df.empty: 
        st.caption("Rolling BS balances (Cumulative across all years from periodic activity)...")
        sorted_period_columns = sorted(bs_balances_df.columns.tolist())
        bs_balances_df = bs_balances_df[sorted_period_columns].cumsum(axis=1)
    elif not bs_balances_df.empty:
        st.caption("BS balances NOT rolled cumulatively. Displaying periodic activity/balances as is from JE sums.")
    
    ast_items, lia_items, eq_base_items = (pd.DataFrame(index=empty_bs_idx, columns=periods) for _ in range(3))
    if not bs_balances_df.empty and isinstance(bs_balances_df.index, pd.MultiIndex):
        def get_aid_for_cat(val_tuple_or_str, multi_idx_obj):
            return str(val_tuple_or_str[0]) if isinstance(val_tuple_or_str, tuple) and val_tuple_or_str else str(val_tuple_or_str)

        ast_items = bs_balances_df[bs_balances_df.index.map(lambda x: utils.assign_category_using_rules(get_aid_for_cat(x, bs_balances_df.index), use_custom_ranges_flag, parsed_gl_ranges_from_state, utils.get_prefix_based_category) == "Assets")]
        lia_items = bs_balances_df[bs_balances_df.index.map(lambda x: utils.assign_category_using_rules(get_aid_for_cat(x, bs_balances_df.index), use_custom_ranges_flag, parsed_gl_ranges_from_state, utils.get_prefix_based_category) == "Liabilities")]
        eq_base_items = bs_balances_df[
            (bs_balances_df.index.map(lambda x: utils.assign_category_using_rules(get_aid_for_cat(x, bs_balances_df.index), use_custom_ranges_flag, parsed_gl_ranges_from_state, utils.get_prefix_based_category) == "Equity")) &
            (bs_balances_df.index.map(lambda x: get_aid_for_cat(x, bs_balances_df.index) != re_id))
        ]
            
    sum_ast = ast_items.sum(axis=0).reindex(periods, fill_value=0) if not ast_items.empty else pd.Series(0, index=periods, dtype=float)
    sum_lia = lia_items.sum(axis=0).reindex(periods, fill_value=0) if not lia_items.empty else pd.Series(0, index=periods, dtype=float)
    sum_eq_base = eq_base_items.sum(axis=0).reindex(periods, fill_value=0) if not eq_base_items.empty else pd.Series(0, index=periods, dtype=float)
    
    recalc_eq_row_df = pd.DataFrame(index=pd.MultiIndex.from_tuples([(re_id, re_name)], names=bs_idx_names), columns=periods).fillna(0.0)
    cumulative_re_series = pd.Series(0, index=periods, dtype=float)

    if roll_equity_flag and not ni_series_recalc.empty:
        if not periods: st.warning("Cannot roll Retained Earnings: no periods defined.")
        else:
            st.caption("Rolling Net Income into Equity (Cumulative across all years)...")
            ni_for_re = ni_series_recalc.reindex(periods, fill_value=0.0)
            
            cumulative_economic_profit = 0.0 
            re_values_for_series = {}

            for period_col in periods: 
                period_net_income_raw = ni_for_re.get(period_col, 0.0)
                
                current_period_economic_profit = period_net_income_raw
                if pnl_sign_convention == "RevNeg_ExpPos": 
                    current_period_economic_profit *= -1 
                
                cumulative_economic_profit += current_period_economic_profit
                re_values_for_series[period_col] = cumulative_economic_profit * -1 # Store as negative for credit balance in A+L+E=0
            
            cumulative_re_series = pd.Series(re_values_for_series).reindex(periods, fill_value=0.0)
            if (re_id, re_name) in recalc_eq_row_df.index:
                recalc_eq_row_df.loc[(re_id, re_name), :] = cumulative_re_series.values
    elif not roll_equity_flag and (re_id, re_name) in bs_balances_df.index: 
        cumulative_re_series = bs_balances_df.loc[(re_id, re_name)].reindex(periods, fill_value=0)
        if (re_id, re_name) in recalc_eq_row_df.index:
             recalc_eq_row_df.loc[(re_id, re_name), :] = cumulative_re_series.values

    bs_list = []
    if not ast_items.empty: bs_list.append(ast_items)
    bs_list.append(pd.DataFrame([sum_ast.values], index=pd.MultiIndex.from_tuples([(ast_id, ast_name)], names=bs_idx_names), columns=periods))
    if not lia_items.empty: bs_list.append(lia_items)
    bs_list.append(pd.DataFrame([sum_lia.values], index=pd.MultiIndex.from_tuples([(lia_id, lia_name)], names=bs_idx_names), columns=periods))
    if not eq_base_items.empty: bs_list.append(eq_base_items) 
    
    # Corrected: Add the RE row if it actually contains data
    if not recalc_eq_row_df.empty and np.any(recalc_eq_row_df.sum(axis=1).fillna(0) != 0):
         bs_list.append(recalc_eq_row_df)
    
    cur_total_eq_series = (sum_eq_base + cumulative_re_series).reindex(periods, fill_value=0)
    bs_list.append(pd.DataFrame([cur_total_eq_series.values], index=pd.MultiIndex.from_tuples([(eq_id, eq_name)], names=bs_idx_names), columns=periods))
    
    bs_final_disp_df = pd.concat(bs_list) if bs_list else pd.DataFrame(index=empty_bs_idx, columns=periods)
    if not bs_final_disp_df.empty: bs_final_disp_df = bs_final_disp_df.reindex(columns=periods, fill_value=0).sort_index(axis=1)
    
    bs_audit_df, bal_check_series = pd.DataFrame(), pd.Series(dtype=float)
    if not bs_final_disp_df.empty and periods:
        bal_check_series = sum_ast + sum_lia + cur_total_eq_series 
        bs_audit_df = pd.DataFrame({ "Assets": sum_ast, "Liabilities": sum_lia, "Equity": cur_total_eq_series, "Audit (A+L+E=0)": bal_check_series }).T.reindex(columns=periods, fill_value=0)
    else: bs_audit_df = pd.DataFrame(columns=["Assets", "Liabilities", "Equity", "Audit (A+L+E=0)"]).T
    
    return pl_disp_df, bs_final_disp_df, ni_series_recalc, bs_audit_df, bal_check_series, uncat_sum_df


# --- Config Helper Functions & UI Logic --- (Starting from here, it's the standard UI)
def get_config_to_export():
    config_data = [{"SettingKey": "app_config_version", "SettingValue": "1.0"}, {"SettingKey": "selected_je_sheets", "SettingValue": ",".join(st.session_state.get('selected_je_sheets', []))},
                   {"SettingKey": "debit_column", "SettingValue": st.session_state.get('debit_column_selection', "") or ""}, {"SettingKey": "credit_column", "SettingValue": st.session_state.get('credit_column_selection', "") or ""}]
    core_maps = st.session_state.get('column_mappings', {})
    config_data.extend([{"SettingKey": "mapping_account_id", "SettingValue": core_maps.get("account_id", "") or ""}, {"SettingKey": "mapping_account_name", "SettingValue": core_maps.get("mapping", "") or ""},
                        {"SettingKey": "mapping_amount", "SettingValue": core_maps.get("amount", "") or ""}, {"SettingKey": "mapping_date", "SettingValue": core_maps.get("date", "") or ""}])
    config_data.extend([{"SettingKey": "pnl_sign_convention", "SettingValue": st.session_state.get('pnl_sign_convention_selection', "RevNeg_ExpPos")},
                        {"SettingKey": "roll_bs_balances_flag", "SettingValue": str(st.session_state.get('roll_bs_selection', False))}, {"SettingKey": "roll_equity_flag", "SettingValue": str(st.session_state.get('roll_equity_selection', False))}])
    config_data.append({"SettingKey": "use_custom_ranges_flag", "SettingValue": str(st.session_state.get('use_custom_ranges', False))})
    custom_ranges = st.session_state.get('custom_gl_range_inputs', {})
    for category in utils.RANGE_INPUT_CATEGORIES: 
        config_data.append({"SettingKey": f"custom_range_{category.replace(' ', '_').replace('/', '_')}", "SettingValue": custom_ranges.get(category, "")})
    return pd.DataFrame(config_data)

def trigger_custom_range_processing():
    parsed_ranges = {} ; all_ranges_valid_overall = True
    range_inputs_to_process = st.session_state.get('custom_gl_range_inputs', {cat: "" for cat in utils.RANGE_INPUT_CATEGORIES})
    for category, range_str in range_inputs_to_process.items():
        if range_str and isinstance(range_str, str) and range_str.strip():
            parsed_current_cat = utils.parse_gl_ranges(range_str) 
            if parsed_current_cat is None: 
                all_ranges_valid_overall = False; 
                st.warning(f"Error parsing GL range for '{category}': '{range_str}'. Check format (e.g., 1000-1999, 2500). Reverting to previous for this category.")
                parsed_ranges[category] = st.session_state.get('parsed_gl_ranges', {}).get(category, []) 
            else: parsed_ranges[category] = parsed_current_cat
        else: parsed_ranges[category] = [] 
    st.session_state.parsed_gl_ranges = parsed_ranges
    st.session_state.use_custom_ranges = any(bool(v_list) for v_list in parsed_ranges.values() if isinstance(v_list, list) and v_list)
    return all_ranges_valid_overall

def load_config_data_from_excel(excel_file_buffer, sheet_name_const): 
    try:
        excel_file_buffer.seek(0); df_config = pd.read_excel(excel_file_buffer, sheet_name=sheet_name_const)
        return pd.Series(df_config['SettingValue'].astype(str).replace('nan', '').values, index=df_config['SettingKey']).to_dict()
    except Exception as e: st.warning(f"Could not read/parse config sheet '{sheet_name_const}': {e}"); return None

def apply_config_to_session_state(config_settings): 
    if not config_settings: return
    def get_config_val(key, default=""): val = str(config_settings.get(key, default)).strip(); return "" if val.lower() == 'nan' else val
    st.session_state.selected_je_sheets = [s.strip() for s in get_config_val('selected_je_sheets').split(',') if s.strip()] if get_config_val('selected_je_sheets') else []
    st.session_state.debit_column_selection = get_config_val('debit_column') or None
    st.session_state.credit_column_selection = get_config_val('credit_column') or None
    if 'column_mappings' not in st.session_state: st.session_state.column_mappings = {} 
    st.session_state.column_mappings["account_id"] = get_config_val('mapping_account_id') or None
    st.session_state.column_mappings["mapping"] = get_config_val('mapping_account_name') or None
    st.session_state.column_mappings["amount"] = get_config_val('mapping_amount') or None
    st.session_state.column_mappings["date"] = get_config_val('mapping_date') or None
    st.session_state.pnl_sign_convention_selection = get_config_val('pnl_sign_convention', "RevNeg_ExpPos")
    st.session_state.roll_bs_selection = get_config_val('roll_bs_balances_flag', "False").lower() == 'true'
    st.session_state.roll_equity_selection = get_config_val('roll_equity_flag', "False").lower() == 'true'
    temp_custom_ranges = {cat: get_config_val(f"custom_range_{cat.replace(' ', '_').replace('/', '_')}", "") for cat in utils.RANGE_INPUT_CATEGORIES}
    st.session_state.custom_gl_range_inputs = temp_custom_ranges
    st.session_state.use_custom_ranges = get_config_val('use_custom_ranges_flag', "False").lower() == 'true'
    trigger_custom_range_processing()

_initial_session_state_defaults = {
    'sheet_names': [], 'selected_je_sheets': [], 'compiled_je_df': None,
    'column_mappings': {"account_id": None, "mapping": None, "amount": None, "date": None },
    'custom_gl_range_inputs': {cat: "" for cat in utils.RANGE_INPUT_CATEGORIES}, 'parsed_gl_ranges': {},
    'use_custom_ranges': False, 'show_statements': False, 'debit_column_selection': None,
    'credit_column_selection': None, 'roll_equity_selection': False, 'roll_bs_selection': False,
    'pnl_sign_convention_selection': "RevNeg_ExpPos", 'active_pl_flat_df': None,
    'active_bs_flat_df': None, 'uploaded_data_ready': False,
    'pl_wide_df_processed': None, 'bs_wide_df_processed': None,
    'bs_audit_df_processed_display': None, 'internal_bs_check_series_logic': None,
    'uncategorized_accounts_summary': None
}
for key, default_value in _initial_session_state_defaults.items():
    if key not in st.session_state: st.session_state[key] = default_value
if 'uploaded_excel_file_info' not in st.session_state:
    st.session_state.uploaded_excel_file_info = None

st.subheader("1. Upload Excel File")
uploaded_file_widget = st.file_uploader("Choose an Excel file (.xlsx or .xls)", type=["xlsx", "xls"], key="file_uploader_data_upload_main_key_v9_fix")

if uploaded_file_widget:
    is_new_file_upload_event = False
    current_file_info_state = st.session_state.get('uploaded_excel_file_info')
    if current_file_info_state is None or \
       current_file_info_state.get('name') != uploaded_file_widget.name or \
       current_file_info_state.get('size') != uploaded_file_widget.size:
        is_new_file_upload_event = True

    if is_new_file_upload_event:
        st.session_state.uploaded_excel_file_info = {
            'name': uploaded_file_widget.name, 'size': uploaded_file_widget.size,
            'data': BytesIO(uploaded_file_widget.getvalue())
        }
        st.success(f"Uploaded '{uploaded_file_widget.name}' successfully.")
        keys_to_reset_on_new_file = list(_initial_session_state_defaults.keys()) 
        for key_to_reset in keys_to_reset_on_new_file:
            if key_to_reset not in ['llm_provider', 'chosen_ollama_model', 'openai_api_key', 'openai_model_name', 'azure_endpoint', 'azure_api_key', 'azure_deployment_name', 'azure_api_version', 'llm_context_limit']:
                 st.session_state[key_to_reset] = _initial_session_state_defaults[key_to_reset]
        st.session_state.column_mappings = {"account_id": None, "mapping": None, "amount": None, "date": None }
        st.session_state.custom_gl_range_inputs = {cat: "" for cat in utils.RANGE_INPUT_CATEGORIES}
        st.session_state.parsed_gl_ranges = {}
        st.session_state.show_statements = False
        st.session_state.uploaded_data_ready = False
    
    excel_file_buffer_is_valid = False
    if st.session_state.uploaded_excel_file_info and 'data' in st.session_state.uploaded_excel_file_info:
        excel_file_buffer_instance = st.session_state.uploaded_excel_file_info['data']
        excel_file_buffer_instance.seek(0) 
        excel_file_buffer_is_valid = True

    if excel_file_buffer_is_valid and is_new_file_upload_event: 
        try:
            excel_file_obj = pd.ExcelFile(excel_file_buffer_instance)
            st.session_state.sheet_names = excel_file_obj.sheet_names
            excel_file_buffer_instance.seek(0) 
            if CONFIG_SHEET_NAME in st.session_state.sheet_names:
                st.info(f"Configuration sheet '{CONFIG_SHEET_NAME}' detected. Attempting to load parameters...")
                loaded_settings = load_config_data_from_excel(excel_file_buffer_instance, CONFIG_SHEET_NAME)
                excel_file_buffer_instance.seek(0) 
                if loaded_settings:
                    apply_config_to_session_state(loaded_settings) 
                    st.success("Configuration parameters loaded from Excel sheet!")
                else:
                    st.warning("Could not load settings from config sheet. Please configure manually.")
            else: st.info("No configuration sheet found. Please configure selections manually below.")
            st.rerun() 
        except Exception as e:
            st.error(f"Could not process Excel file content (sheets/config): {e}")
            st.session_state.uploaded_excel_file_info = None; st.session_state.sheet_names = []
    elif not excel_file_buffer_is_valid and uploaded_file_widget:
         st.error("File data is missing or corrupted after upload. Please re-upload.")

if st.session_state.get('uploaded_excel_file_info') and st.session_state.get('sheet_names'):
    st.subheader("2. Select Sheets for Journal Entries (JEs)")
    current_je_sheets_sel = st.session_state.get('selected_je_sheets', [])
    if isinstance(current_je_sheets_sel, str) and current_je_sheets_sel:
        current_je_sheets_sel = [s.strip() for s in current_je_sheets_sel.split(',')]
    elif not isinstance(current_je_sheets_sel, list):
        current_je_sheets_sel = []
    valid_current_je_sheets_sel = [s for s in current_je_sheets_sel if s in st.session_state.sheet_names]

    new_je_sheets_selection = st.multiselect( 
        "Which sheet(s) contain your JE data?", 
        options=st.session_state.sheet_names, 
        default=valid_current_je_sheets_sel, 
        key="selected_je_sheets_widget_s2_v9_fix" 
    )
    if new_je_sheets_selection != valid_current_je_sheets_sel: 
        st.session_state.selected_je_sheets = new_je_sheets_selection
        st.session_state.compiled_je_df = None 
        st.session_state.show_statements = False; st.session_state.uploaded_data_ready = False
        st.rerun()

    if st.session_state.selected_je_sheets:
        if st.button("Compile Selected JE Sheets", key="compile_je_sheets_button_s2_v9_fix"):
            st.session_state.compiled_je_df = None; st.session_state.show_statements = False
            st.session_state.active_pl_flat_df = None; st.session_state.active_bs_flat_df = None
            st.session_state.uploaded_data_ready = False; all_dfs_s2 = []
            
            if st.session_state.uploaded_excel_file_info and 'data' in st.session_state.uploaded_excel_file_info:
                excel_buf_s2 = st.session_state.uploaded_excel_file_info['data']
                for sheet_name_s2 in st.session_state.selected_je_sheets:
                    excel_buf_s2.seek(0) 
                    sheet_df_s2 = safe_read_excel_sheet(excel_buf_s2, sheet_name_s2)
                    if sheet_df_s2 is not None: all_dfs_s2.append(sheet_df_s2)
            
            if all_dfs_s2:
                try:
                    st.session_state.compiled_je_df = pd.concat(all_dfs_s2, ignore_index=True)
                    st.success(f"Successfully compiled {len(all_dfs_s2)} JE sheet(s). Total rows: {len(st.session_state.compiled_je_df)}")
                    if st.session_state.column_mappings.get("amount") == CALCULATED_AMOUNT_COL_NAME and \
                       st.session_state.debit_column_selection and st.session_state.credit_column_selection:
                        try_create_total_amount_column(st.session_state.debit_column_selection, st.session_state.credit_column_selection)
                except Exception as e: st.error(f"Error during JE sheet compilation: {e}")
            else: st.warning("No dataframes were successfully read from the selected sheets.")
            st.rerun()
            
if st.session_state.get('compiled_je_df') is not None and not st.session_state.compiled_je_df.empty:
    st.subheader("Preview of Compiled JE Data (First 5 Rows)")
    st.dataframe(st.session_state.compiled_je_df.head(), height=200, use_container_width=True)
    
    expand_dc_s2 = st.session_state.get('debit_column_selection') is not None or st.session_state.get('credit_column_selection') is not None
    with st.expander("Optional: Handle Separate Debit/Credit Columns", expanded=expand_dc_s2):
        df_cols_for_dc_exp_s2 = [""] + list(st.session_state.compiled_je_df.columns)
        current_debit_sel_exp_s2 = st.session_state.get('debit_column_selection')
        debit_idx_exp_s2 = df_cols_for_dc_exp_s2.index(current_debit_sel_exp_s2) if current_debit_sel_exp_s2 and current_debit_sel_exp_s2 in df_cols_for_dc_exp_s2 else 0
        new_debit_col_sel_exp_s2 = st.selectbox("Select Debit Column:", options=df_cols_for_dc_exp_s2, index=debit_idx_exp_s2, key="debit_col_exp_widget_s2_v9_fix")
        if new_debit_col_sel_exp_s2 != current_debit_sel_exp_s2: st.session_state.debit_column_selection = new_debit_col_sel_exp_s2 if new_debit_col_sel_exp_s2 else None; st.rerun()
        
        current_credit_sel_exp_s2 = st.session_state.get('credit_column_selection')
        credit_idx_exp_s2 = df_cols_for_dc_exp_s2.index(current_credit_sel_exp_s2) if current_credit_sel_exp_s2 and current_credit_sel_exp_s2 in df_cols_for_dc_exp_s2 else 0
        new_credit_col_sel_exp_s2 = st.selectbox("Select Credit Column:", options=df_cols_for_dc_exp_s2, index=credit_idx_exp_s2, key="credit_col_exp_widget_s2_v9_fix")
        if new_credit_col_sel_exp_s2 != current_credit_sel_exp_s2: st.session_state.credit_column_selection = new_credit_col_sel_exp_s2 if new_credit_col_sel_exp_s2 else None; st.rerun()
        
        if st.button(f"Create or Verify '{CALCULATED_AMOUNT_COL_NAME}'", key="create_total_amount_exp_button_s2_v9_fix"):
            if try_create_total_amount_column(st.session_state.debit_column_selection, st.session_state.credit_column_selection):
                st.success(f"'{CALCULATED_AMOUNT_COL_NAME}' handled. Map in Step 3 if needed.")
                if st.session_state.column_mappings.get("amount") != CALCULATED_AMOUNT_COL_NAME : 
                    st.session_state.column_mappings["amount"] = CALCULATED_AMOUNT_COL_NAME
                st.rerun()
    st.markdown("---")

    st.subheader("3. Map Core Data Columns & P&L Convention")
    pnl_conv_map_s3 = { "RevNeg_ExpPos": "Revenue Negative (Credits) / COGS & Expenses Positive (Debits)", "RevPos_ExpNeg": "Revenue Positive / COGS & Expenses Negative"}
    pnl_conv_opts_s3 = list(pnl_conv_map_s3.keys())
    current_pnl_convention_idx_s3 = pnl_conv_opts_s3.index(st.session_state.pnl_sign_convention_selection) if st.session_state.pnl_sign_convention_selection in pnl_conv_opts_s3 else 0
    
    new_pnl_conv_s3 = st.radio("P&L Value Representation:", options=pnl_conv_opts_s3, index=current_pnl_convention_idx_s3, format_func=lambda x: pnl_conv_map_s3[x], key="pnl_radio_s3_v9_fix", horizontal=True)
    if new_pnl_conv_s3 != st.session_state.pnl_sign_convention_selection: st.session_state.pnl_sign_convention_selection = new_pnl_conv_s3; st.rerun()
    
    df_cols_map_s3 = [""] + list(st.session_state.compiled_je_df.columns) 
    map_state_s3 = st.session_state.get('column_mappings', {})
    map_c1_s3, map_c2_s3 = st.columns(2); map_changed_s3 = False
    
    with map_c1_s3: 
        def get_idx_s3(k): v = map_state_s3.get(k); return df_cols_map_s3.index(v) if v and v in df_cols_map_s3 else 0
        
        # Account ID
        current_mapped_aid = map_state_s3.get("account_id")
        new_aid_selection = st.selectbox("Account ID Col:", df_cols_map_s3, get_idx_s3("account_id"), key="map_aid_s3_v9_fix")
        effective_new_aid = new_aid_selection if new_aid_selection else None
        if effective_new_aid != current_mapped_aid:
            map_state_s3["account_id"] = effective_new_aid; map_changed_s3 = True
        
        # Amount
        current_mapped_amount = map_state_s3.get("amount")
        is_calc_amt = (CALCULATED_AMOUNT_COL_NAME in df_cols_map_s3 and current_mapped_amount == CALCULATED_AMOUNT_COL_NAME)
        if is_calc_amt: 
            st.write(f"**Amount Column:** `{CALCULATED_AMOUNT_COL_NAME}` (auto-created)")
            if st.button(f"Change Amount Column", key="chg_calc_amt_s3_v9_fix"): 
                map_state_s3["amount"] = None; map_changed_s3 = True
        else: 
            new_amt_selection = st.selectbox("Amount Col:", df_cols_map_s3, get_idx_s3("amount"), key="map_amt_s3_v9_fix")
            effective_new_amt = new_amt_selection if new_amt_selection else None
            if effective_new_amt != current_mapped_amount:
                map_state_s3["amount"] = effective_new_amt; map_changed_s3 = True
                
    with map_c2_s3: 
        # Account Mapping/Desc
        current_mapped_desc = map_state_s3.get("mapping")
        new_desc_selection = st.selectbox("Account Mapping/Desc Col:", df_cols_map_s3, get_idx_s3("mapping"), key="map_map_s3_v9_fix")
        effective_new_desc = new_desc_selection if new_desc_selection else None
        if effective_new_desc != current_mapped_desc:
            map_state_s3["mapping"] = effective_new_desc; map_changed_s3 = True
        
        # Date
        current_mapped_date = map_state_s3.get("date")
        new_date_selection = st.selectbox("Date Col:", df_cols_map_s3, get_idx_s3("date"), key="map_date_s3_v9_fix")
        effective_new_date = new_date_selection if new_date_selection else None
        if effective_new_date != current_mapped_date:
            map_state_s3["date"] = effective_new_date; map_changed_s3 = True
            
    if map_changed_s3: 
        st.session_state.column_mappings = map_state_s3
        st.session_state.show_statements = False # Force regeneration if mappings change
        st.session_state.uploaded_data_ready = False
        st.rerun()
    
    st.markdown("---"); st.subheader("4. Define Financial Statement Structure (GL Groupings)")
    st.info("Default: Assets (1xxxx), Liabilities (2xxxx), Equity (3xxxx), Revenue (4xxxx), COGS (5xxxx), OpEx (6xxxx, 7xxxx), Other P&L (8xxxx, 9xxxx). Refine below if needed.")
    with st.expander("Refine GL Account Grouping Ranges (Optional)", expanded=st.session_state.get('use_custom_ranges', False)):
        temp_ranges_s4 = {}
        for cat_s4 in utils.RANGE_INPUT_CATEGORIES:
            temp_ranges_s4[cat_s4] = st.text_input(f"Ranges for {cat_s4}:", value=st.session_state.custom_gl_range_inputs.get(cat_s4,""), placeholder=f"Default: {DEFAULT_PREFIX_MAP.get(cat_s4, 'N/A')}", key=f"range_{cat_s4}_s4_v9_fix")
        if st.button("Apply Custom Ranges", key="apply_ranges_s4_v9_fix"):
            st.session_state.custom_gl_range_inputs = temp_ranges_s4
            if trigger_custom_range_processing(): st.success("Custom GL ranges processed.")
            else: st.error("Custom ranges had errors and were not fully applied. Please correct them.")
            st.session_state.show_statements = False; st.session_state.uploaded_data_ready = False
            st.rerun()
    if st.session_state.use_custom_ranges and any(bool(v) for v in st.session_state.get('parsed_gl_ranges', {}).values() if v): 
        st.success("Custom GL grouping ranges are active where specified.")
    
    st.markdown("---"); st.subheader("5. Final Options & Generation")
    roll_bs_s5 = st.checkbox("Roll BS balances (Cumulative YTD across all years)", value=st.session_state.get('roll_bs_selection', False), key="roll_bs_s5_v9_fix")
    if roll_bs_s5 != st.session_state.roll_bs_selection: 
        st.session_state.roll_bs_selection = roll_bs_s5
        st.session_state.show_statements = False; st.session_state.uploaded_data_ready = False
        st.rerun()
    roll_eq_s5 = st.checkbox("Roll Net Income into Equity (Cumulative across all years)", value=st.session_state.get('roll_equity_selection', False), key="roll_eq_s5_v9_fix")
    if roll_eq_s5 != st.session_state.roll_equity_selection: 
        st.session_state.roll_equity_selection = roll_eq_s5
        st.session_state.show_statements = False; st.session_state.uploaded_data_ready = False
        st.rerun()
    
    if st.button("Generate Financial Statements", key="generate_statements_button_s5_v9_fix", type="primary"):
        # Validation before processing
        valid_mappings = True
        for core_col_key in ["account_id", "mapping", "amount", "date"]:
            if not st.session_state.column_mappings.get(core_col_key):
                st.error(f"Core column for '{core_col_key.replace('_', ' ').title()}' is not mapped. Please complete mappings in Step 3.")
                valid_mappings = False
        
        chosen_amount_col_s5 = st.session_state.column_mappings.get("amount")
        if valid_mappings and (st.session_state.compiled_je_df is None or chosen_amount_col_s5 not in st.session_state.compiled_je_df.columns):
             st.error(f"Amount column ('{chosen_amount_col_s5}') invalid/not found in compiled JE data. Please check Step 2 & 3."); valid_mappings = False
        
        if valid_mappings:
            with st.spinner("Processing data and generating statements..."):
                pl_df_final, bs_df_final, _, bs_audit_table_for_display, internal_bs_check_series, uncat_summary = process_financial_data(
                    st.session_state.compiled_je_df, st.session_state.column_mappings["account_id"],
                    st.session_state.column_mappings["mapping"], chosen_amount_col_s5, st.session_state.column_mappings["date"],
                    st.session_state.use_custom_ranges, st.session_state.parsed_gl_ranges,   
                    st.session_state.roll_equity_selection, st.session_state.roll_bs_selection, st.session_state.pnl_sign_convention_selection)
            
            st.session_state.pl_wide_df_processed = pl_df_final
            st.session_state.bs_wide_df_processed = bs_df_final 
            st.session_state.bs_audit_df_processed_display = bs_audit_table_for_display
            st.session_state.internal_bs_check_series_logic = internal_bs_check_series
            st.session_state.uncategorized_accounts_summary = uncat_summary
            st.session_state.show_statements = True # Mark statements as ready to be shown
            
            pl_prep_successful, bs_prep_successful = False, False
            st.session_state.active_pl_flat_df, st.session_state.active_bs_flat_df = None, None

            if pl_df_final is not None and not pl_df_final.empty:
                try:
                    pl_melt_df = pl_df_final.reset_index()
                    if 'Account ID' not in pl_melt_df.columns and 'level_0' in pl_melt_df.columns: pl_melt_df = pl_melt_df.rename(columns={'level_0': 'Account ID'})
                    if 'Account Name' not in pl_melt_df.columns and 'level_1' in pl_melt_df.columns: pl_melt_df = pl_melt_df.rename(columns={'level_1': 'Account Name'})
                    if 'Account ID' not in pl_melt_df.columns: raise ValueError("P&L: 'Account ID' missing after reset/rename.")
                    pl_melt_df['_Account_Type_'] = pl_melt_df['Account ID'].apply(lambda x: "Calculated Grouping" if str(x) in PL_CALCULATED_IDS else "Individual Account")
                    id_vars_pl = ['Account ID', 'Account Name', '_Account_Type_']
                    valid_id_vars_pl = [v for v in id_vars_pl if v in pl_melt_df.columns]
                    if not ('Account ID' in valid_id_vars_pl and 'Account Name' in valid_id_vars_pl): raise ValueError(f"P&L melt needs 'Account ID' & 'Name'. Valid: {valid_id_vars_pl}")
                    temp_active_pl_flat = pd.melt(pl_melt_df, id_vars=valid_id_vars_pl, var_name='Period', value_name='Amount')
                    temp_active_pl_flat['Period_dt'] = pd.to_datetime(temp_active_pl_flat['Period'] + '-01', errors='coerce') 
                    st.session_state.active_pl_flat_df = temp_active_pl_flat; pl_prep_successful = True
                except Exception as e: st.error(f"Error prepping flat P&L: {e}"); st.session_state.active_pl_flat_df = None; pl_prep_successful = False
            elif pl_df_final is not None: st.session_state.active_pl_flat_df = pd.DataFrame(); pl_prep_successful = True
            
            if bs_df_final is not None and not bs_df_final.empty:
                try:
                    bs_melt_df = bs_df_final.reset_index()
                    if 'Account ID' not in bs_melt_df.columns and 'level_0' in bs_melt_df.columns: bs_melt_df = bs_melt_df.rename(columns={'level_0': 'Account ID'})
                    if 'Account Name' not in bs_melt_df.columns and 'level_1' in bs_melt_df.columns: bs_melt_df = bs_melt_df.rename(columns={'level_1': 'Account Name'})
                    if 'Account ID' not in bs_melt_df.columns: raise ValueError("BS: 'Account ID' missing after reset/rename.")
                    bs_melt_df['_Account_Type_'] = bs_melt_df['Account ID'].apply(lambda x: "Calculated Grouping" if str(x) in BS_CALCULATED_IDS else "Individual Account")
                    id_vars_bs = ['Account ID', 'Account Name', '_Account_Type_']
                    valid_id_vars_bs = [v for v in id_vars_bs if v in bs_melt_df.columns]
                    if not ('Account ID' in valid_id_vars_bs and 'Account Name' in valid_id_vars_bs): raise ValueError(f"BS melt needs 'Account ID' & 'Name'. Valid: {valid_id_vars_bs}")
                    temp_active_bs_flat = pd.melt(bs_melt_df, id_vars=valid_id_vars_bs, var_name='Period', value_name='Amount')
                    temp_active_bs_flat['Period_dt'] = pd.to_datetime(temp_active_bs_flat['Period'] + '-01', errors='coerce')
                    st.session_state.active_bs_flat_df = temp_active_bs_flat; bs_prep_successful = True
                except Exception as e: st.error(f"Error prepping flat BS: {e}"); st.session_state.active_bs_flat_df = None; bs_prep_successful = False
            elif bs_df_final is not None: st.session_state.active_bs_flat_df = pd.DataFrame(); bs_prep_successful = True

            if st.session_state.compiled_je_df is not None and not st.session_state.compiled_je_df.empty and (pl_prep_successful or bs_prep_successful): 
                st.session_state.uploaded_data_ready = True
                if pl_prep_successful and bs_prep_successful: st.success("Statements generated & data fully prepped for visualizations!")
                elif pl_prep_successful: st.warning("P&L prepped for viz. BS prep had issues or was empty.")
                elif bs_prep_successful: st.warning("BS prepped for viz. P&L prep had issues or was empty.")
                else: st.error("Critical errors prepping P&L/BS data for viz, though statements might be generated.")
            else: 
                st.session_state.uploaded_data_ready = False
                if st.session_state.compiled_je_df is None or st.session_state.compiled_je_df.empty:
                    st.error("Compiled JE data missing. Cannot set data as ready.")
                else:
                    st.error("Neither P&L nor BS data could be prepped for visualization, though statements might be generated.")
            st.rerun()
        else: # Not valid mappings
            st.session_state.show_statements = False
            st.session_state.uploaded_data_ready = False


if st.session_state.get('show_statements', False):
    st.markdown("---"); st.subheader("Generated Financial Statements")
    uncategorized_summary_to_display = st.session_state.get('uncategorized_accounts_summary')
    if uncategorized_summary_to_display is not None and not uncategorized_summary_to_display.empty:
        with st.expander("Accounts Not Included in P&L/BS by Current Rules", expanded=True):
            st.caption("These accounts were not categorized by active GL grouping rules and are excluded from P&L/BS totals.")
            st.dataframe(uncategorized_summary_to_display.style.format({"Total Amount": "{:,.0f}"}), use_container_width=True)
        st.markdown("---")
    st.markdown("#### Profit & Loss Statement")
    pl_df_to_display = st.session_state.get('pl_wide_df_processed')
    if pl_df_to_display is not None and not pl_df_to_display.empty:
        actual_pl_calculated_ids_for_style = [str(id_val) for id_val in PL_CALCULATED_IDS] 
        styled_pl_df = style_financial_table(pl_df_to_display, actual_pl_calculated_ids_for_style)
        st.dataframe(styled_pl_df, use_container_width=True)
    elif pl_df_to_display is not None and pl_df_to_display.empty: st.info("P&L Statement is empty (no P&L accounts or activity found).")
    else: st.warning("P&L Statement could not be generated or is not available.")
    
    st.markdown("---"); st.markdown("#### Balance Sheet")
    bs_df_to_display = st.session_state.get('bs_wide_df_processed')
    bs_audit_df_for_user_display = st.session_state.get('bs_audit_df_processed_display') 
    internal_check_series_for_logic = st.session_state.get('internal_bs_check_series_logic')
    if bs_df_to_display is not None and not bs_df_to_display.empty:
        actual_bs_calculated_ids_for_style = [str(id_val) for id_val in BS_CALCULATED_IDS] 
        styled_bs_df = style_financial_table(bs_df_to_display, actual_bs_calculated_ids_for_style)
        st.dataframe(styled_bs_df, use_container_width=True)
        st.markdown("##### Balance Sheet Audit Check") 
        if bs_audit_df_for_user_display is not None and not bs_audit_df_for_user_display.empty:
            st.dataframe(bs_audit_df_for_user_display.fillna(0).style.format("{:,.0f}", na_rep="0"), use_container_width=True)
            if internal_check_series_for_logic is not None and not internal_check_series_for_logic.empty:
                is_balanced = np.allclose(internal_check_series_for_logic.fillna(0), 0, atol=0.51) # Allow small rounding diff
                if is_balanced: st.success("Balance Sheet is balanced (Audit check sums to zero or very close)!")
                else: st.warning(f"Balance Sheet does not balance. Max difference in Audit Check: {internal_check_series_for_logic.abs().max():,.0f}.")
            else: st.warning("Internal balance check data (sum of A+L+E) not available for audit.")
        else: st.info("BS Audit Check data is not available for display (e.g., if BS itself is empty).")
    elif bs_df_to_display is not None and bs_df_to_display.empty: st.info("Balance Sheet Statement is empty (no BS accounts or activity found).")
    else: st.warning("Balance Sheet Statement could not be generated or is not available.")

    st.markdown("---"); st.subheader("Export Page Parameters to Excel") 
    output_buffer_export_final_s5_v3 = BytesIO() 
    config_df_to_export_final_s5_v3 = get_config_to_export()
    if not config_df_to_export_final_s5_v3.empty: 
        instructions_text_final_s5_v3 = """
Instructions for Using the Configuration Sheet (_FINANCIAL_APP_CONFIG_):
1. Purpose: This sheet contains parameters from the 'Data Upload & Validation' page.
2. How to Use: Copy this ENTIRE sheet into your main Excel data file. Name it exactly '_FINANCIAL_APP_CONFIG_'. The app will auto-load these settings on your next upload.
3. Parameters Saved: JE sheets, D/C columns, core mappings, P&L sign convention, custom GL ranges, roll flags.
Note: If data file structures change, re-export or update this config."""
        instruction_lines_final_s5_v3 = [line.strip() for line in instructions_text_final_s5_v3.strip().split('\n') if line.strip()]
        instructions_df_final_s5_v3 = pd.DataFrame(instruction_lines_final_s5_v3, columns=["Instructions"])
        with pd.ExcelWriter(output_buffer_export_final_s5_v3, engine='openpyxl') as writer_final_s5_v3:
            config_df_to_export_final_s5_v3.to_excel(writer_final_s5_v3, sheet_name=CONFIG_SHEET_NAME, index=False)
            instructions_df_final_s5_v3.to_excel(writer_final_s5_v3, sheet_name="Instructions", index=False)
        output_buffer_export_final_s5_v3.seek(0)
        st.download_button(label="Export Parameters to Excel", data=output_buffer_export_final_s5_v3, file_name="financial_app_config_profile.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="export_config_button_final_display_s5_v9_fix")
    else: st.button("Export Parameters to Excel", disabled=True, help="Configure settings on the page first.")