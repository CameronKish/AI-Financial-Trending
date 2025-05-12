import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re 

# --- Page Config & Title ---
st.set_page_config(layout="wide", page_title="Data Upload & Validation")
st.title("📄 Data Upload & Initial Validation")
st.caption("Upload your financial data (Journal Entries) in an Excel file, map the necessary columns, and view a generated P&L and Balance Sheet for validation.")

# --- Predefined Categories for Range Input ---
RANGE_INPUT_CATEGORIES = ["Assets", "Liabilities", "Equity", "Revenue", "COGS", "Operating Expenses", "Other P&L Items"]
DEFAULT_PREFIX_MAP = {
    "Assets": "1xxxx", "Liabilities": "2xxxx", "Equity": "3xxxx",
    "Revenue": "4xxxx", "COGS": "5xxxx", "Operating Expenses": "6xxxx, 7xxxx",
    "Other P&L Items": "8xxxx, 9xxxx"
}
CONFIG_SHEET_NAME = "_FINANCIAL_APP_CONFIG_" 
CALCULATED_AMOUNT_COL_NAME = "Total_Amount_(calc)" # ENSURE THIS IS DEFINED

# --- Helper Functions ---
# ... (All helper functions: safe_read_excel_sheet, style_financial_table, parse_gl_ranges, get_prefix_based_category, assign_category_using_rules, try_create_total_amount_column, process_financial_data, get_config_to_export, trigger_custom_range_processing, load_config_data_from_excel, apply_config_to_session_state should be identical to the previous version where they were correctly defined and working, with try_create_total_amount_column using CALCULATED_AMOUNT_COL_NAME)

def safe_read_excel_sheet(excel_file, sheet_name):
    try: return pd.read_excel(excel_file, sheet_name=sheet_name)
    except Exception as e: st.error(f"Error reading sheet '{sheet_name}': {e}"); return None

def style_financial_table(df_to_style, calculated_row_ids):
    def highlight_specific_rows(row):
        if row.name in calculated_row_ids:
            return ['font-weight: bold; background-color: #e6f3ff'] * len(row)
        return [''] * len(row)
    return df_to_style.fillna(0).style.apply(highlight_specific_rows, axis=1).format("{:,.0f}", na_rep="0")

def parse_gl_ranges(range_str):
    parsed = []
    if not isinstance(range_str, str) or not range_str.strip(): return parsed
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if not part: continue
        if '-' in part:
            start_str, end_str = part.split('-', 1)
            try: parsed.append((int(start_str.strip()), int(end_str.strip())))
            except ValueError: st.warning(f"Invalid range format: '{part}'. Skipping this part."); return None 
        else:
            try: parsed.append((int(part.strip()), int(part.strip())))
            except ValueError: st.warning(f"Invalid number format: '{part}'. Skipping this part."); return None
    return parsed

def get_prefix_based_category(acc_id_str):
    if not isinstance(acc_id_str, str) or not acc_id_str: return "Uncategorized"
    first_char = acc_id_str[0]
    if first_char == '1': return "Assets"
    if first_char == '2': return "Liabilities"
    if first_char == '3': return "Equity"
    if first_char == '4': return "Revenue"
    if first_char == '5': return "COGS"
    if first_char == '6' or first_char == '7': return "Operating Expenses"
    if first_char == '8' or first_char == '9': return "Other P&L Items" 
    return "Uncategorized"

def assign_category_using_rules(acc_id_str, use_custom_flag, custom_parsed_ranges_dict, prefix_logic_func):
    if use_custom_flag and custom_parsed_ranges_dict:
        try:
            match = re.match(r"^\d+", acc_id_str)
            if match:
                acc_id_numeric = int(match.group(0))
                for cat_key_ordered in RANGE_INPUT_CATEGORIES:
                    if cat_key_ordered in custom_parsed_ranges_dict and custom_parsed_ranges_dict[cat_key_ordered]:
                        for r_start, r_end in custom_parsed_ranges_dict[cat_key_ordered]:
                            if r_start <= acc_id_numeric <= r_end:
                                return cat_key_ordered
        except ValueError: pass
    potential_prefix_category = prefix_logic_func(acc_id_str)
    if use_custom_flag and custom_parsed_ranges_dict and \
       potential_prefix_category != "Uncategorized" and \
       custom_parsed_ranges_dict.get(potential_prefix_category):
        return "Uncategorized" 
    return potential_prefix_category

def try_create_total_amount_column(debit_col_name, credit_col_name): # Removed df_to_modify, uses session_state
    if st.session_state.compiled_je_df is None or st.session_state.compiled_je_df.empty:
        return False
    if CALCULATED_AMOUNT_COL_NAME in st.session_state.compiled_je_df.columns:
        return True 
    if debit_col_name and credit_col_name and \
       debit_col_name in st.session_state.compiled_je_df.columns and \
       credit_col_name in st.session_state.compiled_je_df.columns:
        try:
            temp_df = st.session_state.compiled_je_df.copy()
            debit_val = pd.to_numeric(temp_df[debit_col_name], errors='coerce').fillna(0)
            credit_val = pd.to_numeric(temp_df[credit_col_name], errors='coerce').fillna(0)
            temp_df[CALCULATED_AMOUNT_COL_NAME] = debit_val - credit_val
            st.session_state.compiled_je_df = temp_df
            return True
        except Exception as e: st.error(f"Error creating '{CALCULATED_AMOUNT_COL_NAME}' column: {e}"); return False
    else:
        missing_cols = []
        if not debit_col_name: missing_cols.append("Debit column not specified.")
        elif debit_col_name not in st.session_state.compiled_je_df.columns: missing_cols.append(f"Debit column '{debit_col_name}' not found.")
        if not credit_col_name: missing_cols.append("Credit column not specified.")
        elif credit_col_name not in st.session_state.compiled_je_df.columns: missing_cols.append(f"Credit column '{credit_col_name}' not found.")
        if missing_cols: st.warning(f"Could not create '{CALCULATED_AMOUNT_COL_NAME}': " + " ".join(missing_cols))
        return False

def process_financial_data(df, account_id_col, mapping_col, amount_col, date_col,
                           use_custom_ranges_flag, parsed_gl_ranges, 
                           roll_equity_flag=False, roll_bs_overall_flag=False,
                           pnl_sign_convention="RevNeg_ExpPos"):
    # ... (Function content is identical to the last correct version)
    processed_df = df.copy()
    if not all([account_id_col, mapping_col, amount_col, date_col]):
        st.error("Core column mappings are incomplete."); return None, None, None, None, None, None
    for col_name, user_col_name in zip(['Account ID', 'Mapping', 'Date'], [account_id_col, mapping_col, date_col]):
        if user_col_name not in processed_df.columns: st.error(f"Mapped '{col_name}' column ('{user_col_name}') not found."); return None, None, None, None, None, None
    if amount_col not in processed_df.columns: st.error(f"Mapped Amount column ('{amount_col}') not found."); return None, None, None, None, None, None
    try: processed_df[amount_col] = pd.to_numeric(processed_df[amount_col], errors='coerce').fillna(0)
    except Exception as e: st.error(f"Could not convert Amount column '{amount_col}' to numeric: {e}"); return None, None, None, None, None, None
    try:
        processed_df['_Standard_Date_'] = pd.to_datetime(processed_df[date_col], errors='coerce')
        if processed_df['_Standard_Date_'].isnull().any(): st.warning(f"Some dates in '{date_col}' could not be parsed."); processed_df.dropna(subset=['_Standard_Date_'], inplace=True)
        if processed_df.empty: st.error("No valid date entries after parsing."); return None, None, None, None, None, None
        processed_df['_YearMonth_'] = processed_df['_Standard_Date_'].dt.strftime('%Y-%m')
    except Exception as e: st.error(f"Could not process Date column '{date_col}': {e}"); return None, None, None, None, None, None
    processed_df['_Standard_Account_ID_'] = processed_df[account_id_col].astype(str).str.strip()
    processed_df['_Standard_Mapping_'] = processed_df[mapping_col].astype(str).str.strip()
    processed_df['_Assigned_Category_'] = processed_df['_Standard_Account_ID_'].apply(lambda x: assign_category_using_rules(x, use_custom_ranges_flag, parsed_gl_ranges, get_prefix_based_category))
    if use_custom_ranges_flag and any(v_list for v_list in parsed_gl_ranges.values() if v_list): st.caption("Using custom GL ranges where specified. Other accounts classified by prefix unless overridden by a custom rule for their prefix category.")
    else: st.caption("Using default prefix-based GL classification for all accounts.")
    pnl_categories_list = ["Revenue", "COGS", "Operating Expenses", "Other P&L Items"]; bs_categories_list = ["Assets", "Liabilities", "Equity"]
    processed_df['_Statement_Section_'] = processed_df['_Assigned_Category_'].apply(lambda cat: "P&L" if cat in pnl_categories_list else ("BS" if cat in bs_categories_list else "Uncategorized"))
    pl_data = processed_df[processed_df['_Statement_Section_'] == 'P&L']; bs_data_for_activity = processed_df[processed_df['_Statement_Section_'] == 'BS']
    uncategorized_data_rows = processed_df[processed_df['_Statement_Section_'] == 'Uncategorized']
    uncategorized_summary_df = pd.DataFrame()
    if not uncategorized_data_rows.empty:
        uncategorized_summary_df = uncategorized_data_rows.groupby(['_Standard_Account_ID_', '_Standard_Mapping_'])[amount_col].sum().reset_index()
        uncategorized_summary_df.rename(columns={amount_col: 'Total Amount', '_Standard_Account_ID_': 'Account ID', '_Standard_Mapping_':'Account Name'}, inplace=True)
    pl_display_df = pd.DataFrame(); net_income_series_for_recalc = pd.Series(dtype=float)
    gp_calc_id = "GP_CALC"; gp_calc_name = "Gross Profit"; ni_calc_id = "NET_INCOME_CALC"; ni_calc_name = "Net Income"
    pl_index_names = ['Account ID', 'Account Name'] 
    if not pl_data.empty:
        pl_grouped = pl_data.groupby(['_Standard_Account_ID_', '_Standard_Mapping_', '_Assigned_Category_', '_YearMonth_'])[amount_col].sum().reset_index()
        pl_itemized_df_with_cat = pl_grouped.pivot_table(index=['_Standard_Account_ID_', '_Standard_Mapping_', '_Assigned_Category_'], columns='_YearMonth_', values=amount_col, fill_value=0)
        if not pl_itemized_df_with_cat.empty:
            pl_itemized_df_with_cat = pl_itemized_df_with_cat.sort_index(level=0); pl_itemized_df_with_cat = pl_itemized_df_with_cat.sort_index(axis=1)
            pl_itemized_df = pl_itemized_df_with_cat.droplevel('_Assigned_Category_'); pl_itemized_df.index.names = pl_index_names
            revenue_accounts_df = pl_itemized_df_with_cat[pl_itemized_df_with_cat.index.get_level_values('_Assigned_Category_') == "Revenue"].droplevel('_Assigned_Category_')
            cogs_accounts_df = pl_itemized_df_with_cat[pl_itemized_df_with_cat.index.get_level_values('_Assigned_Category_') == "COGS"].droplevel('_Assigned_Category_')
            sum_revenue_series = revenue_accounts_df.sum(axis=0); sum_cogs_series = cogs_accounts_df.sum(axis=0)
            gross_profit_series = sum_revenue_series + sum_cogs_series
            gp_row_df = pd.DataFrame([gross_profit_series], index=pd.MultiIndex.from_tuples([(gp_calc_id, gp_calc_name)], names=pl_index_names))
            pl_construction_list = []
            if not revenue_accounts_df.empty: pl_construction_list.append(revenue_accounts_df)
            if not cogs_accounts_df.empty: pl_construction_list.append(cogs_accounts_df)
            pl_construction_list.append(gp_row_df)
            opex_accounts_df = pl_itemized_df_with_cat[pl_itemized_df_with_cat.index.get_level_values('_Assigned_Category_') == "Operating Expenses"].droplevel('_Assigned_Category_')
            if not opex_accounts_df.empty: pl_construction_list.append(opex_accounts_df)
            other_pl_items_df = pl_itemized_df_with_cat[pl_itemized_df_with_cat.index.get_level_values('_Assigned_Category_') == "Other P&L Items"].droplevel('_Assigned_Category_')
            if not other_pl_items_df.empty: pl_construction_list.append(other_pl_items_df)
            if pl_construction_list: pl_display_df = pd.concat(pl_construction_list)
            else: pl_display_df = pl_itemized_df.copy() if not pl_itemized_df.empty else pd.DataFrame(index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=pl_index_names))
            net_income_series_for_recalc = pl_itemized_df.sum(axis=0)
            ni_row_df = pd.DataFrame([net_income_series_for_recalc.copy()], index=pd.MultiIndex.from_tuples([(ni_calc_id, ni_calc_name)], names=pl_index_names))
            if not pl_display_df.empty: pl_display_df = pd.concat([pl_display_df, ni_row_df])
            elif not ni_row_df.empty: pl_display_df = ni_row_df 
            if not pl_display_df.empty: pl_display_df = pl_display_df.sort_index(axis=1)
    else: st.info("No data classified as P&L by your assignments.")
    bs_index_names = ['Account ID', 'Account Name']
    bs_activity_df = pd.DataFrame(index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=bs_index_names))
    calculated_re_acc_id = "RECALC"; calculated_re_acc_name = "Calculated Retained Earnings"
    assets_calc_id = "ASSETS_TOTAL"; assets_calc_name = "Total Assets"; liab_calc_id = "LIAB_TOTAL"; liab_calc_name = "Total Liabilities"; equity_calc_id = "EQUITY_TOTAL"; equity_calc_name = "Total Equity"
    if not bs_data_for_activity.empty:
        bs_grouped_activity = bs_data_for_activity.groupby(['_Standard_Account_ID_', '_Standard_Mapping_', '_YearMonth_'])[amount_col].sum().reset_index()
        bs_activity_df = bs_grouped_activity.pivot_table(index=['_Standard_Account_ID_', '_Standard_Mapping_'], columns='_YearMonth_', values=amount_col, fill_value=0)
        if not bs_activity_df.empty: bs_activity_df = bs_activity_df.sort_index(level=0); bs_activity_df = bs_activity_df.sort_index(axis=1) ; bs_activity_df.index.names = bs_index_names
    current_bs_items_df = bs_activity_df.copy()
    if roll_bs_overall_flag and not current_bs_items_df.empty: 
        st.caption("Rolling forward Balance Sheet account balances (Yearly YTD)...")
        bs_balances_df = pd.DataFrame(index=current_bs_items_df.index, columns=current_bs_items_df.columns, dtype=float)
        for account_idx_tuple in current_bs_items_df.index: 
            if account_idx_tuple[0] == calculated_re_acc_id: bs_balances_df.loc[account_idx_tuple, :] = current_bs_items_df.loc[account_idx_tuple, :]; continue
            current_year_cumulative_balance = 0.0; last_year_processed = None
            for period_col in current_bs_items_df.columns:
                year_of_period = period_col.split('-')[0]
                if last_year_processed is None or year_of_period != last_year_processed: current_year_cumulative_balance = 0.0; last_year_processed = year_of_period
                activity_for_period = current_bs_items_df.loc[account_idx_tuple, period_col] 
                current_year_cumulative_balance += activity_for_period
                bs_balances_df.loc[account_idx_tuple, period_col] = current_year_cumulative_balance
        current_bs_items_df = bs_balances_df 
    if not current_bs_items_df.empty and isinstance(current_bs_items_df.index, pd.MultiIndex) and 'Account ID' in current_bs_items_df.index.names:
        asset_items_df = current_bs_items_df[current_bs_items_df.index.get_level_values('Account ID').map(lambda x: assign_category_using_rules(x, use_custom_ranges_flag, parsed_gl_ranges, get_prefix_based_category) == "Assets")]
        liability_items_df = current_bs_items_df[current_bs_items_df.index.get_level_values('Account ID').map(lambda x: assign_category_using_rules(x, use_custom_ranges_flag, parsed_gl_ranges, get_prefix_based_category) == "Liabilities")]
        equity_items_base_df = current_bs_items_df[current_bs_items_df.index.get_level_values('Account ID').map(lambda x: assign_category_using_rules(x, use_custom_ranges_flag, parsed_gl_ranges, get_prefix_based_category) == "Equity") & (current_bs_items_df.index.get_level_values('Account ID') != calculated_re_acc_id)]
    else: 
        cols_for_empty = current_bs_items_df.columns if not current_bs_items_df.empty else (pl_display_df.columns if not pl_display_df.empty else None)
        asset_items_df = pd.DataFrame(columns=cols_for_empty, index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=bs_index_names)); liability_items_df = pd.DataFrame(columns=cols_for_empty, index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=bs_index_names)); equity_items_base_df = pd.DataFrame(columns=cols_for_empty, index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=bs_index_names))
    sum_assets_series_signed = asset_items_df.sum(axis=0); sum_liabilities_series_signed = liability_items_df.sum(axis=0); sum_equity_base_series_signed = equity_items_base_df.sum(axis=0)
    recalc_equity_row_df = pd.DataFrame(columns=current_bs_items_df.columns if not current_bs_items_df.empty else (pl_display_df.columns if not pl_display_df.empty else None))
    if roll_equity_flag and not net_income_series_for_recalc.empty:
        target_cols_for_re = current_bs_items_df.columns if not current_bs_items_df.empty else (pl_display_df.columns if not pl_display_df.empty else None)
        if target_cols_for_re is None or target_cols_for_re.empty: st.warning("Cannot roll RE as no periods defined.")
        else:
            st.caption("Rolling Net Income into Equity...")
            net_income_for_re = net_income_series_for_recalc.reindex(target_cols_for_re, fill_value=0.0)
            cumulative_re_values = {}; current_year_cumulative_ni = 0.0; last_year_processed = None
            for period_col in target_cols_for_re:
                year_of_period = period_col.split('-')[0]
                if last_year_processed is None or year_of_period != last_year_processed: current_year_cumulative_ni = 0.0; last_year_processed = year_of_period
                current_year_cumulative_ni += net_income_for_re.get(period_col, 0.0)
                if pnl_sign_convention == "RevPos_ExpNeg": cumulative_re_values[period_col] = current_year_cumulative_ni * -1
                else: cumulative_re_values[period_col] = current_year_cumulative_ni
            re_row_data = pd.Series(cumulative_re_values, name=(calculated_re_acc_id, calculated_re_acc_name))
            recalc_equity_row_df = re_row_data.to_frame().T; recalc_equity_row_df.index.names = ['Account ID', 'Account Name']
    bs_display_list = []
    if not asset_items_df.empty: bs_display_list.append(asset_items_df)
    bs_display_list.append(pd.DataFrame([sum_assets_series_signed], index=pd.MultiIndex.from_tuples([(assets_calc_id, assets_calc_name)], names=['Account ID', 'Account Name'])))
    if not liability_items_df.empty: bs_display_list.append(liability_items_df)
    bs_display_list.append(pd.DataFrame([sum_liabilities_series_signed], index=pd.MultiIndex.from_tuples([(liab_calc_id, liab_calc_name)], names=['Account ID', 'Account Name'])))
    if not equity_items_base_df.empty: bs_display_list.append(equity_items_base_df)
    if not recalc_equity_row_df.empty: bs_display_list.append(recalc_equity_row_df)
    empty_series_template_index = current_bs_items_df.columns if not current_bs_items_df.empty else (pl_display_df.columns if not pl_display_df.empty else None)
    sum_recalc_equity_series = recalc_equity_row_df.sum(axis=0) if not recalc_equity_row_df.empty else pd.Series(0, index=empty_series_template_index if empty_series_template_index is not None else None)
    _sum_equity_base_series_signed = sum_equity_base_series_signed if not sum_equity_base_series_signed.empty else pd.Series(0, index=empty_series_template_index if empty_series_template_index is not None else None)
    if not sum_recalc_equity_series.empty and not _sum_equity_base_series_signed.empty and not sum_recalc_equity_series.index.equals(_sum_equity_base_series_signed.index):
        common_index = sum_recalc_equity_series.index.union(_sum_equity_base_series_signed.index); sum_recalc_equity_series = sum_recalc_equity_series.reindex(common_index, fill_value=0); _sum_equity_base_series_signed = _sum_equity_base_series_signed.reindex(common_index, fill_value=0)
    elif sum_recalc_equity_series.empty and not _sum_equity_base_series_signed.empty: sum_recalc_equity_series = pd.Series(0, index=_sum_equity_base_series_signed.index)
    elif _sum_equity_base_series_signed.empty and not sum_recalc_equity_series.empty: _sum_equity_base_series_signed = pd.Series(0, index=sum_recalc_equity_series.index)
    current_total_equity_series_signed = _sum_equity_base_series_signed + sum_recalc_equity_series
    bs_display_list.append(pd.DataFrame([current_total_equity_series_signed], index=pd.MultiIndex.from_tuples([(equity_calc_id, equity_calc_name)], names=['Account ID', 'Account Name'])))
    bs_final_display_df = pd.concat(bs_display_list) if bs_display_list else pd.DataFrame(index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=bs_index_names))
    if not bs_final_display_df.empty: bs_final_display_df = bs_final_display_df.sort_index(axis=1)
    bs_audit_table_for_display = pd.DataFrame(); balance_check_series_for_logic = pd.Series(dtype=float)
    if not bs_final_display_df.empty :
        actual_assets_total = sum_assets_series_signed; actual_liabilities_total = sum_liabilities_series_signed
        actual_equity_total = current_total_equity_series_signed
        balance_check_series_for_logic = actual_assets_total + actual_liabilities_total + actual_equity_total
        bs_audit_table_for_display = pd.DataFrame({ "Total Assets": actual_assets_total, "Total Liabilities": actual_liabilities_total, "Total Equity": actual_equity_total, "Audit Check (A = L + E)": balance_check_series_for_logic }).T
    else: st.info("Balance Sheet is empty, cannot perform audit check.")
    return pl_display_df, bs_final_display_df, net_income_series_for_recalc, bs_audit_table_for_display, balance_check_series_for_logic, uncategorized_summary_df

def get_config_to_export():
    config_data = []
    config_data.append({"SettingKey": "app_config_version", "SettingValue": "1.0"})
    config_data.append({"SettingKey": "selected_je_sheets", "SettingValue": ",".join(st.session_state.get('selected_je_sheets', []))})
    config_data.append({"SettingKey": "debit_column", "SettingValue": st.session_state.get('debit_column_selection', "") or ""})
    config_data.append({"SettingKey": "credit_column", "SettingValue": st.session_state.get('credit_column_selection', "") or ""})
    core_maps = st.session_state.get('column_mappings', {})
    config_data.append({"SettingKey": "mapping_account_id", "SettingValue": core_maps.get("account_id", "") or ""})
    config_data.append({"SettingKey": "mapping_account_name", "SettingValue": core_maps.get("mapping", "") or ""})
    config_data.append({"SettingKey": "mapping_amount", "SettingValue": core_maps.get("amount", "") or ""})
    config_data.append({"SettingKey": "mapping_date", "SettingValue": core_maps.get("date", "") or ""})
    config_data.append({"SettingKey": "pnl_sign_convention", "SettingValue": st.session_state.get('pnl_sign_convention_selection', "RevNeg_ExpPos")})
    config_data.append({"SettingKey": "roll_bs_balances_flag", "SettingValue": str(st.session_state.get('roll_bs_selection', False))})
    config_data.append({"SettingKey": "roll_equity_flag", "SettingValue": str(st.session_state.get('roll_equity_selection', False))})
    config_data.append({"SettingKey": "use_custom_ranges_flag", "SettingValue": str(st.session_state.get('use_custom_ranges', False))})
    custom_ranges = st.session_state.get('custom_gl_range_inputs', {})
    for category in RANGE_INPUT_CATEGORIES:
        config_data.append({"SettingKey": f"custom_range_{category.replace(' ', '_').replace('/', '_')}", "SettingValue": custom_ranges.get(category, "")})
    return pd.DataFrame(config_data)

def trigger_custom_range_processing():
    parsed_ranges = {}
    all_ranges_valid_overall = True
    range_inputs_to_process = st.session_state.get('custom_gl_range_inputs', {cat: "" for cat in RANGE_INPUT_CATEGORIES})
    for category, range_str in range_inputs_to_process.items():
        if range_str and isinstance(range_str, str) and range_str.strip():
            parsed_current_cat = parse_gl_ranges(range_str)
            if parsed_current_cat is None: 
                all_ranges_valid_overall = False
                parsed_ranges[category] = st.session_state.parsed_gl_ranges.get(category, []) 
            else:
                parsed_ranges[category] = parsed_current_cat
        else:
            parsed_ranges[category] = [] 
    if all_ranges_valid_overall:
        st.session_state.parsed_gl_ranges = parsed_ranges
        st.session_state.use_custom_ranges = any(bool(v_list) for v_list in parsed_ranges.values() if v_list)
        return True 
    else:
        st.session_state.parsed_gl_ranges = parsed_ranges 
        st.session_state.use_custom_ranges = any(bool(v_list) for v_list in parsed_ranges.values() if v_list) 
        return False

def load_config_data_from_excel(excel_file_buffer, sheet_name_const):
    try:
        excel_file_buffer.seek(0)
        df_config = pd.read_excel(excel_file_buffer, sheet_name=sheet_name_const)
        settings_dict = pd.Series(df_config['SettingValue'].astype(str).replace('nan', '').values, index=df_config['SettingKey']).to_dict()
        return settings_dict
    except Exception as e: st.warning(f"Could not read or parse configuration sheet '{sheet_name_const}': {e}"); return None

def apply_config_to_session_state(config_settings):
    if not config_settings: return
    def get_config_val(key, default=""): val = str(config_settings.get(key, default)).strip(); return "" if val.lower() == 'nan' else val
    raw_sheets = get_config_val('selected_je_sheets')
    st.session_state.selected_je_sheets = [s.strip() for s in raw_sheets.split(',') if s.strip()] if raw_sheets else []
    st.session_state.debit_column_selection = get_config_val('debit_column') or None
    st.session_state.credit_column_selection = get_config_val('credit_column') or None
    st.session_state.column_mappings["account_id"] = get_config_val('mapping_account_id') or None
    st.session_state.column_mappings["mapping"] = get_config_val('mapping_account_name') or None
    st.session_state.column_mappings["amount"] = get_config_val('mapping_amount') or None
    st.session_state.column_mappings["date"] = get_config_val('mapping_date') or None
    st.session_state.pnl_sign_convention_selection = get_config_val('pnl_sign_convention', "RevNeg_ExpPos")
    st.session_state.roll_bs_selection = get_config_val('roll_bs_balances_flag', "False").lower() == 'true'
    st.session_state.roll_equity_selection = get_config_val('roll_equity_flag', "False").lower() == 'true'
    temp_custom_ranges = {}
    for category in RANGE_INPUT_CATEGORIES:
        key = f"custom_range_{category.replace(' ', '_').replace('/', '_')}"
        temp_custom_ranges[category] = get_config_val(key, "")
    st.session_state.custom_gl_range_inputs = temp_custom_ranges
    use_custom_ranges_str = get_config_val('use_custom_ranges_flag', "False")
    st.session_state.use_custom_ranges = use_custom_ranges_str.lower() == 'true'

# --- Session State Initialization ---
if 'uploaded_excel_file_info' not in st.session_state: st.session_state.uploaded_excel_file_info = None
if 'sheet_names' not in st.session_state: st.session_state.sheet_names = []
if 'selected_je_sheets' not in st.session_state: st.session_state.selected_je_sheets = []
if 'compiled_je_df' not in st.session_state: st.session_state.compiled_je_df = None
if 'column_mappings' not in st.session_state: 
    st.session_state.column_mappings = {"account_id": None, "mapping": None, "amount": None, "date": None }
if 'custom_gl_range_inputs' not in st.session_state: st.session_state.custom_gl_range_inputs = {cat: "" for cat in RANGE_INPUT_CATEGORIES}
if 'parsed_gl_ranges' not in st.session_state: st.session_state.parsed_gl_ranges = {}
if 'use_custom_ranges' not in st.session_state: st.session_state.use_custom_ranges = False
if 'show_statements' not in st.session_state: st.session_state.show_statements = False
if 'debit_column_selection' not in st.session_state: st.session_state.debit_column_selection = None
if 'credit_column_selection' not in st.session_state: st.session_state.credit_column_selection = None
if 'roll_equity_selection' not in st.session_state: st.session_state.roll_equity_selection = False
if 'roll_bs_selection' not in st.session_state: st.session_state.roll_bs_selection = False
if 'pnl_sign_convention_selection' not in st.session_state: st.session_state.pnl_sign_convention_selection = "RevNeg_ExpPos"

# --- UI Sections ---
st.subheader("1. Upload Excel File")
uploaded_file_widget = st.file_uploader("Choose an Excel file (.xlsx or .xls)", type=["xlsx", "xls"], key="file_uploader_data_upload")
config_loaded_this_run = False 
if uploaded_file_widget:
    new_file_uploaded = False
    if st.session_state.uploaded_excel_file_info is None or \
       st.session_state.uploaded_excel_file_info['name'] != uploaded_file_widget.name or \
       st.session_state.uploaded_excel_file_info['size'] != uploaded_file_widget.size:
        new_file_uploaded = True
        st.session_state.uploaded_excel_file_info = {'name': uploaded_file_widget.name,'size': uploaded_file_widget.size,'data': BytesIO(uploaded_file_widget.getvalue())}
        st.session_state.sheet_names = [] ; st.session_state.selected_je_sheets = [] ; st.session_state.compiled_je_df = None
        st.session_state.column_mappings = {key: None for key in st.session_state.column_mappings}
        st.session_state.custom_gl_range_inputs = {cat: "" for cat in RANGE_INPUT_CATEGORIES}; st.session_state.parsed_gl_ranges = {}; st.session_state.use_custom_ranges = False
        st.session_state.show_statements = False; st.session_state.debit_column_selection = None; st.session_state.credit_column_selection = None
        st.session_state.roll_equity_selection = False; st.session_state.roll_bs_selection = False; st.session_state.pnl_sign_convention_selection = "RevNeg_ExpPos" 
        st.success(f"Uploaded '{uploaded_file_widget.name}' successfully.")
    excel_file_buffer = st.session_state.uploaded_excel_file_info['data']; excel_file_buffer.seek(0) 
    try:
        if not st.session_state.sheet_names: st.session_state.sheet_names = pd.ExcelFile(excel_file_buffer).sheet_names
        if new_file_uploaded and CONFIG_SHEET_NAME in st.session_state.sheet_names:
            st.info(f"Configuration sheet '{CONFIG_SHEET_NAME}' detected. Attempting to load parameters...")
            excel_file_buffer.seek(0); loaded_settings = load_config_data_from_excel(excel_file_buffer, CONFIG_SHEET_NAME)
            if loaded_settings:
                apply_config_to_session_state(loaded_settings)
                if trigger_custom_range_processing(): 
                     st.success("Parameters successfully loaded and custom ranges processed from configuration sheet!")
                else:
                     st.warning("Parameters loaded, but some custom ranges had errors. Please review them in Section 4.")
                config_loaded_this_run = True 
            else: st.warning(f"Could not load parameters from '{CONFIG_SHEET_NAME}'. Please configure manually.")
        elif new_file_uploaded: st.info("No configuration sheet found. Please configure parameters manually.")
    except Exception as e: st.error(f"Could not process the uploaded Excel file: {e}"); st.session_state.uploaded_excel_file_info = None ; st.stop()
    if config_loaded_this_run: st.rerun()

if st.session_state.uploaded_excel_file_info and st.session_state.sheet_names:
    st.subheader("2. Select Sheets for Journal Entries (JEs)")
    current_selection = st.session_state.get('selected_je_sheets', [])
    if not isinstance(current_selection, list): current_selection = []
    st.session_state.selected_je_sheets = st.multiselect( "Which sheet(s) contain your JE data?", options=st.session_state.sheet_names, default=current_selection)
    if st.session_state.selected_je_sheets:
        if st.button("Compile Selected JE Sheets", key="compile_je_btn"):
            st.session_state.compiled_je_df = None; st.session_state.show_statements = False
            # When compiling, D/C selections and amount mapping persist if from config
            # Custom range flags are reset as they depend on compile action
            if not config_loaded_this_run:
                st.session_state.use_custom_ranges = False 
                st.session_state.parsed_gl_ranges = {}
                # custom_gl_range_inputs are preserved if from config, else they are ""

            all_dfs = []
            if st.session_state.uploaded_excel_file_info and 'data' in st.session_state.uploaded_excel_file_info:
                current_excel_buffer_compile = st.session_state.uploaded_excel_file_info['data']
                for sheet in st.session_state.selected_je_sheets:
                    current_excel_buffer_compile.seek(0); sheet_df = safe_read_excel_sheet(current_excel_buffer_compile, sheet)
                    if sheet_df is not None: all_dfs.append(sheet_df)
            else: st.error("Uploaded file data not found."); st.stop()
            
            if all_dfs:
                try:
                    st.session_state.compiled_je_df = pd.concat(all_dfs, ignore_index=True)
                    st.success(f"Successfully compiled {len(all_dfs)} JE sheet(s) into {len(st.session_state.compiled_je_df)} rows.")
                    
                    # Auto-create "Total_Amount_(calc)" if config implies it
                    if st.session_state.column_mappings.get("amount") == CALCULATED_AMOUNT_COL_NAME and \
                       st.session_state.debit_column_selection and st.session_state.credit_column_selection:
                        
                        if try_create_total_amount_column(st.session_state.debit_column_selection, st.session_state.credit_column_selection):
                            st.success(f"'{CALCULATED_AMOUNT_COL_NAME}' column automatically created/verified based on configuration.")
                        else:
                            st.warning(f"Config specified '{CALCULATED_AMOUNT_COL_NAME}', but it could not be auto-created. Check D/C cols or create manually.")
                    # Auto-detect for manual button if not set by config
                    elif st.session_state.debit_column_selection is None and st.session_state.credit_column_selection is None:
                        df_cols_lower = [str(col).lower() for col in st.session_state.compiled_je_df.columns]
                        original_cols = list(st.session_state.compiled_je_df.columns)
                        if "debit" in df_cols_lower: st.session_state.debit_column_selection = original_cols[df_cols_lower.index("debit")]
                        if "credit" in df_cols_lower: st.session_state.credit_column_selection = original_cols[df_cols_lower.index("credit")]
                except Exception as e: st.error(f"Error during JE sheet compilation: {e}")
            else: st.warning("No dataframes were successfully read from the selected sheets.")
            st.rerun() 
            
if st.session_state.compiled_je_df is not None:
    st.subheader("Preview of Compiled JE Data (First 5 Rows)")
    st.dataframe(st.session_state.compiled_je_df.head(), height=200, use_container_width=True)
    
    with st.expander("Optional: Handle Separate Debit/Credit Columns", expanded=False):
        st.markdown("###### Use this if your data has distinct Debit and Credit columns.</h6>", unsafe_allow_html=True)
        df_cols_for_dc = [""] + list(st.session_state.compiled_je_df.columns)
        debit_idx = df_cols_for_dc.index(st.session_state.debit_column_selection) if st.session_state.debit_column_selection and st.session_state.debit_column_selection in df_cols_for_dc else 0
        credit_idx = df_cols_for_dc.index(st.session_state.credit_column_selection) if st.session_state.credit_column_selection and st.session_state.credit_column_selection in df_cols_for_dc else 0
        
        selected_debit_col_widget = st.selectbox( "Select Debit Column:", options=df_cols_for_dc, index=debit_idx, key="debit_col_sel_exp")
        selected_credit_col_widget = st.selectbox( "Select Credit Column:", options=df_cols_for_dc, index=credit_idx, key="credit_col_sel_exp")
        
        if selected_debit_col_widget != st.session_state.debit_column_selection: st.session_state.debit_column_selection = selected_debit_col_widget
        if selected_credit_col_widget != st.session_state.credit_column_selection: st.session_state.credit_column_selection = selected_credit_col_widget

        if st.button(f"Create '{CALCULATED_AMOUNT_COL_NAME}' Column", key="create_total_amount_btn_exp"):
            if try_create_total_amount_column(st.session_state.debit_column_selection, st.session_state.credit_column_selection):
                st.session_state.column_mappings["amount"] = CALCULATED_AMOUNT_COL_NAME 
                st.success(f"Successfully created/verified '{CALCULATED_AMOUNT_COL_NAME}' column.")
                st.rerun()
            else: st.error(f"Failed to create '{CALCULATED_AMOUNT_COL_NAME}' column. Check D/C cols exist/are numeric.")
    st.markdown("---")

    st.subheader("3. Map Core Data Columns & P&L Convention")
    pnl_convention_options_map = { "RevNeg_ExpPos": "Revenue Negative (Credits) / COGS & Expenses Positive (Debits)", "RevPos_ExpNeg": "Revenue Positive / COGS & Expenses Negative"}
    options_keys = list(pnl_convention_options_map.keys())
    current_pnl_convention_idx = options_keys.index(st.session_state.pnl_sign_convention_selection) if st.session_state.pnl_sign_convention_selection in options_keys else 0
    st.session_state.pnl_sign_convention_selection = st.radio("P&L Value Representation:", options=options_keys, index=current_pnl_convention_idx, format_func=lambda x: pnl_convention_options_map[x], key="pnl_sign_radio", horizontal=True)
    df_columns_for_mapping = [""] + list(st.session_state.compiled_je_df.columns) 
    mappings = st.session_state.column_mappings; col_map_1, col_map_2 = st.columns(2)
    with col_map_1:
        def get_idx(key_name): val = mappings.get(key_name); return df_columns_for_mapping.index(val) if val and val in df_columns_for_mapping else 0
        mappings["account_id"] = st.selectbox("Account ID Column:", options=df_columns_for_mapping, index=get_idx("account_id"), help="Column containing GL account numbers.")
        
        if CALCULATED_AMOUNT_COL_NAME in df_columns_for_mapping and mappings.get("amount") == CALCULATED_AMOUNT_COL_NAME: 
            st.write(f"**Amount Column:** `{CALCULATED_AMOUNT_COL_NAME}` (Generated)")
        else: 
            mappings["amount"] = st.selectbox("Amount Column:", options=df_columns_for_mapping, index=get_idx("amount"), help="Numerical column for transaction amounts.")
    with col_map_2:
        mappings["mapping"] = st.selectbox("Primary Account Mapping Column (Name/Description):", options=df_columns_for_mapping, index=get_idx("mapping"), help="Column for P&L/BS line item descriptions.")
        mappings["date"] = st.selectbox("Date Column:", options=df_columns_for_mapping, index=get_idx("date"), help="Column containing transaction dates.")
    st.session_state.column_mappings = mappings
    
    st.markdown("---")
    st.subheader("4. Define Financial Statement Structure (GL Groupings)")
    st.info("Default: Assets (1xxxx), Liabilities (2), Equity (3), Revenue (4), COGS (5), OpEx (6-7), Other P&L (8-9). Refine below.")
    with st.expander("Refine GL Account Grouping Ranges (Optional)", expanded=False):
        st.markdown("Enter comma-separated GL account ranges (e.g., `1000-1999, 2500, 2600-2650`). Invalid entries will be ignored.")
        temp_range_inputs = {} 
        for category in RANGE_INPUT_CATEGORIES:
            input_label = f"Ranges for {category}:"; default_placeholder = f"Default: {DEFAULT_PREFIX_MAP.get(category, 'Not specified')}"
            if category == "COGS": input_label = "Ranges for COGS/COS:"
            temp_range_inputs[category] = st.text_input(input_label, value=st.session_state.custom_gl_range_inputs.get(category, ""), placeholder=default_placeholder, key=f"range_input_{category}")
        
        if st.button("Apply Custom Ranges", key="apply_custom_ranges_btn"):
            st.session_state.custom_gl_range_inputs = temp_range_inputs 
            if trigger_custom_range_processing():
                 if not st.session_state.use_custom_ranges: st.info("No valid custom ranges defined or all inputs cleared. Default prefix logic will be used.")
            else: st.error("Some custom ranges had input errors and were not fully applied. Review inputs. Default prefix logic may apply more broadly where errors occurred.")
            st.rerun()

    if st.session_state.use_custom_ranges and any(bool(v_list) for v_list in st.session_state.get('parsed_gl_ranges', {}).values() if v_list): 
        st.success("Custom GL grouping ranges are active where specified.")
    
    st.markdown("---")
    st.subheader("5. Final Options & Generation")
    st.session_state.roll_bs_selection = st.checkbox("Roll forward all Balance Sheet account balances (Yearly YTD from activity)", value=st.session_state.get('roll_bs_selection', False), key="roll_bs_checkbox")
    st.session_state.roll_equity_selection = st.checkbox("Roll Net Income into Equity (Adds 'Calculated Retained Earnings' to BS)", value=st.session_state.get('roll_equity_selection', False), key="roll_ni_checkbox")
    
    if st.button("Generate Financial Statements", key="generate_statements_btn"):
        chosen_amount_col = st.session_state.column_mappings.get("amount")
        if not (st.session_state.column_mappings["account_id"] and st.session_state.column_mappings["mapping"] and \
                chosen_amount_col and st.session_state.column_mappings["date"]):
            st.error("Please ensure all Core Data Columns are mapped first.")
            st.session_state.show_statements = False
        elif chosen_amount_col not in st.session_state.compiled_je_df.columns:
             st.error(f"Selected Amount column ('{chosen_amount_col}') is not valid or not found."); st.session_state.show_statements = False
        else:
            with st.spinner("Processing data and generating statements..."):
                pl_df_final, bs_df_final, _, \
                bs_audit_table_for_display, internal_bs_check_series, \
                uncategorized_summary = process_financial_data(
                    st.session_state.compiled_je_df, st.session_state.column_mappings["account_id"],
                    st.session_state.column_mappings["mapping"], chosen_amount_col,
                    st.session_state.column_mappings["date"],
                    st.session_state.use_custom_ranges, 
                    st.session_state.parsed_gl_ranges,   
                    st.session_state.roll_equity_selection, st.session_state.roll_bs_selection,
                    st.session_state.pnl_sign_convention_selection)
            st.session_state.pl_wide_df_processed = pl_df_final
            st.session_state.bs_wide_df_processed = bs_df_final
            st.session_state.bs_audit_df_processed_display = bs_audit_table_for_display
            st.session_state.internal_bs_check_series_logic = internal_bs_check_series
            st.session_state.uncategorized_accounts_summary = uncategorized_summary
            st.session_state.show_statements = True; st.success("Statements generated!"); st.rerun()

# --- Statement Display and Export Section ---
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
        pl_calculated_row_ids = [("GP_CALC", "Gross Profit"), ("NET_INCOME_CALC", "Net Income")]
        styled_pl_df = style_financial_table(pl_df_to_display, pl_calculated_row_ids)
        st.dataframe(styled_pl_df, use_container_width=True)
    elif pl_df_to_display is not None and pl_df_to_display.empty: st.info("P&L Statement is empty.")
    else: st.warning("P&L Statement could not be generated.")
    st.markdown("---"); st.markdown("#### Balance Sheet")
    bs_df_to_display = st.session_state.get('bs_wide_df_processed')
    bs_audit_df_for_user_display = st.session_state.get('bs_audit_df_processed_display') 
    internal_check_series_for_logic = st.session_state.get('internal_bs_check_series_logic')
    if bs_df_to_display is not None and not bs_df_to_display.empty:
        bs_calculated_row_ids = [ ("ASSETS_TOTAL", "Total Assets"), ("LIAB_TOTAL", "Total Liabilities"), ("EQUITY_TOTAL", "Total Equity") ]
        styled_bs_df = style_financial_table(bs_df_to_display, bs_calculated_row_ids)
        st.dataframe(styled_bs_df, use_container_width=True)
        st.markdown("##### Balance Sheet Audit Check") 
        if bs_audit_df_for_user_display is not None and not bs_audit_df_for_user_display.empty:
            st.dataframe(bs_audit_df_for_user_display.fillna(0).style.format("{:,.0f}", na_rep="0"), use_container_width=True)
            if internal_check_series_for_logic is not None and not internal_check_series_for_logic.empty:
                is_balanced = np.allclose(internal_check_series_for_logic.fillna(0), 0, atol=0.51) 
                if is_balanced: st.success("Balance Sheet is balanced!")
                else: st.warning(f"Balance Sheet does not balance. Max difference: {internal_check_series_for_logic.abs().max():,.0f}. Consider reviewing Step 5 options (rolling BS activity and/or Net Income).")
            else: st.warning("Internal balance check data not available.")
        else: st.info("BS Audit Check data is not available for display.")
    elif bs_df_to_display is not None and bs_df_to_display.empty: st.info("Balance Sheet Statement is empty.")
    else: st.warning("Balance Sheet Statement could not be generated.")

    st.markdown("---")
    st.subheader("Export Page Parameters to Excel") 
    st.caption("Exports current selections to an Excel sheet. Add this sheet named '_FINANCIAL_APP_CONFIG_' to future data files for auto-setup.") 
    output_buffer_export = BytesIO() 
    config_df_to_export = get_config_to_export()
    can_export = not config_df_to_export.empty 
    if can_export: 
        instructions_text = """
Instructions for Using the Configuration Sheet (_FINANCIAL_APP_CONFIG_):
1. Purpose: This sheet contains parameters from the 'Data Upload & Validation' page.
2. How to Use: Copy this ENTIRE sheet into your main Excel data file. Name it exactly '_FINANCIAL_APP_CONFIG_'. The app will auto-load these settings on your next upload.
3. Parameters Saved: JE sheets, D/C columns, core mappings, P&L sign convention, custom GL ranges, roll flags.
Note: If data file structures change, re-export or update this config."""
        instruction_lines = [line.strip() for line in instructions_text.strip().split('\n') if line.strip()]
        instructions_df = pd.DataFrame(instruction_lines, columns=["Instructions"])
        with pd.ExcelWriter(output_buffer_export, engine='openpyxl') as writer:
            config_df_to_export.to_excel(writer, sheet_name=CONFIG_SHEET_NAME, index=False)
            instructions_df.to_excel(writer, sheet_name="Instructions", index=False)
        output_buffer_export.seek(0)
        st.download_button(label="Export Parameters to Excel", data=output_buffer_export, file_name="financial_app_config_profile.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="export_config_btn_main_final_pos")
    else: st.button("Export Parameters to Excel", disabled=True, help="Configure settings on the page first.")