import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# --- Page Config & Title ---
st.set_page_config(layout="wide", page_title="Data Upload & Validation")
st.title("📄 Data Upload & Initial Validation")
st.caption("Upload your financial data (Journal Entries) in an Excel file, map the necessary columns, and view a generated P&L and Balance Sheet for validation.")

# --- Helper Functions ---
def safe_read_excel_sheet(excel_file, sheet_name):
    try: return pd.read_excel(excel_file, sheet_name=sheet_name)
    except Exception as e: st.error(f"Error reading sheet '{sheet_name}': {e}"); return None

def style_financial_table(df_to_style, calculated_row_ids):
    def highlight_specific_rows(row):
        if row.name in calculated_row_ids:
            return ['font-weight: bold; background-color: #e6f3ff'] * len(row)
        return [''] * len(row)
    return df_to_style.fillna(0).style.apply(highlight_specific_rows, axis=1).format("{:,.0f}", na_rep="0")

def process_financial_data(df, account_id_col, mapping_col, amount_col, date_col,
                           account_type_col=None, roll_equity_flag=False, roll_bs_overall_flag=False,
                           pnl_sign_convention="RevNeg_ExpPos"):
    processed_df = df.copy()
    # --- 1. Basic Validations & Type Conversions ---
    if not all([account_id_col, mapping_col, amount_col, date_col]):
        st.error("Core column mappings are incomplete."); return None, None, None, None
    for col_name, user_col_name in zip(['Account ID', 'Mapping', 'Date'], [account_id_col, mapping_col, date_col]):
        if user_col_name not in processed_df.columns: st.error(f"Mapped '{col_name}' column ('{user_col_name}') not found."); return None, None, None, None
    if amount_col not in processed_df.columns: st.error(f"Mapped Amount column ('{amount_col}') not found."); return None, None, None, None
    # use_account_type_col_for_audit flag (not directly used for BS audit table structure now, but for classifying components)
    if account_type_col and account_type_col in processed_df.columns:
        processed_df['_Standard_Account_Type_'] = processed_df[account_type_col].astype(str).str.strip().str.lower()
    elif account_type_col: st.warning(f"Mapped Account Type column ('{account_type_col}') not found. BS Audit classification will use GL prefixes.")
    try: processed_df[amount_col] = pd.to_numeric(processed_df[amount_col], errors='coerce').fillna(0)
    except Exception as e: st.error(f"Could not convert Amount column '{amount_col}' to numeric: {e}"); return None, None, None, None
    try:
        processed_df['_Standard_Date_'] = pd.to_datetime(processed_df[date_col], errors='coerce')
        if processed_df['_Standard_Date_'].isnull().any(): st.warning(f"Some dates in '{date_col}' could not be parsed."); processed_df.dropna(subset=['_Standard_Date_'], inplace=True)
        if processed_df.empty: st.error("No valid date entries after parsing."); return None, None, None, None
        processed_df['_YearMonth_'] = processed_df['_Standard_Date_'].dt.strftime('%Y-%m')
    except Exception as e: st.error(f"Could not process Date column '{date_col}': {e}"); return None, None, None, None
    processed_df['_Standard_Account_ID_'] = processed_df[account_id_col].astype(str).str.strip()
    processed_df['_Standard_Mapping_'] = processed_df[mapping_col].astype(str).str.strip()
    def classify_statement_section(acc_id_str):
        if acc_id_str and acc_id_str[0] >= '4' and acc_id_str[0] <= '9': return 'P&L'
        return 'BS'
    processed_df['_Statement_Section_'] = processed_df['_Standard_Account_ID_'].apply(classify_statement_section)
    pl_data = processed_df[processed_df['_Statement_Section_'] == 'P&L']
    bs_data_for_activity = processed_df[processed_df['_Statement_Section_'] == 'BS']

    # --- 2. P&L Generation ---
    pl_display_df = pd.DataFrame()
    net_income_series_for_recalc = pd.Series(dtype=float)
    gp_calc_id = "GP_CALC"; gp_calc_name = "Gross Profit"
    ni_calc_id = "NET_INCOME_CALC"; ni_calc_name = "Net Income"
    # ... (P&L logic remains the same as the previously corrected version) ...
    if not pl_data.empty:
        pl_grouped = pl_data.groupby(['_Standard_Account_ID_', '_Standard_Mapping_', '_YearMonth_'])[amount_col].sum().reset_index()
        pl_itemized_df = pl_grouped.pivot_table(index=['_Standard_Account_ID_', '_Standard_Mapping_'], columns='_YearMonth_', values=amount_col, fill_value=0)
        if not pl_itemized_df.empty:
            pl_itemized_df = pl_itemized_df.sort_index(level=0); pl_itemized_df = pl_itemized_df.sort_index(axis=1)
            pl_itemized_df.index.names = ['Account ID', 'Account Name']
            revenue_accounts_df = pl_itemized_df[pl_itemized_df.index.get_level_values('Account ID').str.startswith('4')]
            cogs_accounts_df = pl_itemized_df[pl_itemized_df.index.get_level_values('Account ID').str.startswith('5')]
            sum_revenue_series = revenue_accounts_df.sum(axis=0)
            sum_cogs_series = cogs_accounts_df.sum(axis=0)
            gross_profit_series = sum_revenue_series + sum_cogs_series
            gp_row_df = pd.DataFrame([gross_profit_series], index=pd.MultiIndex.from_tuples([(gp_calc_id, gp_calc_name)], names=['Account ID', 'Account Name']))
            temp_pl_df_construction = pl_itemized_df.copy(); all_pl_account_ids = temp_pl_df_construction.index.get_level_values('Account ID').tolist()
            insertion_point = len(temp_pl_df_construction)
            if not cogs_accounts_df.empty:
                for i in range(len(all_pl_account_ids) - 1, -1, -1):
                    if all_pl_account_ids[i].startswith('5'): insertion_point = i + 1; break
            elif not revenue_accounts_df.empty:
                for i in range(len(all_pl_account_ids) - 1, -1, -1):
                    if all_pl_account_ids[i].startswith('4'): insertion_point = i + 1; break
            part1 = temp_pl_df_construction.iloc[:insertion_point]; part2 = temp_pl_df_construction.iloc[insertion_point:]
            pl_display_df = pd.concat([part1, gp_row_df, part2])
            net_income_series_for_recalc = pl_itemized_df.sum(axis=0)
            ni_row_df = pd.DataFrame([net_income_series_for_recalc.copy()], index=pd.MultiIndex.from_tuples([(ni_calc_id, ni_calc_name)], names=['Account ID', 'Account Name']))
            pl_display_df = pd.concat([pl_display_df, ni_row_df])
            if not pl_display_df.empty: pl_display_df = pl_display_df.sort_index(axis=1)
    else: st.info("No data classified as P&L.")


    # --- BS Generation ---
    bs_activity_df = pd.DataFrame() 
    calculated_re_acc_id = "RECALC"; calculated_re_acc_name = "Calculated Retained Earnings"
    assets_calc_id = "ASSETS_TOTAL"; assets_calc_name = "Total Assets"
    liab_calc_id = "LIAB_TOTAL"; liab_calc_name = "Total Liabilities"
    equity_calc_id = "EQUITY_TOTAL"; equity_calc_name = "Total Equity"
    # ... (BS activity, overall rolling, NI rolling logic as previously corrected) ...
    if not bs_data_for_activity.empty:
        bs_grouped_activity = bs_data_for_activity.groupby(['_Standard_Account_ID_', '_Standard_Mapping_', '_YearMonth_'])[amount_col].sum().reset_index()
        bs_activity_df = bs_grouped_activity.pivot_table(index=['_Standard_Account_ID_', '_Standard_Mapping_'], columns='_YearMonth_', values=amount_col, fill_value=0)
        if not bs_activity_df.empty:
            bs_activity_df = bs_activity_df.sort_index(level=0); bs_activity_df = bs_activity_df.sort_index(axis=1) 
            bs_activity_df.index.names = ['Account ID', 'Account Name']
    current_bs_items_df = bs_activity_df.copy()
    if roll_bs_overall_flag and not bs_activity_df.empty:
        st.caption("Rolling forward Balance Sheet account balances (Yearly YTD)...")
        bs_balances_df = pd.DataFrame(index=bs_activity_df.index, columns=bs_activity_df.columns, dtype=float)
        for account_idx_tuple in bs_activity_df.index:
            if account_idx_tuple[0] == calculated_re_acc_id:
                bs_balances_df.loc[account_idx_tuple, :] = bs_activity_df.loc[account_idx_tuple, :]
                continue
            current_year_cumulative_balance = 0.0; last_year_processed = None
            for period_col in bs_activity_df.columns:
                year_of_period = period_col.split('-')[0]
                if last_year_processed is None or year_of_period != last_year_processed:
                    current_year_cumulative_balance = 0.0; last_year_processed = year_of_period
                activity_for_period = bs_activity_df.loc[account_idx_tuple, period_col]
                current_year_cumulative_balance += activity_for_period
                bs_balances_df.loc[account_idx_tuple, period_col] = current_year_cumulative_balance
        current_bs_items_df = bs_balances_df 
    asset_items_df = current_bs_items_df[current_bs_items_df.index.get_level_values('Account ID').str.startswith('1')]
    liability_items_df = current_bs_items_df[current_bs_items_df.index.get_level_values('Account ID').str.startswith('2')]
    equity_items_base_df = current_bs_items_df[current_bs_items_df.index.get_level_values('Account ID').str.startswith('3') & (current_bs_items_df.index.get_level_values('Account ID') != calculated_re_acc_id)]
    sum_assets_series_signed = asset_items_df.sum(axis=0)
    sum_liabilities_series_signed = liability_items_df.sum(axis=0)
    sum_equity_base_series_signed = equity_items_base_df.sum(axis=0)
    recalc_equity_row_df = pd.DataFrame(columns=current_bs_items_df.columns if not current_bs_items_df.empty else (pl_display_df.columns if not pl_display_df.empty else None))
    if roll_equity_flag and not net_income_series_for_recalc.empty:
        target_cols_for_re = current_bs_items_df.columns if not current_bs_items_df.empty else (pl_display_df.columns if not pl_display_df.empty else None)
        if target_cols_for_re is None or target_cols_for_re.empty: st.warning("Cannot roll RE as no periods defined in BS or P&L.")
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
    sum_recalc_equity_series = recalc_equity_row_df.sum(axis=0) if not recalc_equity_row_df.empty else pd.Series(0, index=sum_equity_base_series_signed.index if not sum_equity_base_series_signed.empty else (sum_assets_series_signed.index if not sum_assets_series_signed.empty else None))
    current_total_equity_series_signed = sum_equity_base_series_signed + sum_recalc_equity_series
    bs_display_list.append(pd.DataFrame([current_total_equity_series_signed], index=pd.MultiIndex.from_tuples([(equity_calc_id, equity_calc_name)], names=['Account ID', 'Account Name'])))
    bs_final_display_df = pd.concat(bs_display_list) if bs_display_list else pd.DataFrame()
    if not bs_final_display_df.empty: bs_final_display_df = bs_final_display_df.sort_index(axis=1)

    # --- BS Audit Check (Simplified Structure for display, robust logic) ---
    bs_audit_table_for_display = pd.DataFrame()
    balance_check_series_for_logic = pd.Series(dtype=float) # For the success/fail message

    if not bs_final_display_df.empty :
        # Signed sums for the actual balance check A_signed + L_signed + E_signed = 0
        actual_assets_total = sum_assets_series_signed
        actual_liabilities_total = sum_liabilities_series_signed
        actual_equity_total = current_total_equity_series_signed

        balance_check_series_for_logic = actual_assets_total + actual_liabilities_total + actual_equity_total
        
        # Create the table for display as requested
        bs_audit_table_for_display = pd.DataFrame({
            "Total Assets": actual_assets_total,       # Signed value
            "Total Liabilities": actual_liabilities_total, # Signed value
            "Total Equity": actual_equity_total,          # Signed value
            "Audit Check (A = L + E)": balance_check_series_for_logic # This sum should be 0
        }).T
        # The caption explaining this will be in the UI section
    else:
        st.info("Balance Sheet is empty, cannot perform audit check.")

    return pl_display_df, bs_final_display_df, net_income_series_for_recalc, bs_audit_table_for_display, balance_check_series_for_logic

# --- Session State Initialization --- (no changes)
if 'uploaded_excel_file_info' not in st.session_state: st.session_state.uploaded_excel_file_info = None
if 'sheet_names' not in st.session_state: st.session_state.sheet_names = []
if 'selected_je_sheets' not in st.session_state: st.session_state.selected_je_sheets = []
if 'compiled_je_df' not in st.session_state: st.session_state.compiled_je_df = None
if 'column_mappings' not in st.session_state: 
    st.session_state.column_mappings = {
        "account_id": None, "mapping": None, "amount": None, "date": None, "account_type": None
    }
if 'show_statements' not in st.session_state: st.session_state.show_statements = False
if 'debit_column_selection' not in st.session_state: st.session_state.debit_column_selection = None
if 'credit_column_selection' not in st.session_state: st.session_state.credit_column_selection = None
if 'roll_equity_selection' not in st.session_state: st.session_state.roll_equity_selection = False
if 'roll_bs_selection' not in st.session_state: st.session_state.roll_bs_selection = False
if 'pnl_sign_convention_selection' not in st.session_state:
    st.session_state.pnl_sign_convention_selection = "RevNeg_ExpPos"

# --- UI Sections ---
# ... (File Upload, Tab Selection, Debit/Credit Expander, Column Mapping UI as before) ...
st.subheader("1. Upload Excel File")
uploaded_file_widget = st.file_uploader("Choose an Excel file (.xlsx or .xls)", type=["xlsx", "xls"], key="file_uploader_data_upload")
if uploaded_file_widget:
    if st.session_state.uploaded_excel_file_info is None or st.session_state.uploaded_excel_file_info['name'] != uploaded_file_widget.name or st.session_state.uploaded_excel_file_info['size'] != uploaded_file_widget.size:
        st.session_state.uploaded_excel_file_info = {'name': uploaded_file_widget.name,'size': uploaded_file_widget.size,'data': BytesIO(uploaded_file_widget.getvalue())}
        st.session_state.sheet_names = [] ; st.session_state.selected_je_sheets = [] ; st.session_state.compiled_je_df = None
        st.session_state.column_mappings = {key: None for key in st.session_state.column_mappings}
        st.session_state.show_statements = False; st.session_state.debit_column_selection = None; st.session_state.credit_column_selection = None
        st.session_state.roll_equity_selection = False; st.session_state.roll_bs_selection = False
        st.session_state.pnl_sign_convention_selection = "RevNeg_ExpPos"
        st.success(f"Uploaded '{uploaded_file_widget.name}' successfully.")
    excel_file_buffer = st.session_state.uploaded_excel_file_info['data']; excel_file_buffer.seek(0)
    try:
        if not st.session_state.sheet_names: st.session_state.sheet_names = pd.ExcelFile(excel_file_buffer).sheet_names
    except Exception as e: st.error(f"Could not read sheet names: {e}"); st.session_state.uploaded_excel_file_info = None; st.stop()

if st.session_state.uploaded_excel_file_info and st.session_state.sheet_names: # Tab Selection
    st.subheader("2. Select Sheets for Journal Entries (JEs)")
    current_selection = st.session_state.get('selected_je_sheets', [])
    if not isinstance(current_selection, list): current_selection = []
    st.session_state.selected_je_sheets = st.multiselect( "Which sheet(s) contain your JE data?", options=st.session_state.sheet_names, default=current_selection)
    if st.session_state.selected_je_sheets:
        if st.button("Compile Selected JE Sheets", key="compile_je_btn"): # Compile button
            st.session_state.compiled_je_df = None; st.session_state.show_statements = False
            st.session_state.debit_column_selection = None; st.session_state.credit_column_selection = None
            st.session_state.column_mappings["amount"] = None; all_dfs = []
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
                    df_cols_lower = [str(col).lower() for col in st.session_state.compiled_je_df.columns]
                    original_cols = list(st.session_state.compiled_je_df.columns)
                    if "debit" in df_cols_lower and st.session_state.debit_column_selection is None:
                        st.session_state.debit_column_selection = original_cols[df_cols_lower.index("debit")]
                    if "credit" in df_cols_lower and st.session_state.credit_column_selection is None:
                        st.session_state.credit_column_selection = original_cols[df_cols_lower.index("credit")]
                except Exception as e: st.error(f"Error concatenating JE sheets: {e}")
            else: st.warning("No dataframes were successfully read from the selected sheets.")
            st.rerun()
            
if st.session_state.compiled_je_df is not None: # JE Preview and D/C handler
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
        if st.button("Create 'Total Amount' Column from Debits/Credits", key="create_total_amount_btn_exp"):
            debit_col_to_use = st.session_state.debit_column_selection; credit_col_to_use = st.session_state.credit_column_selection
            if debit_col_to_use and credit_col_to_use:
                temp_df = st.session_state.compiled_je_df.copy()
                try:
                    debit_val = pd.to_numeric(temp_df[debit_col_to_use], errors='coerce').fillna(0)
                    credit_val = pd.to_numeric(temp_df[credit_col_to_use], errors='coerce').fillna(0)
                    temp_df["Total Amount"] = debit_val - credit_val
                    st.session_state.compiled_je_df = temp_df; st.session_state.column_mappings["amount"] = "Total Amount"
                    st.success("Successfully created 'Total Amount' column."); st.rerun() 
                except KeyError as e: st.error(f"Error creating 'Total Amount': Column '{e}' not found.")
                except Exception as e: st.error(f"Error creating 'Total Amount' column: {e}")
            else: st.warning("Please select both Debit and Credit columns.")
    st.markdown("---")

    st.subheader("3. Map Your Data Columns") # Column Mapping
    st.markdown("Select the columns from your uploaded data that correspond to the required financial dimensions.")
    pnl_convention_options_map = {
        "RevNeg_ExpPos": "Revenue Negative (Credits) / COGS & Expenses Positive (Debits)",
        "RevPos_ExpNeg": "Revenue Positive / COGS & Expenses Negative"}
    options_keys = list(pnl_convention_options_map.keys())
    current_pnl_convention_idx = options_keys.index(st.session_state.pnl_sign_convention_selection) if st.session_state.pnl_sign_convention_selection in options_keys else 0
    st.session_state.pnl_sign_convention_selection = st.radio("P&L Value Representation:", options=options_keys, index=current_pnl_convention_idx, format_func=lambda x: pnl_convention_options_map[x], key="pnl_sign_radio", horizontal=True)
    df_columns_for_mapping = [""] + list(st.session_state.compiled_je_df.columns)
    mappings = st.session_state.column_mappings; col_map_1, col_map_2 = st.columns(2)
    with col_map_1:
        def get_idx(key_name): val = mappings.get(key_name); return df_columns_for_mapping.index(val) if val and val in df_columns_for_mapping else 0
        mappings["account_id"] = st.selectbox("Account ID Column:", options=df_columns_for_mapping, index=get_idx("account_id"), help="Column containing GL account numbers.")
        if "Total Amount" in df_columns_for_mapping and mappings.get("amount") == "Total Amount": st.write(f"**Amount Column:** `Total Amount`")
        else: mappings["amount"] = st.selectbox("Amount Column:", options=df_columns_for_mapping, index=get_idx("amount"), help="Numerical column for transaction amounts.")
        mappings["account_type"] = st.selectbox("Account Type Column (Optional for BS Audit):", options=df_columns_for_mapping, index=get_idx("account_type"), help="Column specifying if account is 'Asset', 'Liability', Equity', 'Revenue', or 'Expense'.")
    with col_map_2:
        mappings["mapping"] = st.selectbox("Primary Account Mapping Column:", options=df_columns_for_mapping, index=get_idx("mapping"), help="Column for P&L/BS line item descriptions.")
        mappings["date"] = st.selectbox("Date Column:", options=df_columns_for_mapping, index=get_idx("date"), help="Column containing transaction dates.")
    st.session_state.roll_bs_selection = st.checkbox("Roll forward all Balance Sheet account balances (Yearly YTD from activity)", value=st.session_state.get('roll_bs_selection', False), key="roll_bs_checkbox")
    st.session_state.roll_equity_selection = st.checkbox("Roll Net Income into Equity (Adds 'Calculated Retained Earnings' to BS)", value=st.session_state.get('roll_equity_selection', False), key="roll_ni_checkbox")
    st.session_state.column_mappings = mappings
    if st.button("Generate Financial Statements", key="generate_statements_btn"):
        chosen_amount_col = mappings.get("amount")
        if not chosen_amount_col or chosen_amount_col not in st.session_state.compiled_je_df.columns:
             st.error(f"Selected Amount column ('{chosen_amount_col}') is not valid or not found."); st.session_state.show_statements = False
        elif all(st.session_state.column_mappings[k] for k in ["account_id", "mapping", "amount", "date"]):
            with st.spinner("Processing data and generating statements..."):
                # process_financial_data now returns 5 items
                pl_df_final, bs_df_final, _, \
                bs_audit_table_for_display, internal_bs_check_series = process_financial_data(
                    st.session_state.compiled_je_df, st.session_state.column_mappings["account_id"],
                    st.session_state.column_mappings["mapping"], st.session_state.column_mappings["amount"],
                    st.session_state.column_mappings["date"], st.session_state.column_mappings["account_type"],
                    st.session_state.roll_equity_selection, st.session_state.roll_bs_selection,
                    st.session_state.pnl_sign_convention_selection)
            st.session_state.pl_wide_df_processed = pl_df_final
            st.session_state.bs_wide_df_processed = bs_df_final
            st.session_state.bs_audit_df_processed_display = bs_audit_table_for_display # For display table
            st.session_state.internal_bs_check_series_logic = internal_bs_check_series # For logic
            st.session_state.show_statements = True; st.success("Statements generated!"); st.rerun()
        else: st.error("Please ensure all essential columns are mapped."); st.session_state.show_statements = False

# --- 4. Statement Display Section ---
if st.session_state.get('show_statements', False):
    st.subheader("4. Generated Financial Statements for Validation")
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
        bs_calculated_row_ids = [
            ("ASSETS_TOTAL", "Total Assets"), 
            ("LIAB_TOTAL", "Total Liabilities"),
            ("EQUITY_TOTAL", "Total Equity") ]
        styled_bs_df = style_financial_table(bs_df_to_display, bs_calculated_row_ids)
        st.dataframe(styled_bs_df, use_container_width=True)
        
        st.markdown("##### Balance Sheet Audit Check") # Condensed Title
        # No additional user-facing caption explaining the math here, as requested
        if bs_audit_df_for_user_display is not None and not bs_audit_df_for_user_display.empty:
            st.dataframe(bs_audit_df_for_user_display.fillna(0).style.format("{:,.0f}", na_rep="0"), use_container_width=True)
            
            # Use the internal_check_series_for_logic for the success/failure message
            if internal_check_series_for_logic is not None and not internal_check_series_for_logic.empty:
                is_balanced = np.allclose(internal_check_series_for_logic.fillna(0), 0, atol=0.51) 
                if is_balanced: 
                    st.success("Balance Sheet equation (Assets = Liabilities + Equity) holds true!")
                else: 
                    st.warning(f"Balance Sheet equation does not hold. Max difference from zero: {internal_check_series_for_logic.abs().max():,.0f}.")
            else:
                st.warning("Internal balance check data not available for final verification.") # Should not happen if audit ran
        else: st.info("BS Audit Check data is not available for display.")
    elif bs_df_to_display is not None and bs_df_to_display.empty: st.info("Balance Sheet Statement is empty.")
    else: st.warning("Balance Sheet Statement could not be generated.")

st.markdown("---")
st.caption("Note: P&L/BS classification uses GL account prefixes (>=4 for P&L). BS Audit uses GL prefixes (1:A, 2:L, 3:E) or mapped 'Account Type'.")