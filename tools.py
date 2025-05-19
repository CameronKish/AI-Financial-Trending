# tools.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field 
from typing import List, Dict, Optional 
from thefuzz import process as fuzzy_process, fuzz
from datetime import datetime as dt_datetime 
import time 
import utils # For categorization functions and constants

# --- Helper Function to Safely Get Data & Apply Transformations for Agent ---
def _get_dataframe(source: str) -> pd.DataFrame:
    """
    Retrieves, standardizes, categorizes (for JEs), and date-filters 
    the requested DataFrame (P&L or JE) for the agent.
    """
    base_df = None
    is_uploaded = False
    df_source_description = "Unknown" 

    if st.session_state.get('uploaded_data_ready', False):
        is_uploaded = True
        if source.upper() == "P&L":
            base_df = st.session_state.get('active_pl_flat_df')
            df_source_description = "P&L (Uploaded/Processed)"
        elif source.upper() == "JE":
            base_df = st.session_state.get('compiled_je_df') 
            df_source_description = "JE (Uploaded/Compiled)"
    elif st.session_state.get('data_loaded', False): 
        if source.upper() == "P&L":
            base_df = st.session_state.get('pl_flat_df')
            df_source_description = "P&L (Sample)"
        elif source.upper() == "JE":
            base_df = st.session_state.get('je_detail_df')
            df_source_description = "JE (Sample)"
    else:
        return pd.DataFrame()

    if base_df is None or base_df.empty:
        return pd.DataFrame()
    
    df = base_df.copy()
    
    standardized_date_col_for_filtering = 'Date' 
    
    if source.upper() == "JE":
        standardized_date_col_for_filtering = 'Transaction Date'
        std_names_je = {
            'account_id': 'Account ID', 'mapping': 'Account Description',
            'amount': 'Amount', 'date': 'Transaction Date'
        }
        other_useful_je_cols = ['Memo', 'Customer', 'Transaction Id'] 
        current_renames = {}

        if is_uploaded:
            user_mappings = st.session_state.get('column_mappings', {})
            for key, standard_name in std_names_je.items():
                user_col_name = user_mappings.get(key)
                if user_col_name and user_col_name in df.columns:
                    current_renames[user_col_name] = standard_name
        else: 
            sample_config = st.session_state.get('column_config', {})
            if sample_config.get("JE_ID") and sample_config["JE_ID"] in df.columns: current_renames[sample_config["JE_ID"]] = 'Account ID'
            sample_je_acc_name_col = sample_config.get("JE_ACCOUNT_NAME")
            if sample_je_acc_name_col and sample_je_acc_name_col in df.columns: current_renames[sample_je_acc_name_col] = 'Account Description'
            elif 'Account Name' in df.columns and 'Account Description' not in current_renames.values(): current_renames['Account Name'] = 'Account Description'
            if sample_config.get("JE_AMOUNT") and sample_config["JE_AMOUNT"] in df.columns: current_renames[sample_config["JE_AMOUNT"]] = 'Amount'
            if sample_config.get("JE_DATE") and sample_config["JE_DATE"] in df.columns: current_renames[sample_config["JE_DATE"]] = 'Transaction Date'
        
        df.rename(columns=current_renames, inplace=True)
        
        if 'Transaction Date' in df.columns:
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
        
        final_columns_for_agent_je = [name for name in std_names_je.values() if name in df.columns]

        if 'Account ID' in df.columns and is_uploaded : # Categorization for uploaded data
            try:
                use_rules = st.session_state.get('use_custom_ranges', False)
                gl_ranges = st.session_state.get('parsed_gl_ranges', {})
                df['_Temp_Categorization_AccID_'] = df['Account ID'].astype(str).str.strip()
                df['Assigned Category'] = df['_Temp_Categorization_AccID_'].apply(
                    lambda x: utils.assign_category_using_rules(x, use_rules, gl_ranges, utils.get_prefix_based_category)
                )
                df['Statement Section'] = df['Assigned Category'].apply(utils.get_statement_section)
                df.drop(columns=['_Temp_Categorization_AccID_'], inplace=True, errors='ignore')
                other_useful_je_cols.extend(['Assigned Category', 'Statement Section'])
            except Exception as e_cat: print(f"tools.py: Could not categorize JEs for agent: {e_cat}")

        for col in other_useful_je_cols:
            if col in df.columns and col not in final_columns_for_agent_je:
                final_columns_for_agent_je.append(col)
        
        original_cols_to_add_back = [c for c in base_df.columns if c not in current_renames.keys() and c in df.columns and c not in final_columns_for_agent_je]
        final_columns_for_agent_je.extend(original_cols_to_add_back)
        df = df[[col for col in list(dict.fromkeys(final_columns_for_agent_je)) if col in df.columns]]

        ai_je_scope = st.session_state.get('ai_je_scope_filter', "All JEs")
        if 'Statement Section' in df.columns and ai_je_scope != "All JEs" and is_uploaded: # Scope filter applies to uploaded categorized JEs
            if ai_je_scope == "P&L JEs Only": df = df[df['Statement Section'] == 'P&L']
            elif ai_je_scope == "BS JEs Only": df = df[df['Statement Section'] == 'BS']
            df_source_description += f" (Scope: {ai_je_scope})"

    elif source.upper() == "P&L":
        standardized_date_col_for_filtering = 'Date'
        if is_uploaded: 
            if '_Account_Type_' in df.columns: df.rename(columns={'_Account_Type_': 'Account Type'}, inplace=True)
            if 'Period_dt' in df.columns: df.rename(columns={'Period_dt': 'Date'}, inplace=True)
            if 'Period' in df.columns: df.rename(columns={'Period': 'Period String'}, inplace=True)
            final_cols_pl = ['Account ID', 'Account Name', 'Account Type', 'Date', 'Period String', 'Amount']
            df = df[[col for col in final_cols_pl if col in df.columns]]
        else: 
            col_config = st.session_state.get('column_config', {})
            renamed_cols_pl = {}
            if col_config.get("PL_ID") and col_config["PL_ID"] in df.columns: renamed_cols_pl[col_config.get("PL_ID")] = 'Account ID'
            if col_config.get("PL_MAP_DISPLAY") and col_config["PL_MAP_DISPLAY"] in df.columns: renamed_cols_pl[col_config.get("PL_MAP_DISPLAY")] = 'Account Name'
            if 'Period_dt' in df.columns: renamed_cols_pl['Period_dt'] = 'Date'
            if 'Period_Str' in df.columns: renamed_cols_pl['Period_Str'] = 'Period String'
            elif 'Period' in df.columns and 'Period_Str' not in df.columns and 'Period String' not in df.columns : renamed_cols_pl['Period'] = 'Period String'
            df.rename(columns=renamed_cols_pl, inplace=True)
            final_cols_pl_sample = ['Account ID', 'Account Name', 'Date', 'Period String', 'Amount']
            df = df[[col for col in final_cols_pl_sample if col in df.columns]]
        if 'Date' in df.columns:
             df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    ai_start_date_val = st.session_state.get('ai_filter_start_date')
    ai_end_date_val = st.session_state.get('ai_filter_end_date')
    if standardized_date_col_for_filtering and standardized_date_col_for_filtering in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[standardized_date_col_for_filtering]):
            df[standardized_date_col_for_filtering] = pd.to_datetime(df[standardized_date_col_for_filtering], errors='coerce')
        if (ai_start_date_val or ai_end_date_val) and df[standardized_date_col_for_filtering].isnull().any():
            df.dropna(subset=[standardized_date_col_for_filtering], inplace=True)
        if not df.empty:
            if ai_start_date_val:
                start_filter_ts = pd.to_datetime(ai_start_date_val).normalize()
                df = df[df[standardized_date_col_for_filtering] >= start_filter_ts]
            if ai_end_date_val:
                end_filter_ts = pd.to_datetime(ai_end_date_val).normalize()
                df = df[df[standardized_date_col_for_filtering] <= end_filter_ts]
            # if ai_start_date_val or ai_end_date_val:
            #     df_source_description += " (Date Filtered for AI)"
    elif ai_start_date_val or ai_end_date_val:
        print(f"tools.py: Date column '{standardized_date_col_for_filtering}' for {source} not usable for filtering.")
    
    # st.sidebar.caption(f"Agent using: {df_source_description}, {len(df)} rows final.") # Optional debug for sidebar
    return df.reset_index(drop=True)

class GetSchemaArgs(BaseModel):
    source: str = Field(..., description="The data source ('P&L' or 'JE') to get the schema for. Data reflects AI Insights filters and standardized column names.")
class SetActiveDataFrameArgs(BaseModel):
    source: str = Field(..., description="Data source ('P&L' or 'JE') to load as active, reflecting standardizations and AI Insights date filters.")
class FindAccountArgs(BaseModel):
    account_query: str = Field(..., description="Name/description to search. P&L: 'Account Name'. JE: 'Account Description'.")
    source: str = Field(..., description="Data source ('P&L' or 'JE').")
    min_score_threshold: int = Field(default=85, description="Min matching score (0-100).")
class FilterArgs(BaseModel):
    source: str = Field(..., description="Original data source ('P&L' or 'JE') to filter. Data standardized and date-filtered first.")
    conditions: str = Field(..., description="Pandas query on standardized columns (e.g., `Account ID`, `Account Description`, `Transaction Date`, `Amount`, `Account Type`, `Assigned Category`, `Statement Section`). Use `find_exact_account_name` first for names.")
class FuzzyFilterArgs(BaseModel):
    source: str = Field(..., description="Original data source ('P&L' or 'JE'). Data standardized and date-filtered first.")
    text_column_to_search: str = Field(..., description="Standardized text column (e.g., 'Account Description', 'Memo', 'Customer') to search.")
    query_text: str = Field(..., description="Text to search for (inexact match).")
    score_threshold: int = Field(default=80, description="Min fuzzy match score (0-100).")
class AggregateArgs(BaseModel):
    group_by_columns: List[str] = Field(..., description="List of standardized column names from active DataFrame to group by.")
    agg_specs: Dict[str, str] = Field(..., description="Dict: keys are standardized columns to aggregate (e.g., 'Amount'), values are functions ('sum', 'count', etc.).")
class SortArgs(BaseModel):
    sort_by_columns: List[str] = Field(..., description="List of standardized column names from active DataFrame to sort by.")
    ascending: bool = Field(default=True, description="Sort order.")
class TopNArgs(BaseModel):
    n: int = Field(..., description="Number of top rows from active DataFrame.")
class PlotArgs(BaseModel):
    plot_type: str = Field(..., description="Type of plot ('bar', 'line', or 'pie').")
    x_col_or_names: str = Field(..., description="Standardized column from active DF for X-axis (bar/line) or names/labels (pie).")
    y_col_or_values: List[str] = Field(..., description="Standardized column(s) from active DF for Y-axis (bar/line) or a single column for values (pie). Typically 'Amount'.")
    title: str = Field(..., description="Title for the plot.")
class SummarizeArgs(BaseModel): pass 
class FindNewItemsArgs(BaseModel):
    source: str = Field(..., description="Data source ('P&L' or 'JE'). Data standardized and date-filtered first.")
    item_column: str = Field(..., description="Standardized column with items (e.g., 'Account Name', 'Customer', 'Account ID').")
    date_column: str = Field(..., description="Standardized date column ('Date' for P&L, 'Transaction Date' for JE).")
    year_to_check: int = Field(..., description="The year to identify new items in (e.g., 2023).")
class AnalyzeVarianceArgs(BaseModel):
    source: str = Field(..., description="Data source ('P&L' or 'JE'). Data standardized and date-filtered first.")
    date_column: str = Field(..., description="Standardized date column ('Date' for P&L, 'Transaction Date' for JE).")
    metric_column: str = Field(default='Amount', description="Numeric column for variance (typically 'Amount').")
    category_column: str = Field(..., description="Standardized column to group by (e.g., 'Account Name', 'Account Description', 'Memo').")
    period1_start_date: str = Field(..., description="P1 start (YYYY-MM-DD). These dates are independent of AI sidebar date filter.")
    period1_end_date: str = Field(..., description="P1 end (YYYY-MM-DD).")
    period2_start_date: str = Field(..., description="P2 start (YYYY-MM-DD).")
    period2_end_date: str = Field(..., description="P2 end (YYYY-MM_DD).")
    account_name_filter: Optional[str] = Field(None, description="Optional: Exact account name/description. Use find_exact_account_name first.")
    top_n_drivers: int = Field(default=5, description="Number of top drivers.")

INTERMEDIATE_DF_KEY = "agent_intermediate_df" 

@tool("get_data_schema", args_schema=GetSchemaArgs)
def get_data_schema(source: str) -> str:
    """Gets column names and data types for the specified data source ('P&L' or 'JE'). Data reflects AI Insights date/scope filters and standardized column names."""
    try:
        df = _get_dataframe(source) 
        if df.empty: return f"Schema for {source}: No data available or source is empty (after filters)."
        schema_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
        return f"Schema for {source} data ({len(df)} rows after filters):\n{json.dumps(schema_dict, indent=2)}"
    except Exception as e: return f"Error getting schema for {source}: {str(e)}"

@tool("set_active_dataframe", args_schema=SetActiveDataFrameArgs)
def set_active_dataframe(source: str) -> str:
    """Loads the specified original data source ('P&L' or 'JE') as the active DataFrame, applying AI Insights filters and standardizing columns. Subsequent tools operate on this active DataFrame."""
    try:
        df = _get_dataframe(source) 
        st.session_state[INTERMEDIATE_DF_KEY] = df
        return f"Successfully set active DataFrame to '{source}' data ({len(df)} rows). Ready for further operations."
    except Exception as e: return f"Error setting active DataFrame from '{source}': {str(e)}"

@tool("find_exact_account_name", args_schema=FindAccountArgs)
def find_exact_account_name(account_query: str, source: str, min_score_threshold: int = 85) -> str:
    """Finds best matching account name/description in standardized P&L ('Account Name') or JE ('Account Description') data."""
    try:
        df = _get_dataframe(source)
        if df.empty: return f"No data in '{source}' to search for account names."
        account_col_to_search = 'Account Name' if source.upper() == 'P&L' else 'Account Description'
        if account_col_to_search not in df.columns: return f"Error: Column '{account_col_to_search}' not found in {source} data for search. Available: {df.columns.tolist()}"
        unique_accounts = df[account_col_to_search].dropna().unique().astype(str)
        if not unique_accounts.size: return f"No unique values in '{account_col_to_search}' for {source}."
        best_match = fuzzy_process.extractOne(account_query, unique_accounts)
        if best_match and best_match[1] >= min_score_threshold: return f"Found: '{best_match[0]}' (Score: {best_match[1]}%) in '{account_col_to_search}' of '{source}'."
        return f"No match >= {min_score_threshold}% in '{account_col_to_search}' of '{source}'. Best: '{best_match[0]}' ({best_match[1]}%)."
    except Exception as e: return f"Error finding account name for '{account_query}' in {source}: {str(e)}"

@tool("filter_dataframe", args_schema=FilterArgs)
def filter_dataframe(source: str, conditions: str) -> str:
    """Filters an original 'P&L' or 'JE' DataFrame (after date/scope filters & standardization) using a pandas query string. Result becomes active DataFrame."""
    try:
        df = _get_dataframe(source)
        if df.empty: return f"Cannot filter: Source '{source}' is empty."
        filtered_df = df.query(conditions).copy()
        st.session_state[INTERMEDIATE_DF_KEY] = filtered_df
        return f"Filtered '{source}' on '{conditions}'. Result: {len(filtered_df)} rows. Now active."
    except Exception as e:
        cols_info = "N/A (Unable to retrieve column info)" 
        try:
            df_for_cols = _get_dataframe(source) # Try to get it again for col info
            if isinstance(df_for_cols, pd.DataFrame) and not df_for_cols.empty: cols_info = df_for_cols.columns.tolist()
            elif isinstance(df_for_cols, pd.DataFrame) and df_for_cols.empty: cols_info = "Source is empty or filtered to empty."
        except Exception as e_cols: cols_info = f"N/A (Error retrieving columns: {type(e_cols).__name__})"
        return f"Error filtering '{source}' with condition '{conditions}': {str(e)}. Available columns for '{source}' (reflecting current AI date/scope filters): {cols_info}. Ensure query uses correct column names and syntax (e.g., backticks for names with spaces)."

@tool("fuzzy_filter_by_text_in_column", args_schema=FuzzyFilterArgs)
def fuzzy_filter_by_text_in_column(source: str, text_column_to_search: str, query_text: str, score_threshold: int = 80) -> dict:
    """Filters an original 'P&L' or 'JE' source by fuzzy text match in a specified column. Result becomes active DataFrame."""
    try:
        df = _get_dataframe(source)
        if df.empty: return {"error": f"Source '{source}' is empty for fuzzy filter."}
        if text_column_to_search not in df.columns: return {"error": f"Column '{text_column_to_search}' not found in {source}. Available: {df.columns.tolist()}"}
        df[text_column_to_search] = df[text_column_to_search].astype(str).fillna('')
        scores = df[text_column_to_search].apply(lambda x: fuzz.partial_ratio(query_text.lower(), str(x).lower()))
        filtered_df = df[scores >= score_threshold].copy()
        st.session_state[INTERMEDIATE_DF_KEY] = filtered_df
        msg = f"Fuzzy filtered on '{text_column_to_search}' for '{query_text}'. Result: {len(filtered_df)} rows. Now active."
        if len(filtered_df) == 0: msg = f"No items in '{text_column_to_search}' fuzzily matched '{query_text}' (score >= {score_threshold}%). Active DF is empty."
        return {"message": msg, "audit_dataframe": filtered_df.to_dict(orient='records')}
    except Exception as e: return {"error": f"Error during fuzzy text filter: {str(e)}"}

@tool("aggregate_dataframe", args_schema=AggregateArgs)
def aggregate_dataframe(group_by_columns: List[str], agg_specs: Dict[str,str]) -> str:
    """Groups the active DataFrame by specified columns and aggregates using a dictionary of aggregation specs. Result becomes active DataFrame."""
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None: return "Error: No active DataFrame. Use 'set_active_dataframe' or 'filter_dataframe' first."
    df = st.session_state[INTERMEDIATE_DF_KEY]
    if df.empty: st.session_state[INTERMEDIATE_DF_KEY] = df; return "Cannot aggregate empty DataFrame."
    try:
        missing_group = [col for col in group_by_columns if col not in df.columns]
        if missing_group: return f"Error: Group by col(s) not found: {missing_group}. Available: {df.columns.tolist()}"
        missing_agg_keys = [col for col in agg_specs.keys() if col not in df.columns]
        if missing_agg_keys: return f"Error: Aggregation key col(s) not found: {missing_agg_keys}. Available: {df.columns.tolist()}"
        aggregated_df = df.groupby(group_by_columns, as_index=False, dropna=False).agg(agg_specs)
        st.session_state[INTERMEDIATE_DF_KEY] = aggregated_df
        return f"Aggregated data. Result: {len(aggregated_df)} rows. Now active."
    except Exception as e: return f"Error aggregating: {str(e)}. Check agg_specs. Cols: {df.columns.tolist()}"

@tool("sort_dataframe", args_schema=SortArgs)
def sort_dataframe(sort_by_columns: List[str], ascending: bool) -> str:
    """Sorts the active DataFrame by specified columns. Sorted DataFrame becomes active."""
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None: return "Error: No active DataFrame."
    df = st.session_state[INTERMEDIATE_DF_KEY];
    if df.empty: return "Cannot sort empty DataFrame."
    try:
        # Ensure all sort_by_columns exist
        missing_cols = [col for col in sort_by_columns if col not in df.columns]
        if missing_cols: return f"Error sorting: Column(s) not found: {missing_cols}. Available: {df.columns.tolist()}"
        
        sorted_df = df.sort_values(by=sort_by_columns, ascending=ascending)
        st.session_state[INTERMEDIATE_DF_KEY] = sorted_df
        return f"Sorted by {sort_by_columns} ({'asc' if ascending else 'desc'})."
    except KeyError as e: return f"Error sorting: Column not found - {str(e)}. Available: {df.columns.tolist()}" # Should be caught by above check
    except Exception as e: return f"Error sorting: {str(e)}"

@tool("get_top_n", args_schema=TopNArgs)
def get_top_n(n: int) -> str:
    """Selects top 'n' rows from active DataFrame. Result becomes active."""
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None: return "Error: No active DataFrame."
    df = st.session_state[INTERMEDIATE_DF_KEY];
    if n <= 0: return "Error: N must be positive."
    top_n_df = df.head(n); st.session_state[INTERMEDIATE_DF_KEY] = top_n_df
    return f"Selected top {n} rows. Result: {len(top_n_df)} rows."

@tool("plot_dataframe", args_schema=PlotArgs)
def plot_dataframe(plot_type: str, x_col_or_names: str, y_col_or_values: List[str], title: str) -> dict:
    """Generates plot ('bar', 'line', 'pie') from the FINAL active DataFrame."""
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None: return {"error": "No final active DataFrame to plot."}
    df_to_plot = st.session_state[INTERMEDIATE_DF_KEY]
    if not isinstance(df_to_plot, pd.DataFrame) or df_to_plot.empty: return {"error": "Active DataFrame to plot is empty/invalid."}
    if x_col_or_names not in df_to_plot.columns: return {"error": f"X-axis/Names column '{x_col_or_names}' not found. Available: {df_to_plot.columns.tolist()}"}
    for y_c in y_col_or_values:
        if y_c not in df_to_plot.columns: return {"error": f"Y-axis/Values column '{y_c}' not found. Available: {df_to_plot.columns.tolist()}"}
    audit_data = df_to_plot.to_dict(orient='records'); fig = None; plot_type = plot_type.lower()
    try:
        if plot_type == 'bar': 
            y_arg = y_col_or_values[0] if len(y_col_or_values)==1 else y_col_or_values
            fig = px.bar(df_to_plot, x=x_col_or_names, y=y_arg, title=title, template="plotly_white")
        elif plot_type == 'line': 
            fig = px.line(df_to_plot, x=x_col_or_names, y=y_col_or_values, title=title, markers=True, template="plotly_white")
        elif plot_type == 'pie': 
            if not y_col_or_values: return {"error": "Pie chart needs a values column in y_col_or_values."}
            val_col = y_col_or_values[0]
            if not pd.api.types.is_numeric_dtype(df_to_plot[val_col]): return {"error": f"Pie chart values column '{val_col}' must be numeric."}
            plot_df_pie = df_to_plot[[x_col_or_names, val_col]].copy()
            plot_df_pie = plot_df_pie[plot_df_pie[val_col].abs() > 1e-6] 
            if plot_df_pie.empty: return {"error": f"No significant data in '{val_col}' for pie chart."}
            PIE_SLICE_THRESHOLD = 10 
            if len(plot_df_pie[x_col_or_names].unique()) > PIE_SLICE_THRESHOLD:
                plot_df_pie = plot_df_pie.sort_values(by=val_col, ascending=False, key=abs) 
                top_df = plot_df_pie.head(PIE_SLICE_THRESHOLD - 1)
                other_sum = plot_df_pie.iloc[PIE_SLICE_THRESHOLD - 1:][val_col].sum()
                if abs(other_sum) > 1e-6: 
                    other_row = pd.DataFrame([{x_col_or_names: "Other", val_col: other_sum}])
                    plot_df_pie = pd.concat([top_df, other_row], ignore_index=True)
                else: plot_df_pie = top_df
            fig = px.pie(plot_df_pie, names=x_col_or_names, values=val_col, title=title, template="plotly_white")
        else: return {"error": f"Unsupported plot type: '{plot_type}'. Use 'bar', 'line', or 'pie'."}
        if fig: 
            plot_json = fig.to_json()
            if plot_type != 'pie': fig.update_layout(yaxis_tickformat=",.0f")
            return {"plot_json": plot_json, "audit_dataframe": audit_data, "message": f"Generated '{title}' {plot_type} plot."}
        return {"error": "Plot figure not generated."}
    except Exception as e: return {"error": f"Failed to generate plot: {str(e)}"}

@tool("summarize_dataframe", args_schema=SummarizeArgs)
def summarize_dataframe() -> dict:
    """Provides a text summary of the FINAL active DataFrame. Includes column names, row count, and basic statistics for numerical and non-numerical columns."""
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None: 
        return {"error": "No final active DataFrame to summarize."}
    df = st.session_state[INTERMEDIATE_DF_KEY]
    if not isinstance(df, pd.DataFrame): 
        return {"error": "Active data is not a valid DataFrame."}
    
    audit_data = df.to_dict(orient='records')
    summary_lines = [f"Summary of final active DataFrame with {len(df)} rows and {len(df.columns)} columns:"]
    
    if not df.empty:
        summary_lines.append(f"Columns: {df.columns.tolist()}")
        
        num_df = df.select_dtypes(include=np.number)
        if not num_df.empty:
            summary_lines.append("\nNumerical Column Statistics:")
            summary_lines.append(num_df.describe().to_string())
        else:
            summary_lines.append("\nNo numerical columns found in the active DataFrame.")
            
        non_num_df = df.select_dtypes(exclude=np.number)
        if not non_num_df.empty:
            summary_lines.append("\nNon-Numerical Column Info:")
            for col in non_num_df.columns:
                unique_count = non_num_df[col].nunique()
                top_val_series = non_num_df[col].mode()
                top_val = str(top_val_series[0]) if not top_val_series.empty else "N/A"
                summary_lines.append(f"- Column '{col}': {unique_count} unique values. Top value: {top_val} (others may exist if multiple modes).")
        else:
            summary_lines.append("\nNo non-numerical columns found in the active DataFrame.")
    else:
        summary_lines.append("The final active DataFrame is empty.")
        
    return {"summary_text": "\n".join(summary_lines), "audit_dataframe": audit_data, "message": "Text summary generated."}

@tool("find_new_items_in_period", args_schema=FindNewItemsArgs)
def find_new_items_in_period(source: str, item_column: str, date_column: str, year_to_check: int) -> dict:
    """Identifies items (e.g., Customers, Products from 'item_column') that appeared for the first time in 'year_to_check' within the specified 'source' (P&L or JE). Uses the standardized 'date_column'. Result becomes active DataFrame."""
    try:
        df = _get_dataframe(source) 
        if df.empty: return {"error": f"Source '{source}' is empty."}
        if date_column not in df.columns: return {"error": f"Date column '{date_column}' not in {source}. Available: {df.columns.tolist()}"}
        if item_column not in df.columns: return {"error": f"Item column '{item_column}' not in {source}. Available: {df.columns.tolist()}"}
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]): return {"error": f"Date column '{date_column}' is not datetime."}
        
        current_items = df[df[date_column].dt.year == year_to_check][item_column].dropna().unique()
        prior_items = df[df[date_column].dt.year < year_to_check][item_column].dropna().unique()
        new_items_list = sorted(list(set(current_items) - set(prior_items)))
        res_df = pd.DataFrame({f"New_{item_column.replace(' ', '_')}_in_{year_to_check}": new_items_list})
        st.session_state[INTERMEDIATE_DF_KEY] = res_df 
        msg = f"Found {len(new_items_list)} new '{item_column}' items in {year_to_check}."
        return {"summary_text": msg, "audit_dataframe": res_df.to_dict(orient='records'), "message": msg + " This result is now active."}
    except Exception as e: return {"error": f"Error finding new items: {str(e)}"}

@tool("analyze_period_variance_drivers", args_schema=AnalyzeVarianceArgs)
def analyze_period_variance_drivers(source: str, date_column: str, metric_column: str, category_column: str,
                                   period1_start_date: str, period1_end_date: str,
                                   period2_start_date: str, period2_end_date: str,
                                   account_name_filter: Optional[str] = None, top_n_drivers: int = 5) -> dict:
    """Analyzes variance of 'metric_column' (typically 'Amount') between two specific date periods from standardized P&L or JE data, grouped by 'category_column'. Result becomes active DataFrame."""
    try:
        df_full = _get_dataframe(source) 
        if df_full.empty: return {"error": f"Source '{source}' is empty after initial filters."}
        req_cols = [date_column, metric_column, category_column]
        account_desc_col_std = 'Account Name' if source.upper() == 'P&L' else 'Account Description'
        if account_name_filter and account_desc_col_std not in df_full.columns: return {"error": f"Account filter column '{account_desc_col_std}' not in {source}."}
        for col in req_cols:
            if col not in df_full.columns: return {"error": f"Required column '{col}' not found in {source}. Available: {df_full.columns.tolist()}"}
        if not pd.api.types.is_datetime64_any_dtype(df_full[date_column]): return {"error": f"Date column '{date_column}' is not datetime."}
        df_full[metric_column] = pd.to_numeric(df_full[metric_column], errors='coerce').fillna(0)
        df_analysis = df_full
        if account_name_filter: 
            df_analysis = df_analysis[df_analysis[account_desc_col_std] == account_name_filter].copy()
            if df_analysis.empty: return {"error": f"No data for account filter '{account_name_filter}' in '{account_desc_col_std}'."}
        p1s,p1e = pd.to_datetime(period1_start_date).normalize(), pd.to_datetime(period1_end_date).normalize()
        p2s,p2e = pd.to_datetime(period2_start_date).normalize(), pd.to_datetime(period2_end_date).normalize()
        df_p1 = df_analysis[(df_analysis[date_column] >= p1s) & (df_analysis[date_column] <= p1e)]
        agg_p1 = df_p1.groupby(category_column, dropna=False, as_index=False)[metric_column].sum().rename(columns={metric_column: 'P1_Value'})
        df_p2 = df_analysis[(df_analysis[date_column] >= p2s) & (df_analysis[date_column] <= p2e)]
        agg_p2 = df_p2.groupby(category_column, dropna=False, as_index=False)[metric_column].sum().rename(columns={metric_column: 'P2_Value'})
        if agg_p1.empty and agg_p2.empty:
            empty_drivers_df = pd.DataFrame(columns=[category_column, 'P1_Value', 'P2_Value', 'Variance', 'Abs_Variance'])
            st.session_state[INTERMEDIATE_DF_KEY] = empty_drivers_df
            return {"summary_text": f"No data for either period for '{metric_column}' by '{category_column}'.", "audit_dataframe": []}
        merged_df = pd.merge(agg_p1, agg_p2, on=category_column, how='outer').fillna(0)
        merged_df['Variance'] = merged_df['P2_Value'] - merged_df['P1_Value'] 
        merged_df['Abs_Variance'] = merged_df['Variance'].abs()
        drivers_df = merged_df[merged_df['Abs_Variance'] > 1e-6].sort_values(by='Abs_Variance', ascending=False).head(top_n_drivers)
        st.session_state[INTERMEDIATE_DF_KEY] = drivers_df
        p1_str, p2_str = f"{period1_start_date} to {period1_end_date}", f"{period2_start_date} to {period2_end_date}"
        title_desc = f"metric '{metric_column}' by '{category_column}'";
        if account_name_filter: title_desc = f"account '{account_name_filter}', {title_desc}"
        summary_text_lines = [f"Top {len(drivers_df)} drivers of variance for {title_desc} (P2 [{p2_str}] vs P1 [{p1_str}]):"]
        if drivers_df.empty: summary_text_lines.append("No significant variance or common categories found.")
        else:
            for _, r_driver in drivers_df.iterrows(): summary_text_lines.append(f"- {str(r_driver[category_column])}: Var={r_driver['Variance']:,.2f} (P2:{r_driver['P2_Value']:,.2f}, P1:{r_driver['P1_Value']:,.2f})")
        full_summary_text = "\n".join(summary_text_lines)
        return {"summary_text": full_summary_text, "audit_dataframe": drivers_df.to_dict(orient='records'), "message": full_summary_text + " This result is now active."}
    except Exception as e: return {"error": f"Error analyzing variance: {str(e)}"}

all_tools = [
    get_data_schema, set_active_dataframe, find_exact_account_name, filter_dataframe,
    fuzzy_filter_by_text_in_column, aggregate_dataframe, sort_dataframe, get_top_n,
    plot_dataframe, summarize_dataframe, find_new_items_in_period, analyze_period_variance_drivers,
]