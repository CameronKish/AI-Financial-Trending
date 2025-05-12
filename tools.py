#tools.py
# **** START OF FULL SCRIPT ****
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field # Explicit v1 import
from typing import List, Dict, Optional # For type hinting

# --- NEW: Import fuzzy matching ---
from thefuzz import process as fuzzy_process
from thefuzz import fuzz # For direct ratio usage
from datetime import datetime as dt_datetime # Alias to avoid conflict
import time # <-- IMPORT TIME for timing

# --- Helper Function to Safely Get Data ---
def _get_dataframe(source: str) -> pd.DataFrame:
    """Safely retrieves the requested DataFrame from session state and returns a copy."""
    source = source.strip().upper()
    if source == "P&L":
        if 'pl_flat_df' in st.session_state and isinstance(st.session_state.pl_flat_df, pd.DataFrame):
            return st.session_state.pl_flat_df.copy()
        else:
            raise ValueError("P&L DataFrame ('pl_flat_df') not found or not a DataFrame in session state.")
    elif source == "JE":
        if 'je_detail_df' in st.session_state and isinstance(st.session_state.je_detail_df, pd.DataFrame):
            return st.session_state.je_detail_df.copy()
        else:
            raise ValueError("JE DataFrame ('je_detail_df') not found or not a DataFrame in session state.")
    else:
        raise ValueError(f"Unknown data source specified: {source}. Use 'P&L' or 'JE'.")

# --- Pydantic Models for Tool Arguments ---

class GetSchemaArgs(BaseModel):
    source: str = Field(..., description="The data source ('P&L' or 'JE') to get the schema for.")

# +++++ ADD THIS NEW PYDANTIC MODEL +++++
class SetActiveDataFrameArgs(BaseModel):
    source: str = Field(..., description="The data source ('P&L' or 'JE') to load as the active DataFrame for subsequent operations.")
# +++++++++++++++++++++++++++++++++++++++

class FindAccountArgs(BaseModel):
    account_query: str = Field(..., description="The account name or description provided by the user.")
    source: str = Field(..., description="The data source ('P&L' or 'JE') to find the account name in.")
    min_score_threshold: int = Field(default=85, description="Minimum matching score (0-100).")

class FilterArgs(BaseModel):
    source: str = Field(..., description="The data source ('P&L' or 'JE') to filter.")
    conditions: str = Field(..., description="Pandas query string. Use exact column names (backticks `` for names with spaces). For account names, use 'find_exact_account_name' first. For fuzzy text in other columns, use 'fuzzy_filter_by_text_in_column'.")

class AggregateArgs(BaseModel):
    group_by_columns: List[str] = Field(..., description="List of column names to group by.")
    agg_specs: Dict[str, str] = Field(..., description="**Mandatory**. Dict specifying aggregation: keys are columns to aggregate, values are functions ('sum', 'count', 'mean', 'nunique', etc.).")

class SortArgs(BaseModel):
    sort_by_columns: List[str] = Field(..., description="List of column names to sort by.")
    ascending: bool = Field(default=True, description="Sort order.")

class TopNArgs(BaseModel):
    n: int = Field(..., description="Number of top rows.")

class PlotArgs(BaseModel):
    plot_type: str = Field(..., description="Type of plot ('bar', 'line', or 'pie').")
    x_col_or_names: str = Field(..., description="Column for X-axis (bar/line) or names/labels (pie).")
    y_col_or_values: List[str] = Field(..., description="Column(s) for Y-axis (bar/line) or a single column for values (pie).")
    title: str = Field(..., description="Title for the plot.")

class SummarizeArgs(BaseModel):
    pass

class FindNewItemsArgs(BaseModel):
    source: str = Field(..., description="Data source ('P&L' or 'JE').")
    item_column: str = Field(..., description="Column containing the items to check (e.g., 'Customer', 'Product').")
    date_column: str = Field(..., description="Column containing the transaction dates.")
    year_to_check: int = Field(..., description="The year to identify new items in (e.g., 2023).")

class AnalyzeVarianceArgs(BaseModel):
    source: str = Field(..., description="Data source ('P&L' or 'JE').")
    date_column: str = Field(..., description="Column with dates.")
    metric_column: str = Field(..., description="Numeric column for variance (e.g., 'Amount (Presentation Currency)').")
    category_column: str = Field(..., description="Column to group by for drivers (e.g., 'Memo').")
    period1_start_date: str = Field(..., description="P1 start (YYYY-MM-DD).")
    period1_end_date: str = Field(..., description="P1 end (YYYY-MM-DD).")
    period2_start_date: str = Field(..., description="P2 start (YYYY-MM-DD).")
    period2_end_date: str = Field(..., description="P2 end (YYYY-MM-DD).")
    account_name_filter: Optional[str] = Field(None, description="Optional: Exact account name to filter. Use find_exact_account_name tool first.")
    top_n_drivers: int = Field(default=5, description="Number of top drivers.")

class FuzzyFilterArgs(BaseModel):
    source: str = Field(..., description="Data source ('P&L' or 'JE').")
    text_column_to_search: str = Field(..., description="Text column to search (e.g., 'Memo').")
    query_text: str = Field(..., description="Text to search for (inexact match).")
    score_threshold: int = Field(default=80, description="Min fuzzy match score (0-100).")


# --- Tool Definitions ---
INTERMEDIATE_DF_KEY = "agent_intermediate_df"

@tool("get_data_schema", args_schema=GetSchemaArgs)
def get_data_schema(source: str) -> str:
    """Gets column names and data types for the specified data source ('P&L' or 'JE'). Useful for understanding data structure before filtering or aggregation."""
    try:
        df = _get_dataframe(source); schema_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
        return f"Schema for {source} data:\n{json.dumps(schema_dict, indent=2)}"
    except Exception as e: return f"Error getting schema for {source}: {str(e)}"

# +++++ ADD THIS NEW TOOL DEFINITION +++++
@tool("set_active_dataframe", args_schema=SetActiveDataFrameArgs)
def set_active_dataframe(source: str) -> str:
    """
    Loads the specified original data source ('P&L' or 'JE') as the active DataFrame.
    This active DataFrame is then used by subsequent tools like 'aggregate_dataframe', 'sort_dataframe', 'plot_dataframe', etc.
    Use this tool at the beginning of a sequence of operations if you intend to work on the entirety of an original data source without initial filtering,
    or to reset the active DataFrame to an original source.
    """
    try:
        df = _get_dataframe(source) # _get_dataframe should return a copy
        st.session_state[INTERMEDIATE_DF_KEY] = df
        return f"Successfully set active DataFrame to original '{source}' data ({len(df)} rows). Ready for further operations."
    except Exception as e:
        return f"Error setting active DataFrame from source '{source}': {str(e)}"
# ++++++++++++++++++++++++++++++++++++++++

@tool("find_exact_account_name", args_schema=FindAccountArgs)
def find_exact_account_name(account_query: str, source: str, min_score_threshold: int = 85) -> str:
    """
    Finds the best matching actual account name in the specified data source ('P&L' or 'JE')
    based on the user's query, using fuzzy matching.
    Returns the best matching account name string if score is above threshold, otherwise indicates no match found.
    Use this BEFORE filtering by account name if the user's query might contain variations or typos.
    """
    try:
        df = _get_dataframe(source); col_config = st.session_state.get('column_config', {})
        account_col = col_config.get("PL_MAP_DISPLAY" if source.upper() == 'P&L' else "JE_ACCOUNT_NAME", "Account Name")
        if account_col not in df.columns and "Account Name" in df.columns: account_col = "Account Name"
        elif account_col not in df.columns: return f"Error: Suitable account name column not found in {source}."
        unique_accounts = df[account_col].dropna().unique().astype(str)
        if not unique_accounts.size: return f"No unique account names in '{account_col}' for {source}."
        best_match = fuzzy_process.extractOne(account_query, unique_accounts)
        if best_match and best_match[1] >= min_score_threshold: return f"Found exact account name: '{best_match[0]}' (Score: {best_match[1]}%)"
        return f"No account name in {source} matching '{account_query}' with score >= {min_score_threshold}%. Best: '{best_match[0]}' ({best_match[1]}%)."
    except Exception as e: return f"Error finding account name for '{account_query}' in {source}: {str(e)}"

@tool("filter_dataframe", args_schema=FilterArgs)
def filter_dataframe(source: str, conditions: str) -> str:
    """
    Filters the original 'P&L' or 'JE' DataFrame based on pandas query string conditions.
    The filtered result is stored internally for subsequent operations, becoming the active DataFrame.
    Use exact column names (use backticks `` for names with spaces). For account names, first use 'find_exact_account_name' tool. For fuzzy text in other columns, use 'fuzzy_filter_by_text_in_column'.
    Example conditions: '`Amount` > 0 and `Account Name` == "Exact Account Name from Find Tool"'
    """
    try:
        df = _get_dataframe(source); filtered_df = df.query(conditions).copy(); count = len(filtered_df)
        st.session_state[INTERMEDIATE_DF_KEY] = filtered_df
        return f"Successfully filtered {source} data on '{conditions}'. Result: {count} rows. Ready for further operations."
    except Exception as e:
        available_cols = _get_dataframe(source).columns.tolist() if 'source' in locals() else 'N/A'
        return f"Error filtering data with condition '{conditions}': {str(e)}. Check syntax, quotes, and column names (available: {available_cols}). Use backticks `` for names with spaces."

@tool("fuzzy_filter_by_text_in_column", args_schema=FuzzyFilterArgs)
def fuzzy_filter_by_text_in_column(source: str, text_column_to_search: str, query_text: str, score_threshold: int = 80) -> dict: # MODIFIED return type to dict
    """
    Filters the DataFrame (source: 'P&L' or 'JE') where the 'text_column_to_search'
    fuzzily matches the 'query_text' above a given 'score_threshold' (default 80).
    Useful for finding inexact matches in text fields like 'Memo' or 'Description'.
    The filtered result is stored internally for subsequent operations, becoming the active DataFrame.
    Returns a message and the filtered data for auditing.
    """
    try:
        df = _get_dataframe(source)
        if text_column_to_search not in df.columns:
            return {"error": f"Text column '{text_column_to_search}' not found in {source}."} # Return error as dict

        df[text_column_to_search] = df[text_column_to_search].astype(str).fillna('') # Ensure string type for fuzzy matching
        
        # Apply fuzzy matching
        scores = df[text_column_to_search].apply(lambda x: fuzz.partial_ratio(query_text.lower(), str(x).lower()))
        filtered_df = df[scores >= score_threshold].copy()
        count = len(filtered_df)
        
        st.session_state[INTERMEDIATE_DF_KEY] = filtered_df # Set the active DataFrame
        
        message = ""
        if count == 0:
            message = f"No items in '{text_column_to_search}' fuzzily matched '{query_text}' (score >= {score_threshold}%). Active DataFrame is empty."
        else:
            message = f"Successfully filtered by fuzzy text on '{text_column_to_search}' for '{query_text}'. Result: {count} rows. This is now the active DataFrame."
        
        return {"message": message, "audit_dataframe": filtered_df.to_dict(orient='records')}
    except ImportError:
        # This error should ideally be caught at app startup if thefuzz is missing
        return {"error": "CRITICAL_ERROR: 'thefuzz' library is not installed. This tool cannot function."}
    except Exception as e:
        return {"error": f"Error during fuzzy text filter: {str(e)}"}

@tool("aggregate_dataframe", args_schema=AggregateArgs)
def aggregate_dataframe(group_by_columns: List[str], agg_specs: Dict[str,str]) -> str:
    """
    Groups the currently active DataFrame (resulting from a previous step like 'filter_dataframe' or 'set_active_dataframe')
    by specified columns and calculates aggregate values using the mandatory 'agg_specs' dictionary.
    The result of this aggregation becomes the new active DataFrame.
    """
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None: return "Error: No active DataFrame from previous step. Use 'filter_dataframe' or 'set_active_dataframe' first."
    try:
        df = st.session_state[INTERMEDIATE_DF_KEY]
        if df.empty:
            st.session_state[INTERMEDIATE_DF_KEY] = df # Keep empty df as active
            return "Cannot aggregate empty DataFrame. Active DataFrame is empty."
        missing_group_cols = [col for col in group_by_columns if col not in df.columns]
        if missing_group_cols: return f"Error: Group by column(s) not found: {', '.join(missing_group_cols)}. Available: {df.columns.tolist()}"
        missing_agg_cols = [col for col in agg_specs.keys() if col not in df.columns]
        if missing_agg_cols: return f"Error: Column(s) for aggregation not found: {', '.join(missing_agg_cols)}. Available: {df.columns.tolist()}"

        aggregated_df = df.groupby(group_by_columns, as_index=False).agg(agg_specs)
        st.session_state[INTERMEDIATE_DF_KEY] = aggregated_df; count = len(aggregated_df)
        return f"Successfully aggregated data. Result: {count} rows. Ready for further operations."
    except Exception as e:
        cols = st.session_state[INTERMEDIATE_DF_KEY].columns.tolist() if INTERMEDIATE_DF_KEY in st.session_state and st.session_state[INTERMEDIATE_DF_KEY] is not None else "N/A (No active DF)"
        return f"Error aggregating data: {str(e)}. Check agg_specs functions and column types. Available columns for aggregation: {cols}"

@tool("sort_dataframe", args_schema=SortArgs)
def sort_dataframe(sort_by_columns: List[str], ascending: bool) -> str:
    """Sorts the currently active DataFrame. The sorted DataFrame becomes the new active DataFrame."""
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None: return "Error: No active DataFrame from previous step."
    try:
        df = st.session_state[INTERMEDIATE_DF_KEY]
        if df.empty: return "Cannot sort empty DataFrame. Active DataFrame is empty."
        sorted_df = df.sort_values(by=sort_by_columns, ascending=ascending)
        st.session_state[INTERMEDIATE_DF_KEY] = sorted_df
        return f"Successfully sorted data by {sort_by_columns} ({'asc' if ascending else 'desc'})."
    except KeyError as e: return f"Error sorting: Column not found - {str(e)}. Available: {st.session_state[INTERMEDIATE_DF_KEY].columns.tolist()}"
    except Exception as e: return f"Error sorting data: {str(e)}"

@tool("get_top_n", args_schema=TopNArgs)
def get_top_n(n: int) -> str:
    """Selects the top 'n' rows from the currently active DataFrame. The result becomes the new active DataFrame."""
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None: return "Error: No active DataFrame from previous step."
    try:
        df = st.session_state[INTERMEDIATE_DF_KEY]
        if n <= 0: return "Error: N must be positive."
        top_n_df = df.head(n); st.session_state[INTERMEDIATE_DF_KEY] = top_n_df
        return f"Successfully selected top {n} rows. Result: {len(top_n_df)} rows."
    except Exception as e: return f"Error selecting top N: {str(e)}"

@tool("plot_dataframe", args_schema=PlotArgs)
def plot_dataframe(plot_type: str, x_col_or_names: str, y_col_or_values: List[str], title: str) -> dict:
    """
    Generates a plot ('bar', 'line', or 'pie') from the FINAL state of the active DataFrame from previous steps.
    For bar/line: x_col_or_names is X-axis, y_col_or_values is Y-axis (can be multiple for line).
    For pie: x_col_or_names is for labels/names, y_col_or_values[0] is for values. Pie charts work best with pre-aggregated data and a limited number of categories.
    Should be the LAST step if plotting. Returns plot spec and audit data.
    """
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None: return {"error": "No final active DataFrame to plot."}
    try:
        df_to_plot = st.session_state[INTERMEDIATE_DF_KEY]
        if not isinstance(df_to_plot, pd.DataFrame) or df_to_plot.empty: return {"error": "Active DataFrame to plot is empty/invalid."}
        audit_data = df_to_plot.to_dict(orient='records'); fig = None; plot_type = plot_type.lower()

        if x_col_or_names not in df_to_plot.columns: return {"error": f"X-axis/Names column '{x_col_or_names}' not found. Available: {df_to_plot.columns.tolist()}"}
        for y_c in y_col_or_values:
            if y_c not in df_to_plot.columns: return {"error": f"Y-axis/Values column '{y_c}' not found. Available: {df_to_plot.columns.tolist()}"}

        start_time_plot = time.time(); plot_generated = False

        if plot_type == 'bar':
            y_arg = y_col_or_values[0] if len(y_col_or_values) == 1 else y_col_or_values
            fig = px.bar(df_to_plot, x=x_col_or_names, y=y_arg, title=title, template="plotly_white")
            plot_generated = True
        elif plot_type == 'line':
            fig = px.line(df_to_plot, x=x_col_or_names, y=y_col_or_values, title=title, markers=True, template="plotly_white")
            plot_generated = True
        elif plot_type == 'pie':
            if not y_col_or_values: return {"error": "Pie chart needs a values column in y_col_or_values."}
            val_col = y_col_or_values[0]
            if not pd.api.types.is_numeric_dtype(df_to_plot[val_col]): return {"error": f"Pie chart values column '{val_col}' must be numeric."}
            
            plot_df_pie = df_to_plot[[x_col_or_names, val_col]].copy()
            plot_df_pie = plot_df_pie[plot_df_pie[val_col] > 0] 
            if plot_df_pie.empty: return {"error": f"No positive data in '{val_col}' for pie chart."}

            PIE_SLICE_THRESHOLD = 10 
            if len(plot_df_pie[x_col_or_names].unique()) > PIE_SLICE_THRESHOLD:
                plot_df_pie = plot_df_pie.sort_values(by=val_col, ascending=False)
                top_df = plot_df_pie.head(PIE_SLICE_THRESHOLD - 1)
                other_sum = plot_df_pie.iloc[PIE_SLICE_THRESHOLD - 1:][val_col].sum()
                if other_sum > 0: 
                    other_row = pd.DataFrame([{x_col_or_names: "Other", val_col: other_sum}])
                    plot_df_pie = pd.concat([top_df, other_row], ignore_index=True)
                else: plot_df_pie = top_df
            
            fig = px.pie(plot_df_pie, names=x_col_or_names, values=val_col, title=title, template="plotly_white")
            plot_generated = True
        else: return {"error": f"Unsupported plot type: '{plot_type}'. Use 'bar', 'line', or 'pie'."}
         
        if plot_generated and fig:
             end_time_plot = time.time(); st.sidebar.info(f"DEBUG: px.{plot_type} took {end_time_plot - start_time_plot:.4f}s")
             start_time_json = time.time(); plot_json = fig.to_json(); end_time_json = time.time()
             st.sidebar.info(f"DEBUG: fig.to_json() took {end_time_json - start_time_json:.4f}s")
             if fig and plot_type != 'pie': fig.update_layout(yaxis_tickformat=",.0f")
             return {"plot_json": plot_json, "audit_dataframe": audit_data, "message": f"Generated '{title}' {plot_type} plot."}
        elif fig is None: return {"error": "Plot figure was not generated."}
    except Exception as e: return {"error": f"Failed to generate plot: {str(e)}"}


@tool("summarize_dataframe", args_schema=SummarizeArgs)
def summarize_dataframe() -> dict:
    """
    Provides text summary of the FINAL state of the active DataFrame from previous steps.
    Should be the LAST step if summarizing. Returns summary and audit data.
    """
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None: return {"error": "No final active DataFrame to summarize."}
    try:
        df = st.session_state[INTERMEDIATE_DF_KEY]
        if not isinstance(df, pd.DataFrame): return {"error": "Active data to summarize not a valid DataFrame."}
        audit_data = df.to_dict(orient='records'); summary_lines = [f"Summary of final {len(df)} rows:"]
        if not df.empty:
            summary_lines.append(f"Columns: {df.columns.tolist()}")
            num_cols = df.select_dtypes(include=np.number).columns
            if not num_cols.empty: summary_lines.append(f"\nNumerical Stats:\n{df[num_cols].describe().to_string()}")
            non_num_cols = df.select_dtypes(exclude=np.number).columns
            if not non_num_cols.empty:
                summary_lines.append("\nNon-Numerical Info:")
                for col in non_num_cols:
                    unique = df[col].nunique(); top = 'N/A'
                    if not df[col].dropna().mode().empty: top = str(df[col].dropna().mode()[0])
                    summary_lines.append(f"- '{col}': {unique} unique. Top: {top}")
        else: summary_lines.append("Final active DataFrame is empty.")
        summary_text = "\n".join(summary_lines)
        return {"summary_text": summary_text, "audit_dataframe": audit_data, "message": "Generated text summary."}
    except Exception as e: return {"error": f"Failed to summarize: {str(e)}"}

@tool("find_new_items_in_period", args_schema=FindNewItemsArgs)
def find_new_items_in_period(source: str, item_column: str, date_column: str, year_to_check: int) -> dict:
    """
    Identifies items (e.g., Customers, Products) that appeared for the first time in a specified year
    from the P&L or JE data. Compares against all prior years in the dataset.
    The result (DataFrame of new items) is stored as the active DataFrame for potential subsequent summarization/plotting.
    """
    try:
        df = _get_dataframe(source)
        if date_column not in df.columns: return {"error": f"Date column '{date_column}' not in {source}."}
        if item_column not in df.columns: return {"error": f"Item column '{item_column}' not in {source}."}
        try: df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e: return {"error": f"Could not convert '{date_column}' to datetime: {str(e)}"}
         
        current_items = df[df[date_column].dt.year == year_to_check][item_column].dropna().unique()
        prior_items = df[df[date_column].dt.year < year_to_check][item_column].dropna().unique()
        new_items = sorted(list(set(current_items) - set(prior_items)))
         
        res_df = pd.DataFrame({f"New_{item_column.replace(' ', '_')}_in_{year_to_check}": new_items})
        st.session_state[INTERMEDIATE_DF_KEY] = res_df # Sets the active DataFrame
        msg = f"Found {len(new_items)} new '{item_column}' items in {year_to_check}."
        if new_items: msg += f" First few: {', '.join(new_items[:10])}{'...' if len(new_items) > 10 else ''}."
        else: msg = f"No new '{item_column}' items found in {year_to_check}."
        return {"summary_text": msg, "audit_dataframe": res_df.to_dict(orient='records'), "message": msg + " This result is now the active DataFrame."}
    except Exception as e: return {"error": f"Error finding new items: {str(e)}"}

@tool("analyze_period_variance_drivers", args_schema=AnalyzeVarianceArgs)
def analyze_period_variance_drivers(source: str, date_column: str, metric_column: str, category_column: str,
                                   period1_start_date: str, period1_end_date: str,
                                   period2_start_date: str, period2_end_date: str,
                                   account_name_filter: Optional[str] = None, top_n_drivers: int = 5) -> dict:
    """
    Analyzes variance of a metric between two periods, grouped by a category, to find key drivers.
    Dates must be 'YYYY-MM-DD'. Uses an original data source specified by 'source'.
    The result (DataFrame of drivers) is stored as the active DataFrame for potential summarization or plotting.
    """
    try:
        df_full = _get_dataframe(source)
        req_cols = [date_column, metric_column, category_column]; acc_col = 'Account Name'
        if account_name_filter and acc_col not in df_full.columns: return {"error": f"Account filter provided, but '{acc_col}' not in {source}."}
        for col in req_cols:
            if col not in df_full.columns: return {"error": f"Required column '{col}' not in {source}."}
        try:
            df_full[date_column] = pd.to_datetime(df_full[date_column])
            df_full[metric_column] = pd.to_numeric(df_full[metric_column], errors='coerce').fillna(0)
        except Exception as e: return {"error": f"Error converting data types: {str(e)}"}

        df = df_full
        if account_name_filter:
            df = df[df[acc_col] == account_name_filter].copy()
            if df.empty: return {"error": f"No data for account '{account_name_filter}'."}
         
        p1s,p1e,p2s,p2e = pd.to_datetime(period1_start_date), pd.to_datetime(period1_end_date), pd.to_datetime(period2_start_date), pd.to_datetime(period2_end_date)
        df_p1 = df[(df[date_column] >= p1s) & (df[date_column] <= p1e)]
        agg_p1 = df_p1.groupby(category_column, as_index=False)[metric_column].sum().rename(columns={metric_column: 'P1_Value'})
        df_p2 = df[(df[date_column] >= p2s) & (df[date_column] <= p2e)]
        agg_p2 = df_p2.groupby(category_column, as_index=False)[metric_column].sum().rename(columns={metric_column: 'P2_Value'})

        if agg_p1.empty and agg_p2.empty:
            # Still set an empty DataFrame as active if no data, so downstream summarize can report it.
            st.session_state[INTERMEDIATE_DF_KEY] = pd.DataFrame(columns=[category_column, 'P1_Value', 'P2_Value', 'Variance', 'Abs_Variance'])
            return {"summary_text": f"No data for either period for '{metric_column}' by '{category_column}'.", "audit_dataframe": []}
         
        merged = pd.merge(agg_p1, agg_p2, on=category_column, how='outer').fillna(0)
        merged['Variance'] = merged['P1_Value'] - merged['P2_Value'] # Note: Original code had P1-P2. Consider if P2-P1 is more standard for "change". Keeping P1-P2.
        merged['Abs_Variance'] = merged['Variance'].abs()
        drivers = merged[merged['Abs_Variance'] > 1e-6].sort_values(by='Abs_Variance', ascending=False).head(top_n_drivers)
        st.session_state[INTERMEDIATE_DF_KEY] = drivers # Sets the active DataFrame

        p1_str, p2_str = f"{period1_start_date}-{period1_end_date}", f"{period2_start_date}-{period2_end_date}"
        title = f"metric '{metric_column}' by '{category_column}'"
        if account_name_filter: title = f"account '{account_name_filter}', {title}"
        summary = [f"Top {len(drivers)} drivers of variance for {title} (P1 [{p1_str}] vs P2 [{p2_str}]):"]
        if drivers.empty: summary.append("No significant variance or common categories.")
        else:
            for _, r in drivers.iterrows(): summary.append(f"- {r[category_column]}: Var={r['Variance']:,.2f} (P1:{r['P1_Value']:,.2f}, P2:{r['P2_Value']:,.2f})")
        summary_text = "\n".join(summary)
        return {"summary_text": summary_text, "audit_dataframe": drivers.to_dict(orient='records'), "message": summary_text + " This result is now the active DataFrame."}
    except Exception as e: return {"error": f"Error analyzing variance: {str(e)}"}

# Ensure the new tool is added to this list
all_tools = [
    get_data_schema,
    set_active_dataframe, # <--- NEW TOOL ADDED HERE
    find_exact_account_name,
    filter_dataframe,
    fuzzy_filter_by_text_in_column,
    aggregate_dataframe,
    sort_dataframe,
    get_top_n,
    plot_dataframe,
    summarize_dataframe,
    find_new_items_in_period,
    analyze_period_variance_drivers,
]
# **** END OF FULL SCRIPT ****