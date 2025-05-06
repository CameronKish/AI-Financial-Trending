# **** START OF FULL SCRIPT ****
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field # Explicit v1 import

# --- NEW: Import fuzzy matching ---
from thefuzz import process as fuzzy_process

# --- Helper Function to Safely Get Data ---
def _get_dataframe(source: str) -> pd.DataFrame:
    """Safely retrieves the requested DataFrame from session state."""
    source = source.strip().upper()
    if source == "P&L":
        if 'pl_flat_df' in st.session_state:
            return st.session_state.pl_flat_df
        else:
            raise ValueError("P&L DataFrame ('pl_flat_df') not found in session state.")
    elif source == "JE":
        if 'je_detail_df' in st.session_state:
            return st.session_state.je_detail_df
        else:
            raise ValueError("JE DataFrame ('je_detail_df') not found in session state.")
    else:
        raise ValueError(f"Unknown data source specified: {source}. Use 'P&L' or 'JE'.")

# --- Pydantic Models for Tool Arguments ---

class GetSchemaArgs(BaseModel):
    source: str = Field(..., description="The data source to get the schema for. Must be 'P&L' or 'JE'.")

# --- NEW: Args for account finding ---
class FindAccountArgs(BaseModel):
    account_query: str = Field(..., description="The account name or description provided by the user.")
    source: str = Field(..., description="The data source ('P&L' or 'JE') to find the account name in.")
    min_score_threshold: int = Field(default=85, description="Minimum matching score (0-100) to consider an account a match.")

class FilterArgs(BaseModel):
    source: str = Field(..., description="The data source ('P&L' or 'JE') to filter.")
    # --- UPDATED Description ---
    conditions: str = Field(..., description="Filtering conditions using pandas query string syntax (e.g., 'Amount > 1000 and Customer == \"ACME Corp\"'). Use exact column names (use backticks `` for names with spaces). For account names, it's recommended to use the 'find_exact_account_name' tool first to get the precise name.")

class AggregateArgs(BaseModel):
    group_by_columns: list[str] = Field(..., description="List of column names to group by.")
    agg_specs: dict = Field(..., description="**Mandatory**. Dictionary specifying the aggregation calculation. Keys are column names, values are functions ('sum', 'count', 'mean', 'nunique', etc.).") # Removed example

class SortArgs(BaseModel):
    sort_by_columns: list[str] = Field(..., description="List of column names to sort by.")
    ascending: bool = Field(default=True, description="Sort order.")

class TopNArgs(BaseModel):
    n: int = Field(..., description="The number of top rows to select.")

class PlotArgs(BaseModel):
    plot_type: str = Field(..., description="Type of plot ('bar' or 'line').")
    x_col: str = Field(..., description="Column name for the X-axis.")
    y_col: list[str] = Field(..., description="List of column names for the Y-axis.")
    title: str = Field(..., description="Title for the plot.")

class SummarizeArgs(BaseModel):
    pass

# --- Tool Definitions ---

INTERMEDIATE_DF_KEY = "agent_intermediate_df"

@tool("get_data_schema", args_schema=GetSchemaArgs)
def get_data_schema(source: str) -> str:
    """Gets column names and data types for the specified data source ('P&L' or 'JE'). Useful for understanding data structure before filtering or aggregation."""
    try:
        df = _get_dataframe(source)
        schema_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
        return f"Schema for {source} data:\n{json.dumps(schema_dict, indent=2)}"
    except Exception as e:
        return f"Error getting schema for {source}: {str(e)}"

# --- NEW TOOL ---
@tool("find_exact_account_name", args_schema=FindAccountArgs)
def find_exact_account_name(account_query: str, source: str, min_score_threshold: int = 85) -> str:
    """
    Finds the best matching actual account name in the specified data source ('P&L' or 'JE')
    based on the user's query, using fuzzy matching.
    Returns the best matching account name string if score is above threshold, otherwise indicates no match found.
    Use this BEFORE filtering by account name if the user's query might contain variations or typos.
    """
    try:
        df = _get_dataframe(source)
        # Determine the relevant account name column
        # Use PL_MAP_DISPLAY for P&L and JE_ACCOUNT_NAME for JE if available from config, otherwise default
        col_config = st.session_state.get('column_config', {})
        if source == 'P&L':
            account_col = col_config.get("PL_MAP_DISPLAY", "Account Name") # Fallback if config missing
        elif source == 'JE':
             account_col = col_config.get("JE_ACCOUNT_NAME", "Account Name") # Fallback
        else: # Should have been caught by _get_dataframe, but double-check
             return f"Error: Invalid source '{source}' specified."

        if account_col not in df.columns:
            return f"Error: Cannot find account column '{account_col}' in {source} data."

        unique_accounts = df[account_col].unique().astype(str)
        if len(unique_accounts) == 0:
             return f"No unique account names found in column '{account_col}' for {source} data."

        # Use fuzzy matching to find the best match
        # process.extractOne returns tuple: (best_match, score)
        best_match = fuzzy_process.extractOne(account_query, unique_accounts)

        if best_match and best_match[1] >= min_score_threshold:
            exact_name = best_match[0]
            return f"Found exact account name: '{exact_name}' (Matching score: {best_match[1]}%)"
        else:
            return f"No account name found in {source} data matching '{account_query}' with score >= {min_score_threshold}%. Best attempt was '{best_match[0]}' ({best_match[1]}%). Consider listing accounts or refining the query."

    except Exception as e:
        return f"Error finding account name for '{account_query}' in {source}: {str(e)}"


@tool("filter_dataframe", args_schema=FilterArgs)
def filter_dataframe(source: str, conditions: str) -> str:
    """
    Filters the original 'P&L' or 'JE' DataFrame based on pandas query string conditions.
    The filtered result is stored internally for subsequent operations.
    Use exact column names (use backticks `` for names with spaces). For account names, first use 'find_exact_account_name' tool.
    Example conditions: '`Amount` > 0 and `Account Name` == "Exact Account Name from Find Tool"'
    """
    try:
        df = _get_dataframe(source)
        filtered_df = df.query(conditions).copy()
        count = len(filtered_df)
        st.session_state[INTERMEDIATE_DF_KEY] = filtered_df
        return f"Successfully filtered the {source} data based on conditions '{conditions}'. Resulting DataFrame has {count} rows."
    except KeyError as e:
         available_cols = []
         try: available_cols = _get_dataframe(source).columns.tolist()
         except Exception: pass
         return f"Error filtering data: Column not found - {str(e)}. Check column names (use backticks `` for names with spaces). Available columns in {source}: {available_cols}"
    except Exception as e:
        # Check for pandas query syntax errors specifically
        if "SyntaxError" in str(e) or "UndefinedVariableError" in str(e):
             return f"Error filtering data due to invalid query syntax or column name in condition '{conditions}': {str(e)}. Ensure proper quoting and use backticks `` for names with spaces."
        return f"Error filtering data with conditions '{conditions}': {str(e)}. Please check query syntax and column names."

@tool("aggregate_dataframe", args_schema=AggregateArgs)
def aggregate_dataframe(group_by_columns: list[str], agg_specs: dict) -> str:
    """
    Groups the DataFrame from the previous step by specified columns and calculates aggregate values using the mandatory 'agg_specs' dictionary (keys are columns, values are functions like 'sum', 'count').
    """
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None:
        return "Error: No DataFrame found from a previous step. Filter/process data first."
    if not agg_specs:
        return "Error: Mandatory 'agg_specs' dictionary is missing."
    try:
        df = st.session_state[INTERMEDIATE_DF_KEY]
        aggregated_df = df.groupby(group_by_columns).agg(agg_specs).reset_index()
        st.session_state[INTERMEDIATE_DF_KEY] = aggregated_df
        count = len(aggregated_df)
        return f"Successfully aggregated data by {group_by_columns}. Result has {count} rows."
    except KeyError as e:
         return f"Error aggregating data: Column not found in intermediate data - {str(e)}. Columns available: {df.columns.tolist()}"
    except Exception as e:
        if "is not a valid function for" in str(e):
             return f"Error aggregating data: Invalid aggregation function in agg_specs. Details: {str(e)}"
        return f"Error aggregating data: {str(e)}"

@tool("sort_dataframe", args_schema=SortArgs)
def sort_dataframe(sort_by_columns: list[str], ascending: bool) -> str:
    """Sorts the DataFrame from the previous step."""
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None:
        return "Error: No DataFrame found from a previous step."
    try:
        df = st.session_state[INTERMEDIATE_DF_KEY]
        sorted_df = df.sort_values(by=sort_by_columns, ascending=ascending)
        st.session_state[INTERMEDIATE_DF_KEY] = sorted_df
        return f"Successfully sorted data by {sort_by_columns} ({'ascending' if ascending else 'descending'})."
    except KeyError as e:
         return f"Error sorting data: Column not found - {str(e)}. Columns available: {df.columns.tolist()}"
    except Exception as e:
        return f"Error sorting data: {str(e)}"

@tool("get_top_n", args_schema=TopNArgs)
def get_top_n(n: int) -> str:
    """Selects the top 'n' rows from the DataFrame from the previous step."""
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None:
        return "Error: No DataFrame found from a previous step."
    try:
        df = st.session_state[INTERMEDIATE_DF_KEY]
        top_n_df = df.head(n)
        st.session_state[INTERMEDIATE_DF_KEY] = top_n_df
        return f"Successfully selected top {n} rows."
    except Exception as e:
        return f"Error selecting top N rows: {str(e)}"

# --- Final Output Tools ---

@tool("plot_dataframe", args_schema=PlotArgs)
def plot_dataframe(plot_type: str, x_col: str, y_col: list[str], title: str) -> dict:
    """
    Generates a plot ('bar' or 'line') from the FINAL DataFrame from previous steps. Should be the LAST step if plotting. Returns plot spec and audit data.
    """
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None:
        return {"error": "No final DataFrame found to plot."}
    try:
        df_to_plot = st.session_state[INTERMEDIATE_DF_KEY]
        audit_data = df_to_plot.to_dict(orient='records')
        fig = None; plot_type = plot_type.lower(); y_plot_arg = y_col[0] if len(y_col) == 1 else y_col
        if plot_type == 'bar': fig = px.bar(df_to_plot, x=x_col, y=y_plot_arg, title=title, template="plotly_white")
        elif plot_type == 'line':
             if isinstance(y_plot_arg, str): y_plot_arg = [y_plot_arg]
             fig = px.line(df_to_plot, x=x_col, y=y_plot_arg, title=title, markers=True, template="plotly_white")
        else: return {"error": f"Unsupported plot type: {plot_type}. Use 'bar' or 'line'."}
        fig.update_layout(yaxis_tickformat=",.0f")
        return {"plot_json": fig.to_json(), "audit_dataframe": audit_data, "message": f"Generated '{title}' {plot_type} plot."}
    except KeyError as e: return {"error": f"Plotting Error: Column not found - {str(e)}. Columns available: {df_to_plot.columns.tolist()}"}
    except Exception as e: return {"error": f"Failed to generate plot: {str(e)}"}

@tool("summarize_dataframe", args_schema=SummarizeArgs)
def summarize_dataframe() -> dict:
    """
    Provides text summary of the FINAL DataFrame from previous steps. Should be the LAST step if summarizing. Returns summary and audit data.
    """
    if INTERMEDIATE_DF_KEY not in st.session_state or st.session_state[INTERMEDIATE_DF_KEY] is None:
        return {"error": "No final DataFrame found to summarize."}
    try:
        df_to_summarize = st.session_state[INTERMEDIATE_DF_KEY]; audit_data = df_to_summarize.to_dict(orient='records')
        summary_lines = []; summary_lines.append(f"Summary of final {len(df_to_summarize)} rows:")
        if len(df_to_summarize) > 0:
            summary_lines.append(f"Columns: {df_to_summarize.columns.tolist()}")
            numeric_cols = df_to_summarize.select_dtypes(include=np.number).columns
            if not numeric_cols.empty: summary_lines.append("\nNumerical Statistics:\n" + df_to_summarize[numeric_cols].describe().to_string())
            non_numeric_cols = df_to_summarize.select_dtypes(exclude=np.number).columns
            if not non_numeric_cols.empty:
                summary_lines.append("\nNon-Numerical Info:")
                for col in non_numeric_cols:
                    unique_count = df_to_summarize[col].nunique(); top_value_str = 'N/A'
                    if not df_to_summarize[col].mode().empty: top_value_str = str(df_to_summarize[col].mode()[0])
                    summary_lines.append(f"- '{col}': {unique_count} unique values. Top value: {top_value_str}")
        else: summary_lines.append("Final DataFrame is empty.")
        summary_text = "\n".join(summary_lines)
        return {"summary_text": summary_text, "audit_dataframe": audit_data, "message": "Generated text summary."}
    except Exception as e: return {"error": f"Failed to generate summary: {str(e)}"}

# --- UPDATED: List of tools including the new one ---
all_tools = [
    get_data_schema,
    find_exact_account_name, # Added new tool
    filter_dataframe,
    aggregate_dataframe,
    sort_dataframe,
    get_top_n,
    plot_dataframe,
    summarize_dataframe,
]
# **** END OF FULL SCRIPT ****