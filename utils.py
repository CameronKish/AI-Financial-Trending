# utils.py
import pandas as pd
import numpy as np
import requests
import json
import os
import streamlit as st
from openai import OpenAI, AzureOpenAI
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import re
import logging # Added for format_je_for_llm logging

# --- Define EY Parthenon Inspired Colors ---
EY_YELLOW = "#FFE600"; EY_DARK_BLUE_GREY = "#2E2E38"; EY_TEXT_ON_YELLOW = EY_DARK_BLUE_GREY
OLLAMA_BASE_URL = "http://localhost:11434"
SETTINGS_FILE = "llm_analyzer_settings.json"
DEFAULT_SETTINGS = {
    'llm_provider': 'Ollama', 'chosen_ollama_model': None, 'openai_api_key': '',
    'openai_model_name': "gpt-4o-mini", 'azure_endpoint': '', 'azure_api_key': '',
    'azure_deployment_name': '', 'azure_api_version': "2024-02-01",
    'llm_context_limit': 32000,
}

RANGE_INPUT_CATEGORIES = ["Assets", "Liabilities", "Equity", "Revenue", "COGS", "Operating Expenses", "Other P&L Items"]
PNL_CATEGORIES_LIST = ["Revenue", "COGS", "Operating Expenses", "Other P&L Items"]
BS_CATEGORIES_LIST = ["Assets", "Liabilities", "Equity"]


def get_local_ollama_models(ollama_base_url=OLLAMA_BASE_URL):
    """
    Fetches the list of locally available Ollama models using the /api/tags endpoint.
    Returns a list of model names (e.g., ['mistral:latest', 'llama3:latest']).
    """
    try:
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=5) # 5-second timeout
        response.raise_for_status() # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        models_data = response.json()
        
        # The structure is typically {'models': [{'name': 'modelname:tag', ...}, ...]}
        model_names = [model['name'] for model in models_data.get('models', [])]
        return sorted(model_names)
    except requests.exceptions.ConnectionError:
        # This error is common if Ollama isn't running or not accessible at the URL
        # Log it or print for server-side debugging, avoid st.error in utils if possible
        print(f"Warning (get_local_ollama_models): Could not connect to Ollama at {ollama_base_url}. Is Ollama running and accessible?")
        return [] # Return empty list on connection error
    except requests.exceptions.Timeout:
        print(f"Warning (get_local_ollama_models): Timeout when trying to connect to Ollama at {ollama_base_url}.")
        return []
    except requests.exceptions.RequestException as e:
        # For other request-related errors (like 404, 500 from Ollama server if API changes)
        print(f"Warning (get_local_ollama_models): Request to Ollama failed: {e}")
        return []
    except Exception as e:
        # Catch any other unexpected errors during parsing or processing
        print(f"Warning (get_local_ollama_models): An unexpected error occurred while fetching Ollama models: {e}")
        return []


def parse_gl_ranges(range_str: str) -> list | None:
    parsed = []
    if not isinstance(range_str, str) or not range_str.strip(): return parsed
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if not part: continue
        if '-' in part:
            start_str, end_str = part.split('-', 1)
            try: parsed.append((int(start_str.strip()), int(end_str.strip())))
            except ValueError:
                print(f"Warning (parse_gl_ranges): Invalid range format: '{part}'. Skipping this part.")
                return None
        else:
            try: parsed.append((int(part.strip()), int(part.strip())))
            except ValueError:
                print(f"Warning (parse_gl_ranges): Invalid number format: '{part}'. Skipping this part.")
                return None
    return parsed

def get_prefix_based_category(acc_id_str: str) -> str:
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

def assign_category_using_rules(acc_id_str: str, use_custom_flag: bool,
                                custom_parsed_ranges_dict: dict,
                                prefix_logic_func) -> str:
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

def get_statement_section(assigned_category: str) -> str:
    if assigned_category in PNL_CATEGORIES_LIST: return "P&L"
    elif assigned_category in BS_CATEGORIES_LIST: return "BS"
    else: return "Uncategorized"

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f: loaded_settings = json.load(f)
            settings = DEFAULT_SETTINGS.copy(); settings.update(loaded_settings)
            return settings
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading settings file '{SETTINGS_FILE}': {e}. Using defaults.")
            return DEFAULT_SETTINGS.copy()
    else: return DEFAULT_SETTINGS.copy()

def save_settings(settings_dict: dict):
    try:
        settings_to_save = {k: settings_dict.get(k) for k in DEFAULT_SETTINGS}
        with open(SETTINGS_FILE, 'w') as f: json.dump(settings_to_save, f, indent=4)
    except IOError as e:
        if 'st' in globals() and hasattr(st, 'error'): st.error(f"Error saving settings to '{SETTINGS_FILE}': {e}")
        else: print(f"Error saving settings to '{SETTINGS_FILE}': {e}")
    except Exception as e:
        if 'st' in globals() and hasattr(st, 'error'): st.error(f"An unexpected error occurred while saving settings: {e}")
        else: print(f"An unexpected error occurred while saving settings: {e}")

def ensure_settings_loaded():
    app_settings = load_settings()
    for key in DEFAULT_SETTINGS.keys():
        st.session_state[key] = app_settings.get(key, DEFAULT_SETTINGS.get(key))

def format_amount_safely(value):
    if pd.isna(value): return ""
    if isinstance(value, (int, float, np.number)):
        try: return f"{value:,.0f}"
        except (TypeError, ValueError): return str(value)
    else:
        try: num = pd.to_numeric(value); return f"{num:,.0f}"
        except (TypeError, ValueError): return str(value)

def get_index(options_list, value):
    if not options_list: return 0
    try: return options_list.index(value)
    except ValueError: return 0

def highlight_outliers_pandas(row, diffs_df, thresholds_series, color=EY_YELLOW, text_color=EY_TEXT_ON_YELLOW):
    try:
        if row.name not in diffs_df.index or row.name not in thresholds_series.index: return [''] * len(row)
        row_diff = diffs_df.loc[row.name]; threshold_value = thresholds_series.loc[row.name]
        if pd.isna(threshold_value) or threshold_value < 1e-6: return [''] * len(row)
    except KeyError: return [''] * len(row)
    except Exception as e: print(f"Highlighting error: {e}"); return [''] * len(row)
    styles = [''] * len(row.index)
    for i, period in enumerate(row.index):
        if i == 0 and isinstance(row.index, pd.MultiIndex): continue # Skip first col if it's part of multi-index name
        if i == 0 and not isinstance(row.index, pd.MultiIndex) and len(row.index) > 1 and period == row.name : continue # Skip if first col is index name for single index

        diff_val = row_diff.get(period)
        if pd.notna(diff_val) and abs(diff_val) > threshold_value:
            if i < len(styles): styles[i] = f'background-color: {color}; color: {text_color};'
    return styles

def format_je_for_llm(je_df, amount_col, max_raw_rows=30):
    if je_df is None or je_df.empty:
        return "No Journal Entry details provided."

    je_df_clean = je_df.copy()
    actual_amount_col = amount_col # Store original to avoid modifying it if fallback occurs

    if actual_amount_col not in je_df_clean.columns:
        potential_amount_cols = [col for col in je_df_clean.columns if 'amount' in col.lower()]
        if potential_amount_cols:
            actual_amount_col = potential_amount_cols[0]
            logging.warning(f"Original amount_col '{amount_col}' not found in format_je_for_llm, using fallback '{actual_amount_col}'.")
        else:
            logging.error("format_je_for_llm: JE details missing a recognizable amount column for formatting.")
            return "JE details missing a recognizable amount column."

    if actual_amount_col in je_df_clean.columns:
        je_df_clean[actual_amount_col] = pd.to_numeric(je_df_clean[actual_amount_col], errors='coerce')
    else:
        logging.error(f"format_je_for_llm: Amount column '{actual_amount_col}' still not found after checks.")
        return f"Specified amount column '{actual_amount_col}' not found in JE data."

    # For detailed display part, use rows where the amount is valid
    je_df_for_raw_display = je_df_clean.dropna(subset=[actual_amount_col]) if actual_amount_col in je_df_clean.columns else je_df_clean.copy()
    num_rows_for_raw_display = len(je_df_for_raw_display)

    if num_rows_for_raw_display == 0 and actual_amount_col in je_df_clean.columns: # All amounts were NaN
        if len(je_df_clean) > max_raw_rows: # If original DF was large, proceed to summarize it
            pass # Summarization logic will handle the full je_df_clean
        else:
            return "No Journal Entry details with valid amounts found to display raw."
    elif num_rows_for_raw_display == 0: # Original DF was empty or amount col missing
        return "No Journal Entry details found."

    # --- Dynamic Column Selection and Ordering ---
    all_available_cols = je_df_for_raw_display.columns.tolist()
    cols_to_show = []
    
    # Attempt to find a date column (heuristic)
    date_col_candidate = None
    for col_name in all_available_cols:
        if any(substr in col_name.lower() for substr in ['date', 'effective', 'period']):
            try: # Check if it can be reasonably parsed as datetime for a sample
                if pd.to_datetime(je_df_for_raw_display[col_name], errors='coerce').notna().sum() > len(je_df_for_raw_display) / 2:
                    date_col_candidate = col_name
                    break
            except Exception: continue # Ignore if a column errors out during this check
    
    # Preferred order: date, descriptive fields, other fields, amount
    if date_col_candidate and date_col_candidate not in cols_to_show:
        cols_to_show.append(date_col_candidate)

    # Common descriptive fields (add if they exist and not already added)
    # User might have custom names, these are just common ones to try and front-load
    common_descriptive_try = ['Transaction Id', 'ID', 'Memo', 'Description', 'Line Description', 'Journal Entry Name', 'Customer', 'Vendor', 'account_description']
    for desc_col in common_descriptive_try:
        # Find case-insensitive match if direct match fails
        matched_desc_col = next((col for col in all_available_cols if col.lower() == desc_col.lower()), None)
        if matched_desc_col and matched_desc_col not in cols_to_show:
            cols_to_show.append(matched_desc_col)

    # Add remaining columns, excluding the amount column (to add it last) and already added ones
    for col in all_available_cols:
        if col not in cols_to_show and col != actual_amount_col:
            cols_to_show.append(col)
            
    # Ensure amount column is last if it exists
    if actual_amount_col in all_available_cols and actual_amount_col not in cols_to_show: # Should be caught if not already added
        cols_to_show.append(actual_amount_col)
    elif actual_amount_col in cols_to_show and cols_to_show[-1] != actual_amount_col: # If added but not last
        cols_to_show.remove(actual_amount_col)
        cols_to_show.append(actual_amount_col)
        
    if not cols_to_show and all_available_cols: # Fallback if preferred logic results in empty list
        cols_to_show = all_available_cols[:]
        if actual_amount_col in cols_to_show and cols_to_show[-1] != actual_amount_col: # Try to move amount to end
            cols_to_show.remove(actual_amount_col)
            cols_to_show.append(actual_amount_col)

    # Filter the DataFrame for display with the chosen columns
    df_markdown_ready = je_df_for_raw_display[cols_to_show].copy()

    # Format specific columns
    if actual_amount_col in df_markdown_ready.columns:
        df_markdown_ready[actual_amount_col] = df_markdown_ready[actual_amount_col].apply(format_amount_safely)
    if date_col_candidate and date_col_candidate in df_markdown_ready.columns:
         try: df_markdown_ready[date_col_candidate] = pd.to_datetime(df_markdown_ready[date_col_candidate], errors='coerce').dt.strftime('%Y-%m-%d')
         except Exception: df_markdown_ready[date_col_candidate] = df_markdown_ready[date_col_candidate].astype(str).fillna("")


    if num_rows_for_raw_display <= max_raw_rows:
        return f"Found {num_rows_for_raw_display} entries:\n" + df_markdown_ready.to_markdown(index=False)
    else: # Summarization for many rows
        # Use full je_df_clean for summary stats to reflect all data initially passed
        total_amount_full = je_df_clean[actual_amount_col].sum() if actual_amount_col in je_df_clean.columns else 0
        num_positive_full = len(je_df_clean[je_df_clean[actual_amount_col] > 0]) if actual_amount_col in je_df_clean.columns else 0
        num_negative_full = len(je_df_clean[je_df_clean[actual_amount_col] < 0]) if actual_amount_col in je_df_clean.columns else 0
        
        # For Top N, use df_markdown_ready which is already formatted and has valid amounts
        top_n_df = df_markdown_ready.copy()
        # We need the numeric version of amount for nlargest, before it was formatted as string
        if actual_amount_col in je_df_for_raw_display.columns: # Use unformatted numeric amount for sorting
            top_n_df_sorting = je_df_for_raw_display.copy()
            top_n_df_sorting['Abs_Amount_Numeric'] = top_n_df_sorting[actual_amount_col].abs()
            top_n = min(5, num_rows_for_raw_display)
            top_indices = top_n_df_sorting.nlargest(top_n, 'Abs_Amount_Numeric').index
            top_entries_markdown_df = df_markdown_ready.loc[top_indices] # Get formatted rows
        else: # Fallback if amount column was problematic
            top_n = min(5, len(df_markdown_ready))
            top_entries_markdown_df = df_markdown_ready.head(top_n)

        summary_intro = (f"Summary of {len(je_df_clean)} Journal Entries (due to large volume; {num_rows_for_raw_display} with valid amounts shown in Top N if applicable):\n"
                        f"- Net Amount (all entries): {format_amount_safely(total_amount_full)}\n"
                        f"- Positive Entries (all entries): {num_positive_full}\n"
                        f"- Negative Entries (all entries): {num_negative_full}\n\n"
                        f"Top {len(top_entries_markdown_df)} Entries (from those with valid amounts) by Absolute Amount:\n"
                        f"{top_entries_markdown_df.to_markdown(index=False)}")
        
        summary_parts = [summary_intro]
        top_n_agg = min(3, len(je_df_clean))

        # Try to find Customer or similar column
        customer_col_candidate = next((col for col in je_df_clean.columns if 'customer' in col.lower() and not je_df_clean[col].isnull().all()), None)
        if customer_col_candidate and actual_amount_col in je_df_clean.columns:
            top_customers = je_df_clean.groupby(customer_col_candidate)[actual_amount_col].sum().nlargest(top_n_agg)
            if not top_customers.empty: summary_parts.append(f"\nTop {len(top_customers)} '{customer_col_candidate}' by Net Amount:\n{top_customers.apply(format_amount_safely).to_markdown()}")
        
        memo_col_candidate = next((col for col in je_df_clean.columns if any(substr in col.lower() for substr in ['memo', 'description', 'details']) and not je_df_clean[col].isnull().all()), None)
        if memo_col_candidate and actual_amount_col in je_df_clean.columns:
            top_memos = je_df_clean.groupby(memo_col_candidate)[actual_amount_col].sum().nlargest(top_n_agg)
            if not top_memos.empty: summary_parts.append(f"\nTop {len(top_memos)} '{memo_col_candidate}' entries by Net Amount:\n{top_memos.apply(format_amount_safely).to_markdown()}")
            
        return "\n".join(summary_parts)


def estimate_token_count(text: str) -> int:
    if not isinstance(text, str) or not text: return 0
    return int(len(text) / 4) + 1 # Basic estimation

def _call_ollama_stream_internal(prompt, model_name, base_url=OLLAMA_BASE_URL):
    api_url = f"{base_url}/api/generate"; payload = {"model": model_name, "prompt": prompt, "stream": True}
    headers = {'Content-Type': 'application/json'}; response_stream = None
    try:
        response_stream = requests.post(api_url, headers=headers, json=payload, stream=True, timeout=180); response_stream.raise_for_status()
        for line in response_stream.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if 'response' in chunk: yield chunk['response']
                    if 'error' in chunk: yield f"\n\n**Error from Ollama:** {chunk['error']}\n"; return
                    if chunk.get('done'): return
                except json.JSONDecodeError: continue
                except Exception as e: yield f"\n\n**Error processing stream chunk:** {str(e)}\n"; return
    except requests.exceptions.ConnectionError: yield f"**Error:** Could not connect to Ollama at `{base_url}`."
    except requests.exceptions.Timeout: yield f"**Error:** Request to Ollama timed out."
    except requests.exceptions.RequestException as e: error_detail = str(e); yield f"**Error:** Ollama API request failed: {error_detail}"
    except Exception as e: yield f"**An unexpected error occurred:** {str(e)}"
    finally:
        if response_stream:
            try: response_stream.close()
            except Exception: pass

def _call_openai_stream_internal(prompt, model_name, api_key):
    if not api_key: yield "**Error:** OpenAI API Key not provided."; return
    try:
        client = OpenAI(api_key=api_key); stream = client.chat.completions.create( model=model_name, messages=[{"role": "user", "content": prompt}], stream=True, timeout=180,)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None: yield chunk.choices[0].delta.content
    except Exception as e: yield f"\n\n**Error calling OpenAI API:** {str(e)}\n"

def _call_azure_openai_stream_internal(prompt, deployment_name, endpoint, api_key, api_version):
    if not all([deployment_name, endpoint, api_key, api_version]): yield "**Error:** Azure OpenAI config incomplete."; return
    try:
        client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        stream = client.chat.completions.create( model=deployment_name, messages=[{"role": "user", "content": prompt}], stream=True, timeout=180,)
        for chunk in stream:
             if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None: yield chunk.choices[0].delta.content
    except Exception as e: yield f"\n\n**Error calling Azure OpenAI API:** {str(e)}\n"

def call_llm_stream(provider: str, config: dict, prompt: str):
    if provider == "Ollama":
        model_name = config.get('model_name');
        if not model_name: yield "**Error:** Ollama model name missing."; return
        yield from _call_ollama_stream_internal(prompt=prompt, model_name=model_name)
    elif provider == "OpenAI":
        model_name = config.get('model_name'); api_key = config.get('api_key')
        if not model_name or not api_key: yield "**Error:** OpenAI model or API key missing."; return
        yield from _call_openai_stream_internal(prompt=prompt, model_name=model_name, api_key=api_key)
    elif provider == "Azure OpenAI":
        deployment_name = config.get('deployment_name'); endpoint = config.get('endpoint')
        api_key = config.get('api_key'); api_version = config.get('api_version')
        if not all([deployment_name, endpoint, api_key, api_version]): yield "**Error:** Azure OpenAI config missing."; return
        yield from _call_azure_openai_stream_internal(prompt=prompt, deployment_name=deployment_name, endpoint=endpoint, api_key=api_key, api_version=api_version)
    else: yield f"**Error:** Unknown LLM provider '{provider}'."; return

def get_langchain_llm(temperature=0.0, streaming=True):
    provider = st.session_state.get('llm_provider')
    llm = None
    try:
        if provider == "Ollama":
            model_name = st.session_state.get('chosen_ollama_model')
            if not model_name: st.error("Ollama provider selected, but no model chosen/available."); return None
            llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, temperature=temperature)
        elif provider == "OpenAI":
            api_key = st.session_state.get('openai_api_key')
            model_name = st.session_state.get('openai_model_name', DEFAULT_SETTINGS['openai_model_name'])
            if not api_key: st.error("OpenAI provider selected, but API Key is missing."); return None
            llm = ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature, streaming=streaming)
        elif provider == "Azure OpenAI":
            api_key = st.session_state.get('azure_api_key')
            endpoint = st.session_state.get('azure_endpoint')
            deployment_name = st.session_state.get('azure_deployment_name')
            api_version = st.session_state.get('azure_api_version', DEFAULT_SETTINGS['azure_api_version'])
            if not all([api_key, endpoint, deployment_name, api_version]): st.error("Azure OpenAI provider selected, but configuration is incomplete."); return None
            llm = AzureChatOpenAI(azure_endpoint=endpoint, openai_api_key=api_key, azure_deployment=deployment_name, openai_api_version=api_version, temperature=temperature, streaming=streaming)
        else: st.error(f"Unsupported LLM provider selected: {provider}"); return None
        return llm
    except Exception as e: st.error(f"Failed to initialize LangChain LLM for provider {provider}: {e}"); return None