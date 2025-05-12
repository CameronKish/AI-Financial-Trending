#utils.py
# **** START OF FULL SCRIPT ****
import pandas as pd
import numpy as np
import requests
import json
import os
import streamlit as st
from openai import OpenAI, AzureOpenAI

# --- LangChain Imports ---
# Add these imports for LangChain LLM wrappers
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage # Example, might be needed later


# --- Define EY Parthenon Inspired Colors ---
EY_YELLOW = "#FFE600"; EY_DARK_BLUE_GREY = "#2E2E38"; EY_TEXT_ON_YELLOW = EY_DARK_BLUE_GREY

# --- Ollama Base URL ---
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Settings File Path ---
# Store settings in the same directory as the script for simplicity
SETTINGS_FILE = "llm_analyzer_settings.json"

# --- Settings Defaults ---
DEFAULT_SETTINGS = { # Define defaults in one place
    'llm_provider': 'Ollama',
    'chosen_ollama_model': None,
    'openai_api_key': '',
    'openai_model_name': "gpt-4o-mini", # Updated Default
    'azure_endpoint': '',
    'azure_api_key': '',
    'azure_deployment_name': '',
    'azure_api_version': "2024-02-01", # Default Azure Version
    'llm_context_limit': 32000,
}

# --- Settings Load/Save Functions ---

def load_settings() -> dict:
    """Loads settings from the JSON file."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                loaded_settings = json.load(f)
                # Ensure all keys from DEFAULT_SETTINGS exist, add if missing
                settings = DEFAULT_SETTINGS.copy()
                settings.update(loaded_settings) # Overwrite defaults with loaded values
                return settings
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading settings file '{SETTINGS_FILE}': {e}. Using defaults.")
            # Fallback to defaults if file is corrupt or unreadable
            return DEFAULT_SETTINGS.copy()
    else:
        # Return defaults if file doesn't exist
        return DEFAULT_SETTINGS.copy()

def save_settings(settings_dict: dict):
    """Saves the provided settings dictionary to the JSON file."""
    try:
        # Only save keys that are in our DEFAULT_SETTINGS template
        settings_to_save = {k: settings_dict.get(k) for k in DEFAULT_SETTINGS}
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings_to_save, f, indent=4)
    except IOError as e:
        # Use st.error for visibility in the app if saving fails
        st.error(f"Error saving settings to '{SETTINGS_FILE}': {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while saving settings: {e}")


# **** CORRECTED FUNCTION TO LOAD SETTINGS INTO SESSION STATE ****
def ensure_settings_loaded():
    """Loads settings from file and ensures they are in session state."""
    app_settings = load_settings()
    # print(f"DEBUG ensure_settings_loaded: Loaded {app_settings}") # Optional debug
    settings_keys = DEFAULT_SETTINGS.keys()
    for key in settings_keys:
        # ALWAYS update session state from file/defaults at start of page run
        # This overwrites potentially stale/corrupted state from page transitions
        st.session_state[key] = app_settings.get(key, DEFAULT_SETTINGS.get(key))
        # Optional Debug: print(f"DEBUG ensure_settings_loaded: Set st.session_state['{key}'] = {st.session_state[key]}")

# --- Other Utility Functions ---
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
        if i == 0: continue
        diff_val = row_diff.get(period)
        if pd.notna(diff_val) and abs(diff_val) > threshold_value:
            if i < len(styles): styles[i] = f'background-color: {color}; color: {text_color};'
    return styles

def format_je_for_llm(je_df, amount_col, max_raw_rows=30):
    if je_df is None or je_df.empty: return "No Journal Entry details provided."
    if amount_col not in je_df.columns:
        potential_amount_cols = [col for col in je_df.columns if 'Amount' in col]
        if potential_amount_cols: amount_col = potential_amount_cols[0]
        else: return f"JE details missing a recognizable amount column."
    je_df_copy = je_df.copy(); je_df_copy[amount_col] = pd.to_numeric(je_df_copy[amount_col], errors='coerce')
    je_df_clean = je_df_copy.dropna(subset=[amount_col]); num_rows = len(je_df_clean)
    if num_rows == 0: return "No valid JE details found (after cleaning)."
    cols_to_show = [col for col in ['Transaction Date', 'Memo', 'Customer', 'Transaction Id', amount_col] if col in je_df_clean.columns]
    if amount_col in cols_to_show: cols_to_show.remove(amount_col); cols_to_show.append(amount_col)
    if not cols_to_show: cols_to_show = je_df_clean.columns.tolist()
    if num_rows <= max_raw_rows:
        df_markdown = je_df_clean[cols_to_show].copy()
        df_markdown[amount_col] = df_markdown[amount_col].apply(format_amount_safely)
        if 'Transaction Date' in df_markdown.columns:
             try: df_markdown['Transaction Date'] = pd.to_datetime(df_markdown['Transaction Date']).dt.strftime('%Y-%m-%d')
             except Exception: df_markdown['Transaction Date'] = df_markdown['Transaction Date'].astype(str)
        return f"Found {num_rows} entries:\n" + df_markdown.to_markdown(index=False)
    else: # Format summary
        total_amount = je_df_clean[amount_col].sum(); num_positive = len(je_df_clean[je_df_clean[amount_col] > 0]); num_negative = len(je_df_clean[je_df_clean[amount_col] < 0])
        je_df_clean['Abs_Amount'] = je_df_clean[amount_col].abs(); top_n = min(5, num_rows); top_entries = je_df_clean.nlargest(top_n, 'Abs_Amount')
        top_entries_markdown = top_entries[cols_to_show].copy(); top_entries_markdown[amount_col] = top_entries_markdown[amount_col].apply(format_amount_safely)
        if 'Transaction Date' in top_entries_markdown.columns:
             try: top_entries_markdown['Transaction Date'] = pd.to_datetime(top_entries_markdown['Transaction Date']).dt.strftime('%Y-%m-%d')
             except Exception: top_entries_markdown['Transaction Date'] = top_entries_markdown['Transaction Date'].astype(str)
        summary = ( f"Summary of {num_rows} Journal Entries (due to large volume):\n- Net Amount: {format_amount_safely(total_amount)}\n- Positive Entries: {num_positive}\n- Negative Entries: {num_negative}\n\nTop {top_n} Entries by Absolute Amount:\n{top_entries_markdown.to_markdown(index=False)}" )
        top_n_agg = min(3, num_rows)
        if 'Customer' in je_df_clean.columns and not je_df_clean['Customer'].isnull().all():
            top_customers = je_df_clean.groupby('Customer')[amount_col].sum().nlargest(top_n_agg)
            if not top_customers.empty: summary += f"\n\nTop {len(top_customers)} Customers by Net Amount:\n" + top_customers.apply(format_amount_safely).to_markdown()
        if 'Memo' in je_df_clean.columns and not je_df_clean['Memo'].isnull().all():
            top_memos = je_df_clean.groupby('Memo')[amount_col].sum().nlargest(top_n_agg)
            if not top_memos.empty: summary += f"\n\nTop {len(top_memos)} Memos by Net Amount:\n" + top_memos.apply(format_amount_safely).to_markdown()
        return summary

def estimate_token_count(text: str) -> int:
    if not isinstance(text, str) or not text: return 0
    return int(len(text) / 4) + 1

# --- LLM Call Functions (Original - Kept for P&L Page Analysis) ---
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
    """ Original streaming function for P&L page direct LLM calls."""
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


# --- NEW: LangChain LLM Instantiation ---
def get_langchain_llm(temperature=0.0, streaming=True):
    """
    Instantiates and returns a LangChain Chat LLM based on session state settings.

    Args:
        temperature (float): The temperature to use for the LLM.
        streaming (bool): Whether to enable streaming from the LLM.

    Returns:
        A LangChain ChatLLM object (e.g., ChatOpenAI, ChatOllama, AzureChatOpenAI) or None if config is invalid.
    """
    # Ensure settings are loaded into session state first if needed elsewhere
    # ensure_settings_loaded() # Called at start of page scripts anyway

    provider = st.session_state.get('llm_provider')
    llm = None

    try:
        if provider == "Ollama":
            model_name = st.session_state.get('chosen_ollama_model')
            if not model_name:
                st.error("Ollama provider selected, but no model chosen/available.")
                return None
            llm = ChatOllama(
                model=model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature,
                # Streaming handled by agent executor potentially
            )
            print(f"INFO: Initialized ChatOllama with model: {model_name}")

        elif provider == "OpenAI":
            api_key = st.session_state.get('openai_api_key')
            model_name = st.session_state.get('openai_model_name', DEFAULT_SETTINGS['openai_model_name'])
            if not api_key:
                st.error("OpenAI provider selected, but API Key is missing.")
                return None
            llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                streaming=streaming,
                # max_tokens= # Consider adding if needed
            )
            print(f"INFO: Initialized ChatOpenAI with model: {model_name}")

        elif provider == "Azure OpenAI":
            api_key = st.session_state.get('azure_api_key')
            endpoint = st.session_state.get('azure_endpoint')
            deployment_name = st.session_state.get('azure_deployment_name')
            api_version = st.session_state.get('azure_api_version', DEFAULT_SETTINGS['azure_api_version'])
            if not all([api_key, endpoint, deployment_name, api_version]):
                st.error("Azure OpenAI provider selected, but configuration is incomplete.")
                return None
            llm = AzureChatOpenAI(
                azure_endpoint=endpoint,
                openai_api_key=api_key,
                azure_deployment=deployment_name,
                openai_api_version=api_version,
                temperature=temperature,
                streaming=streaming,
                # max_tokens= # Consider adding if needed
            )
            print(f"INFO: Initialized AzureChatOpenAI with deployment: {deployment_name}")

        else:
            st.error(f"Unsupported LLM provider selected: {provider}")
            return None

        return llm

    except Exception as e:
        st.error(f"Failed to initialize LangChain LLM for provider {provider}: {e}")
        return None

# **** END OF FULL SCRIPT ****