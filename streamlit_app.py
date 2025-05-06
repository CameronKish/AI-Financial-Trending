# **** START OF FULL SCRIPT ****
import streamlit as st
import pandas as pd
from datetime import datetime
import time
import requests
import data_processor
import utils # Now includes ensure_settings_loaded, DEFAULT_SETTINGS, get_langchain_llm
import tools # <--- ADD THIS IMPORT

# --- Page Config ---
st.set_page_config(layout="wide", page_title="P&L Analyzer")

# --- Styling & Constants ---
OLLAMA_BASE_URL = "http://localhost:11434"
# **** UPDATED OpenAI Model Options ****
OPENAI_MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4.1-nano"] # As requested by user
# Use default from utils to avoid duplication
AZURE_DEFAULT_API_VERSION = utils.DEFAULT_SETTINGS['azure_api_version']


# --- Ensure Settings are Loaded into Session State ---
# CALL THIS ON EVERY RUN, AT THE TOP
utils.ensure_settings_loaded()


# --- Data Loading & Trigger ---
@st.cache_data(ttl=3600)
def load_data_cached(): # Corrected version
    """Loads and processes data using data_processor and caches the result."""
    try:
        start_time = time.time(); pl_flat, je_detail = data_processor.load_and_process_data()
        end_time = time.time();
        if pl_flat is None or je_detail is None: return None, None, None
        je_account_name_col = 'Account Name' if 'Account Name' in je_detail.columns else None
        return pl_flat, je_detail, je_account_name_col
    except Exception as e: raise

def trigger_data_load():
    """Triggers data loading and handles exceptions."""
    try:
        pl_flat_df, je_detail_df, je_account_name_col = load_data_cached() # Call the cached function
        if pl_flat_df is not None and je_detail_df is not None:
            st.session_state.pl_flat_df = pl_flat_df; st.session_state.je_detail_df = je_detail_df
            st.session_state.data_loaded = True
            st.session_state.column_config = { "PL_ID": data_processor.PL_ID_COLUMN, "PL_MAP1": data_processor.PL_MAP1_COLUMN,"PL_MAP2": data_processor.PL_MAP2_COLUMN, "PL_MAP_DISPLAY": data_processor.PL_MAP_COLUMN, "JE_ID": data_processor.JE_ID_COLUMN, "JE_DATE": data_processor.JE_DATE_COLUMN, "JE_AMOUNT": data_processor.JE_AMOUNT_COLUMN, "JE_ACCOUNT_NAME": je_account_name_col, "JE_DETAILS_BASE": data_processor.JE_DETAIL_COLUMNS_BASE }
            if 'chart_accounts_selection' not in st.session_state:
                 temp_account_options = sorted(st.session_state.pl_flat_df[st.session_state.column_config["PL_MAP_DISPLAY"]].unique().tolist()); default_chart_selection = []; potential_defaults = ["Total Net Sales", "Total COGS/COS", "Total Operating Expenses"]
                 for acc in potential_defaults:
                     if acc in temp_account_options: default_chart_selection.append(acc)
                 if not default_chart_selection and temp_account_options: default_chart_selection = temp_account_options[:min(3, len(temp_account_options))]
                 st.session_state.chart_accounts_selection = default_chart_selection
            # Generate and store schema info for agent tools
            # No need to store schema in session state, the tool can generate it on the fly
            # try:
            #     st.session_state['pl_schema_info'] = {col: str(dtype) for col, dtype in pl_flat_df.dtypes.items()}
            #     st.session_state['je_schema_info'] = {col: str(dtype) for col, dtype in je_detail_df.dtypes.items()}
            # except Exception as e_schema:
            #     st.warning(f"Could not pre-generate schema info: {e_schema}")
            return True
        else:
             st.session_state.data_loaded = False; return False
    except Exception as e: # Catch exceptions from load_data_cached OR within this function
        st.error(f"CRITICAL ERROR during data loading trigger: {e}")
        st.session_state.data_loaded = False; return False

if 'data_loaded' not in st.session_state:
    if not trigger_data_load(): st.error("Data loading failed."); st.stop()

# --- Initialize OTHER Session State Keys ---
# LLM config keys are now handled by ensure_settings_loaded() called above
other_required_keys = {
    'selected_account_id': None, 'selected_account_name': None, 'selected_period': None, 'prev_selected_account_id': None, 'prev_selected_period': None,
    'related_jes_df': pd.DataFrame(columns=st.session_state.get('column_config', {}).get("JE_DETAILS_BASE", [])), 'needs_je_refetch': False,
    'dup_col': None, 'dup_val': None, 'dup_search_triggered': False, 'chart_accounts_selection': [], 'outlier_threshold': 2.0,
    'ollama_status': "Not Checked", 'ollama_models': [], # Keep dynamic Ollama state here
    'llm_analyses': {}, 'llm_streaming_key': None, # Keep runtime state here
    'ai_insights_messages': [], # Initialize chat history for new page
    tools.INTERMEDIATE_DF_KEY: None, # Initialize key used by agent tools
}
for key, default_value in other_required_keys.items():
     if key not in st.session_state: st.session_state[key] = default_value

# Remove backup keys explicitly if they exist (no longer needed)
st.session_state.pop('chosen_ollama_model_bck', None)
st.session_state.pop('llm_context_limit_bck', None)

# --- Callback to Save Settings ---
def save_current_settings():
    """Gathers current LLM settings from session state and saves them."""
    current_settings = {}
    # Use the keys defined in utils.DEFAULT_SETTINGS as the source of truth
    for key in utils.DEFAULT_SETTINGS.keys():
        # Use .get() to safely retrieve value or its default if key happens to be missing when called
        current_settings[key] = st.session_state.get(key, utils.DEFAULT_SETTINGS.get(key))
    utils.save_settings(current_settings)

# --- Global Sidebar Elements ---
st.sidebar.header("App Navigation")
current_time_disp = datetime.now()
st.sidebar.write(f"Timestamp: {current_time_disp.strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.write(f"Location Context: Denver, CO, USA")

# --- Main Page Content (Home) ---
st.title("Welcome to the P&L Analyzer")
st.markdown(f"""
<div style="background-color:{utils.EY_DARK_BLUE_GREY}; padding: 15px; border-radius: 5px; color: white;">
Use the navigation sidebar on the left ( < ) to switch between the analysis pages:
<ul>
    <li><b>📊 P&L Analysis & Drilldown:</b> View the P&L, identify outliers, drill down into Journal Entries, and analyze period activity with an LLM.</li>
    <li><b>📈 Visualizations:</b> Explore trends and patterns in your P&L and JE data.</li>
    <li><b>🤖 AI Insights:</b> Chat with an AI assistant to ask questions and get insights directly from your data.</li>
</ul>
Data is loaded from <code>{data_processor.EXCEL_FILE_PATH}</code>. Configure LLM settings below. Settings are saved locally to <code>{utils.SETTINGS_FILE}</code>.
</div>
""", unsafe_allow_html=True)
st.divider()

# --- LLM Configuration Section ---
st.header("LLM Configuration")
st.warning("API Keys entered below are stored in a local JSON file for persistence. Ensure this file is kept secure.", icon="⚠️")

# --- Provider Selection ---
# Read index safely using .get() which should be populated by ensure_settings_loaded()
provider_value = st.session_state.get('llm_provider', 'Ollama')
provider_options = ["Ollama", "OpenAI", "Azure OpenAI"]
# Ensure provider value is valid before finding index
if provider_value not in provider_options: provider_value = 'Ollama'
provider_index = provider_options.index(provider_value)
st.selectbox(
    "Select LLM Provider:", options=provider_options,
    key='llm_provider', on_change=save_current_settings, # Save on change
    index=provider_index
)
st.write(f"Selected Provider: **{st.session_state.llm_provider}**")
st.markdown("---")

# --- Ollama Configuration ---
if st.session_state.llm_provider == "Ollama":
    st.subheader("Ollama Settings")
    if st.button("Check Ollama & List Models"):
         with st.spinner("Contacting Ollama server..."):
            try:
                response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5); response.raise_for_status(); models_data = response.json()
                model_names = sorted([model['name'] for model in models_data.get('models', [])])
                if model_names: st.session_state.ollama_models = model_names; st.session_state.ollama_status = "Connected"
                else: st.session_state.ollama_models = []; st.session_state.ollama_status = "Connected (No Models Found)"
            except Exception as e: st.session_state.ollama_status = f"Failed ({type(e).__name__})"; st.session_state.ollama_models = []
         st.rerun()
    status = st.session_state.ollama_status
    if status == "Connected": st.success(f"✅ Connected. Found {len(st.session_state.ollama_models)} models.")
    elif status == "Connected (No Models Found)": st.warning("⚠️ Connected, but no models found.")
    elif status == "Failed (Connection Error)": st.error(f"❌ Failed to connect at `{OLLAMA_BASE_URL}`.")
    elif status.startswith("Failed"): st.error(f"❌ Ollama API Error: {status}")
    else: st.info("Click button to check Ollama connection.")
    available_models_widget = st.session_state.get('ollama_models', [])
    # Use .get() for chosen_ollama_model which is now reliably loaded/defaulted
    current_ollama_model = st.session_state.get('chosen_ollama_model')
    select_index_widget = utils.get_index(available_models_widget, current_ollama_model) if current_ollama_model and available_models_widget else 0
    st.selectbox( "Select Ollama Model:", options=available_models_widget, index=select_index_widget,
                 key='chosen_ollama_model', on_change=save_current_settings,
                 disabled=(status != "Connected" or not available_models_widget) )
    display_model = st.session_state.get('chosen_ollama_model')
    if display_model: st.write(f"Selected Model: `{display_model}`")
    else: st.write("No Ollama model selected.")

# --- OpenAI Configuration ---
elif st.session_state.llm_provider == "OpenAI":
    st.subheader("OpenAI Settings")
    st.text_input( "OpenAI API Key:", type="password", key="openai_api_key", on_change=save_current_settings, help="Stored locally in JSON file." )
    # Use NEW model list
    current_openai_model = st.session_state.get('openai_model_name', OPENAI_MODEL_OPTIONS[0])
    # Ensure index is valid for the NEW list
    if current_openai_model not in OPENAI_MODEL_OPTIONS: current_openai_model = OPENAI_MODEL_OPTIONS[0] # Default to first in new list
    openai_index = OPENAI_MODEL_OPTIONS.index(current_openai_model)
    st.selectbox( "OpenAI Model Name:", options=OPENAI_MODEL_OPTIONS, key="openai_model_name", on_change=save_current_settings, index=openai_index )
    st.write(f"Selected Model: `{st.session_state.openai_model_name}`")
    # Display API Key status without showing key
    if st.session_state.get('openai_api_key'): st.success("API Key entered.", icon="🔑")
    else: st.warning("API Key not entered.")


# --- Azure OpenAI Configuration ---
elif st.session_state.llm_provider == "Azure OpenAI":
    st.subheader("Azure OpenAI Settings")
    st.text_input( "Azure OpenAI Endpoint URL:", key="azure_endpoint", on_change=save_current_settings, placeholder="e.g., https://YOUR_RESOURCE_NAME.openai.azure.com/", help="Stored locally in JSON file." )
    st.text_input( "Azure API Key:", type="password", key="azure_api_key", on_change=save_current_settings, help="Stored locally in JSON file." )
    st.text_input( "Deployment Name:", key="azure_deployment_name", on_change=save_current_settings, placeholder="e.g., gpt-4o-deployment", help="Stored locally in JSON file." )
    # Ensure api_version has a value before passing to text_input if using value=
    if 'azure_api_version' not in st.session_state or not st.session_state.azure_api_version:
         st.session_state.azure_api_version = utils.DEFAULT_SETTINGS['azure_api_version']
    st.text_input( "Azure API Version:", key="azure_api_version", on_change=save_current_settings, placeholder=f"Default: {st.session_state.azure_api_version}" )
    # Ensure default is used if blank IN STATE before saving/using
    if not st.session_state.azure_api_version: st.session_state.azure_api_version = utils.DEFAULT_SETTINGS['azure_api_version']
    # Validation checks / Status
    if not st.session_state.get('azure_endpoint',"").startswith("https://"): st.warning("Endpoint URL doesn't look valid.")
    if not st.session_state.get('azure_api_key',""): st.warning("Azure API Key is missing.")
    else: st.success("Azure API Key entered.", icon="🔑")
    if not st.session_state.get('azure_deployment_name',""): st.warning("Deployment Name is missing.")

# --- Shared LLM Settings ---
st.markdown("---"); st.subheader("Shared Settings")
# Read limit from state, now reliably loaded/updated by ensure_settings_loaded()
current_limit_value = st.session_state.get('llm_context_limit', utils.DEFAULT_SETTINGS['llm_context_limit'])
# Ensure value passed to number_input is int
if not isinstance(current_limit_value, int): current_limit_value = utils.DEFAULT_SETTINGS['llm_context_limit']

st.number_input( "LLM Context Token Limit:", min_value=1024, max_value=256000,
    value=current_limit_value, # Use value directly from state
    step=1024, key='llm_context_limit', # Updates state
    on_change=save_current_settings, # Save any change
    help="Set approximate token limit for warnings." )
st.write(f"Context Limit Setting: `{st.session_state.llm_context_limit:,}` tokens")

st.divider(); st.info("Select a page from the sidebar to begin analysis.")
# **** END OF FULL SCRIPT ****