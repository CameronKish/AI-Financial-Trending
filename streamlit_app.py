# streamlit_app.py
import streamlit as st
import pandas as pd
from datetime import datetime
import utils 

st.set_page_config(layout="wide", page_title="Financial Analyzer Suite")

utils.ensure_settings_loaded() 

st.sidebar.header("App Navigation")

st.title("Welcome to the Financial Analyzer Suite")

if not st.session_state.get('uploaded_data_ready', False):
    st.warning("No data has been uploaded and processed yet. Please go to the 'Data Upload & Validation' page to begin.", icon="⚠️")
    st.markdown(f"""
    <div style="background-color:{utils.EY_DARK_BLUE_GREY}; padding: 15px; border-radius: 5px; color: white;">
    To start your analysis:
    <ol>
        <li>Navigate to the <b>📄 Data Upload & Validation</b> page using the sidebar ( < ).</li>
        <li>Upload your Excel file containing Journal Entry data.</li>
        <li>Follow the steps on the page to map columns and generate initial financial statements.</li>
    </ol>
    Once data is processed, you can access the other analysis pages.
    </div>
    <br/>
    """, unsafe_allow_html=True)
else:
    st.success("Data has been successfully uploaded and processed! You can now navigate to the analysis pages.", icon="🎉")
    st.markdown(f"""
    <div style="background-color:{utils.EY_DARK_BLUE_GREY}; padding: 15px; border-radius: 5px; color: white;">
    Use the navigation sidebar on the left ( < ) to switch between the analysis pages:
    <ul>
        <li><b>📄 Data Upload & Validation:</b> Upload new data or review current settings.</li>
        <li><b>📊 Financial Statement Analysis:</b> View P&L/BS, drill down into Journal Entries, and analyze period activity with an LLM.</li>
        <li><b>📈 Visualizations:</b> Explore trends and patterns in your P&L, BS, and JE data.</li>
        <li><b>🤖 AI Insights:</b> Chat with an AI assistant to ask questions and get insights directly from your data.</li>
    </ul>
    </div>
    <br/>
    """, unsafe_allow_html=True)

st.divider()

st.header("LLM Configuration")
st.caption("Settings are saved automatically when you change a field.") # Added caption

provider_value = st.session_state.get('llm_provider', utils.DEFAULT_SETTINGS['llm_provider'])
provider_options = ["Ollama", "OpenAI", "Azure OpenAI"]
if provider_value not in provider_options: provider_value = provider_options[0] # Default to Ollama if invalid
provider_index = provider_options.index(provider_value)

def save_current_settings():
    current_settings = {}
    for key in utils.DEFAULT_SETTINGS.keys():
        current_settings[key] = st.session_state.get(key, utils.DEFAULT_SETTINGS.get(key))
    utils.save_settings(current_settings)
    st.toast("LLM settings updated!", icon="⚙️") # Provide feedback on auto-save

# Main provider selection
st.selectbox("Select LLM Provider:", options=provider_options, 
             key='llm_provider', 
             on_change=save_current_settings, # This will also save if provider changes
             index=provider_index)

if st.session_state.llm_provider == "Ollama":
    st.subheader("Ollama Settings")
    st.markdown(f"Connects to Ollama at: `{utils.OLLAMA_BASE_URL}` (ensure Ollama is running).")
    
    local_models = utils.get_local_ollama_models() 
    
    if local_models:
        current_chosen_model_state = st.session_state.get('chosen_ollama_model')
        model_index = 0 
        if current_chosen_model_state and current_chosen_model_state in local_models:
            model_index = local_models.index(current_chosen_model_state)
        elif not current_chosen_model_state and local_models: 
            st.session_state.chosen_ollama_model = local_models[0] 
            # No need to call save_current_settings() here, 
            # the selectbox's on_change or the main provider's on_change will handle it
            # if this default assignment is considered a "change" by the user interacting with the selectbox.
            # For robustness, the selectbox's on_change will handle saving the actual selection.
        
        selected_model_ollama = st.selectbox(
            "Select Local Ollama Model:",
            options=local_models,
            index=model_index,
            key='widget_chosen_ollama_model_selector', # Ensure key for selectbox state
            help="Select a model from your locally running Ollama instance. Refresh page if new models were recently pulled.",
            on_change= lambda: ( # Use lambda for multi-action on_change
                st.session_state.update(chosen_ollama_model=st.session_state.widget_chosen_ollama_model_selector), # Update the main state key
                save_current_settings() # Then save
            )
        )
        # If the widget value is different from the main state key (e.g. on initial load with default), sync them
        if st.session_state.widget_chosen_ollama_model_selector != st.session_state.get('chosen_ollama_model'):
            st.session_state.chosen_ollama_model = st.session_state.widget_chosen_ollama_model_selector
            # save_current_settings() # Already handled by on_change if it was a user action
            # st.rerun() # Rerun might be useful if other parts of the UI depend on chosen_ollama_model immediately

        st.caption("If models are missing: 1. Ensure Ollama is running. 2. Pull models (e.g., `ollama pull mistral`). 3. Refresh this page.")

    else:
        st.warning(f"""
        Could not fetch local Ollama models or no models found.
        - Ensure Ollama is running and accessible at `{utils.OLLAMA_BASE_URL}`.
        - Ensure you have pulled models (e.g., run `ollama pull mistral` in your terminal).
        - Check the terminal where you launched Streamlit for any connection error messages.
        """)
        if st.session_state.get('chosen_ollama_model') is not None:
            st.session_state.chosen_ollama_model = None 
            save_current_settings()

elif st.session_state.llm_provider == "OpenAI":
    st.subheader("OpenAI Settings")
    # Ensure 'openai_model_name' and 'openai_api_key' are in session_state before widgets access them
    if 'openai_model_name' not in st.session_state: st.session_state.openai_model_name = utils.DEFAULT_SETTINGS['openai_model_name']
    if 'openai_api_key' not in st.session_state: st.session_state.openai_api_key = utils.DEFAULT_SETTINGS['openai_api_key']

    st.text_input("OpenAI Model Name:", 
                  key="widget_openai_model_name", # Use widget-specific key
                  value=st.session_state.openai_model_name, 
                  on_change=lambda: (
                      st.session_state.update(openai_model_name=st.session_state.widget_openai_model_name),
                      save_current_settings()
                  ))
    st.text_input("OpenAI API Key:", type="password", 
                  key="widget_openai_api_key", # Use widget-specific key
                  value=st.session_state.openai_api_key, 
                  on_change=lambda: (
                      st.session_state.update(openai_api_key=st.session_state.widget_openai_api_key),
                      save_current_settings()
                  ))
    st.caption("Ensure your API key has the necessary permissions for the selected model.")

elif st.session_state.llm_provider == "Azure OpenAI":
    st.subheader("Azure OpenAI Settings")
    # Ensure keys exist in session_state
    if 'azure_endpoint' not in st.session_state: st.session_state.azure_endpoint = utils.DEFAULT_SETTINGS['azure_endpoint']
    if 'azure_api_key' not in st.session_state: st.session_state.azure_api_key = utils.DEFAULT_SETTINGS['azure_api_key']
    if 'azure_deployment_name' not in st.session_state: st.session_state.azure_deployment_name = utils.DEFAULT_SETTINGS['azure_deployment_name']
    if 'azure_api_version' not in st.session_state: st.session_state.azure_api_version = utils.DEFAULT_SETTINGS['azure_api_version']

    st.text_input("Azure OpenAI Endpoint:", 
                  key="widget_azure_endpoint", value=st.session_state.azure_endpoint, 
                  on_change=lambda: (st.session_state.update(azure_endpoint=st.session_state.widget_azure_endpoint), save_current_settings()), 
                  placeholder="e.g., https://your-resource-name.openai.azure.com/")
    st.text_input("Azure OpenAI API Key:", type="password", 
                  key="widget_azure_api_key", value=st.session_state.azure_api_key, 
                  on_change=lambda: (st.session_state.update(azure_api_key=st.session_state.widget_azure_api_key), save_current_settings()))
    st.text_input("Azure Deployment Name (Model ID):", 
                  key="widget_azure_deployment_name", value=st.session_state.azure_deployment_name, 
                  on_change=lambda: (st.session_state.update(azure_deployment_name=st.session_state.widget_azure_deployment_name), save_current_settings()))
    st.text_input("Azure API Version:", 
                  key="widget_azure_api_version", value=st.session_state.azure_api_version, 
                  on_change=lambda: (st.session_state.update(azure_api_version=st.session_state.widget_azure_api_version), save_current_settings()), 
                  placeholder="e.g., 2024-02-01")

# Shared settings
if 'llm_context_limit' not in st.session_state: st.session_state.llm_context_limit = utils.DEFAULT_SETTINGS['llm_context_limit']
st.number_input("LLM Context Token Limit (Advanced):", 
                min_value=1024, 
                value=st.session_state.llm_context_limit, 
                step=1024, 
                key="widget_llm_context_limit", # Use widget-specific key
                on_change=lambda: (
                    st.session_state.update(llm_context_limit=st.session_state.widget_llm_context_limit),
                    save_current_settings()
                ),
                help="Maximum tokens the LLM can process. Default is usually fine.")

# Removed the "Save LLM Configuration" button
# st.markdown("---") 
# if st.button("Save LLM Configuration", key="save_llm_config_button_main"):
#     save_current_settings()
#     st.toast("LLM configuration saved!", icon="✅")

st.divider()
if not st.session_state.get('uploaded_data_ready', False):
    st.info("Go to 'Data Upload & Validation' to get started.")
else:
    st.info("Select a page from the sidebar to continue analysis.")