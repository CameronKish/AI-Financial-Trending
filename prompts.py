# prompts.py
import textwrap

def get_pnl_analysis_prompt(account_name, account_id, current_period, current_amount_str, previous_period, previous_amount_str, formatted_je_data, is_outlier_context=""):
    """
    Generates the prompt for P&L account analysis.
    Expects amounts to be pre-formatted strings.
    """
    # Use textwrap.dedent to handle indentation nicely in the prompt definition
    prompt_template = f"""
    You are a helpful financial analyst assistant.
    Analyze the following P&L account activity based *only* on the provided data context.

    **Context:**
    - Account Name: {account_name} (ID: {account_id})
    - Period Analyzed: {current_period}
    - Amount this Period: {current_amount_str}
    """
    if previous_period and previous_amount_str is not None:
         prompt_template += f"- Previous Period: {previous_period}\n- Amount Previous Period: {previous_amount_str}\n"
    else:
         prompt_template += "- Previous Period: N/A (This is the first period in the data)\n"

    # Add optional outlier context if provided
    if is_outlier_context:
        prompt_template += f"{is_outlier_context}\n"

    prompt_template += f"""
    **Journal Entry Details for {current_period}:**
    {formatted_je_data}

    **Your Task:**
    1. Briefly summarize (with bullet points) the main activities or types of transactions recorded in the provided journal entries for {current_period}.
    2. Based *strictly* on these journal entries, explain the most likely drivers for the account's balance or its change from the previous period.
    3. Be specific where possible (e.g., "driven by transaction ID XXX with Customer Y", "increase due to multiple entries related to Z expense category").
    4. Keep your explanation concise and focused solely on the provided JE details. A reader should be able to read at a glance it is so short and to the point. Do not invent information.
    """
    # Remove leading/trailing whitespace and dedent the whole block
    return textwrap.dedent(prompt_template).strip()

# --- Add other prompt functions here later ---
# def get_some_other_prompt(...):
#    ...
#    return textwrap.dedent(prompt).strip()