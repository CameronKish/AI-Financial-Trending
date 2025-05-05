# pages/2_📈_Visualizations.py
# **** START OF FULL SCRIPT ****
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import utils # Includes ensure_settings_loaded

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Visualizations")
st.markdown(f"<style> h2 {{ color: {utils.EY_DARK_BLUE_GREY}; }} </style>", unsafe_allow_html=True)

# --- Ensure Settings are Loaded into Session State ---
# CALL THIS ON EVERY RUN, AT THE TOP (Re-added based on previous findings)
utils.ensure_settings_loaded()

# --- Check if data is loaded ---
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.error("Data not loaded. Please go back to the main page and ensure data is loaded correctly.")
    st.stop()

# --- Retrieve data from Session State ---
# Use .get() for safer retrieval in case keys are missing despite initialization
pl_flat_df = st.session_state.get('pl_flat_df', pd.DataFrame())
je_detail_df = st.session_state.get('je_detail_df', pd.DataFrame())
col_config = st.session_state.get('column_config', {})

# Safely get column names from config, providing fallbacks
PL_MAP_COLUMN = col_config.get("PL_MAP_DISPLAY", 'Account') # Use a generic fallback if needed
JE_AMOUNT_COLUMN = col_config.get("JE_AMOUNT", 'Amount')
JE_ACCOUNT_NAME_COL = col_config.get("JE_ACCOUNT_NAME", None)
JE_ID_COLUMN = col_config.get("JE_ID", None)

# Validate essential dataframes and columns needed for this page
if pl_flat_df.empty or PL_MAP_COLUMN not in pl_flat_df.columns:
    st.error(f"P&L Data ('pl_flat_df') is empty or missing required column '{PL_MAP_COLUMN}'. Cannot proceed.")
    st.stop()
if je_detail_df.empty or JE_AMOUNT_COLUMN not in je_detail_df.columns:
    st.warning(f"JE Data ('je_detail_df') is empty or missing amount column '{JE_AMOUNT_COLUMN}'. Some visualizations may be unavailable.")
    # Don't stop, P&L chart might still work


# --- Initialize Chart Selection State if needed ---
if 'chart_accounts_selection' not in st.session_state:
    try:
        temp_account_options = sorted(pl_flat_df[PL_MAP_COLUMN].unique().tolist())
        default_chart_selection = []
        potential_defaults = ["Total Net Sales", "Total COGS/COS", "Total Operating Expenses"]
        for acc in potential_defaults:
            if acc in temp_account_options: default_chart_selection.append(acc)
        if not default_chart_selection and temp_account_options:
            default_chart_selection = temp_account_options[:min(3, len(temp_account_options))]
        st.session_state.chart_accounts_selection = default_chart_selection
    except Exception as e_init_chart:
        st.warning(f"Could not initialize chart selections: {e_init_chart}")
        st.session_state.chart_accounts_selection = []


# --- Visualization Content ---
st.title("Data Visualizations")

# 1. P&L Account Trends
st.markdown(f"<h2>P&L Account Trends</h2>", unsafe_allow_html=True)

# Check if selections state exists
if 'chart_accounts_selection' not in st.session_state:
    st.warning("Chart selection state error. Please reload.")
    st.stop()

try:
    chart_account_options = sorted(pl_flat_df[PL_MAP_COLUMN].unique())
except Exception as e:
    st.error(f"Could not get account options for chart: {e}")
    chart_account_options = []

c1, c2, c3 = st.columns([4, 1, 1])
with c1:
    # Ensure default is list, handle potential state issues
    default_selection = st.session_state.chart_accounts_selection if isinstance(st.session_state.chart_accounts_selection, list) else []
    user_chart_selection = st.multiselect(
        "Select Account(s) to Plot:",
        options=chart_account_options,
        default=default_selection,
        key="chart_multiselect_widget"
    )
    # Update state only if the user interaction caused a change
    if user_chart_selection != st.session_state.chart_accounts_selection:
        st.session_state.chart_accounts_selection = user_chart_selection
        st.rerun()
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Select All", key='select_all_chart', use_container_width=True):
        st.session_state.chart_accounts_selection = chart_account_options
        st.rerun() # Keep rerun for immediate UI update
with c3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear All", key='clear_chart', use_container_width=True):
        st.session_state.chart_accounts_selection = []
        st.rerun() # Keep rerun for immediate UI update

# Plotting logic
if st.session_state.chart_accounts_selection:
    try:
        # Filter data safely
        chart_data = pl_flat_df[pl_flat_df[PL_MAP_COLUMN].isin(st.session_state.chart_accounts_selection)].copy()

        # Check if essential columns exist before proceeding
        if 'Amount' not in chart_data.columns:
             st.warning("Selected P&L data is missing the 'Amount' column.")
        else:
            # Determine x-axis (prefer datetime if available and clean)
            if 'Period_dt' in chart_data.columns and chart_data['Period_dt'].notna().all():
                x_axis_col_chart = 'Period_dt'; x_axis_label = "Period (Date)"; chart_data = chart_data.sort_values(by=x_axis_col_chart)
            elif 'Period_Str' in chart_data.columns:
                x_axis_col_chart = 'Period_Str'; x_axis_label = "Period (YYYY-MM)";
                try: chart_data = chart_data.sort_values(by=x_axis_col_chart)
                except TypeError: pass # Keep silent pass for sorting mixed types
            elif 'Period' in chart_data.columns: # Fallback to original Period column if others missing
                x_axis_col_chart = 'Period'; x_axis_label = "Period"
            else:
                x_axis_col_chart = None # Indicate x-axis cannot be determined
                st.warning("Could not determine a valid Period column for the P&L chart.")

            if x_axis_col_chart and not chart_data.empty:
                fig = px.line( chart_data, x=x_axis_col_chart, y='Amount', color=PL_MAP_COLUMN, markers=True, title="Monthly Trend for Selected Accounts" )
                fig.update_layout( xaxis_title=x_axis_label, yaxis_title="Amount ($)", yaxis_tickformat=",.0f", hovermode="x unified" )
                fig.update_traces( hovertemplate="<b>%{fullData.name}</b><br>Period: %{x}<br>Amount: %{y:,.0f}<extra></extra>" )
                st.plotly_chart(fig, use_container_width=True)
            elif not chart_data.empty:
                pass # Warning about missing period column already shown
            else:
                st.info("No P&L data available for the selected accounts.")

    except Exception as e:
        st.error(f"Could not generate P&L trend chart: {e}")
        st.exception(e) # Show full traceback for debugging
else:
    st.info("Select one or more accounts using the controls above to display the trend chart.")


# 2. JE Analysis Scatter Plot (Using ALL JEs)
st.markdown(f"<hr><h2>Total JE Analysis (Amount vs. Frequency)</h2>", unsafe_allow_html=True)
st.caption("Analyzes ALL Journal Entries in the dataset by the selected category.")

# Check if JE data is available and has the amount column
if isinstance(je_detail_df, pd.DataFrame) and not je_detail_df.empty and JE_AMOUNT_COLUMN in je_detail_df.columns:
    potential_analysis_cols = ['Customer', 'Memo']
    if JE_ACCOUNT_NAME_COL and JE_ACCOUNT_NAME_COL in je_detail_df.columns: # Check if account name col exists
        potential_analysis_cols.append(JE_ACCOUNT_NAME_COL)

    # Filter for columns that actually exist and have string/object type and more than one unique value
    analysis_cols_options = [
        col for col in potential_analysis_cols
        if col in je_detail_df.columns and
           je_detail_df[col].nunique() > 1 and
           (pd.api.types.is_string_dtype(je_detail_df[col]) or pd.api.types.is_object_dtype(je_detail_df[col]))
    ]

    if analysis_cols_options:
        selected_analysis_col_all = st.selectbox(
            "Analyze ALL Journal Entries by:", options=analysis_cols_options,
            index=0, key="je_analysis_col_tab2_all"
        )

        if selected_analysis_col_all:
            st.write(f"Aggregating all {len(je_detail_df):,} journal entries by **{selected_analysis_col_all}**...")
            try:
                with st.spinner(f"Aggregating by {selected_analysis_col_all}..."):
                    # Determine the column for counting entries robustly
                    count_col = None
                    if 'Transaction Id' in je_detail_df.columns: count_col = 'Transaction Id'
                    elif JE_ID_COLUMN and JE_ID_COLUMN in je_detail_df.columns: count_col = JE_ID_COLUMN

                    # Perform aggregation
                    if count_col:
                         agg_data_all = je_detail_df.groupby(selected_analysis_col_all).agg(
                             Count=(count_col, 'size'), Total_Amount=(JE_AMOUNT_COLUMN, 'sum')
                         ).reset_index()
                    else:
                         # Fallback if no suitable ID column - might double count if index isn't unique per JE
                         st.warning("Using DataFrame index for counting JEs as no unique ID column found.")
                         agg_data_all = je_detail_df.reset_index().groupby(selected_analysis_col_all).agg(
                             Count=('index', 'size'), Total_Amount=(JE_AMOUNT_COLUMN, 'sum')
                         ).reset_index()

                    agg_data_all['Abs_Total_Amount'] = agg_data_all['Total_Amount'].abs()
                    agg_data_all = agg_data_all.sort_values(by='Total_Amount', ascending=False)

                if not agg_data_all.empty:
                    st.write(f"Relationship between Frequency and Total Amount for **{selected_analysis_col_all}** categories:")
                    fig_agg_all = px.scatter(
                        agg_data_all, x='Count', y='Total_Amount', size='Abs_Total_Amount',
                        hover_name=selected_analysis_col_all, color='Total_Amount',
                        color_continuous_scale=px.colors.diverging.Picnic,
                        title=f'All JE Analysis by {selected_analysis_col_all} (Amount vs. Frequency)',
                        labels={'Count': 'Number of Entries (Frequency)', 'Total_Amount': 'Net Total Amount ($)'},
                        size_max=60
                    )
                    fig_agg_all.update_layout( xaxis_title="Frequency", yaxis_title="Net Amount ($)", yaxis_tickformat=",.0f", xaxis_tickformat=",d" )
                    fig_agg_all.update_traces( hovertemplate=f"<b>%{{hovertext}}</b><br>Net Amount: %{{y:,.0f}}<br>Count: %{{x}}<extra></extra>", hovertext=agg_data_all[selected_analysis_col_all] )
                    st.plotly_chart(fig_agg_all, use_container_width=True)

                    # Display Top N table
                    N_TOP = 25; st.write(f"Top {min(N_TOP, len(agg_data_all))} categories by Net Amount:")
                    st.dataframe( agg_data_all.head(N_TOP), use_container_width=True,
                        column_config={ selected_analysis_col_all : st.column_config.TextColumn(selected_analysis_col_all), "Count": st.column_config.NumberColumn("Count", format="%d"), "Total_Amount": st.column_config.NumberColumn("Net Amount ($)", format="$ {:,.0f}"), "Abs_Total_Amount": None },
                        hide_index=True )
                else: st.info(f"No aggregated data for '{selected_analysis_col_all}'.")
            except KeyError as e_agg_key: st.error(f"Aggregation Error: Column mismatch? {e_agg_key}")
            except Exception as e_agg: st.error(f"JE Analysis Error: {e_agg}"); st.exception(e_agg)
    else:
        st.info("No suitable text columns with multiple unique values found in JE data for this analysis.")
elif isinstance(je_detail_df, pd.DataFrame) and not je_detail_df.empty:
     st.warning(f"Amount column '{JE_AMOUNT_COLUMN}' not found in JE data. Cannot perform JE scatter plot analysis.")
else:
    st.warning("JE data (`je_detail_df`) is not available or empty. Cannot perform JE scatter plot analysis.")


# 3. Placeholder for Other Visualizations
st.markdown(f"<hr><h2>Other Visualizations (Placeholders)</h2>", unsafe_allow_html=True)
st.info("Waterfall charts, histograms, or other analyses can be added here.")

# **** END OF FULL SCRIPT ****