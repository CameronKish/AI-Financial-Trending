# pages/2_📈_Visualizations.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Needed if you add Waterfall later

# Import shared utilities
import utils

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Visualizations")
st.markdown(f"<style> h2 {{ color: {utils.EY_DARK_BLUE_GREY}; }} </style>", unsafe_allow_html=True)

# --- Check if data is loaded ---
if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
    st.error("Data not loaded. Please go back to the main page and ensure data is loaded correctly.")
    st.stop()

# --- Retrieve data from Session State ---
try:
    pl_flat_df = st.session_state.pl_flat_df
    je_detail_df = st.session_state.je_detail_df
    col_config = st.session_state.column_config
    PL_MAP_COLUMN = col_config["PL_MAP_DISPLAY"]
    JE_AMOUNT_COLUMN = col_config["JE_AMOUNT"]
    JE_ACCOUNT_NAME_COL = col_config["JE_ACCOUNT_NAME"] # Can be None
    # Add other needed columns like JE_DATE_COLUMN if used directly here later
except KeyError as e:
     st.error(f"Missing required data or configuration in session state: {e}. Please reload the app.")
     st.stop()

# --- Visualization Content ---

st.title("Data Visualizations")

# 1. P&L Account Trends
st.markdown(f"<h2>P&L Account Trends</h2>", unsafe_allow_html=True)

# Use session state for multiselect persistence
if 'chart_accounts_selection' not in st.session_state:
    st.warning("Chart selection state not initialized. Please reload.")
    st.stop()

# Prepare options for multiselect
chart_account_options = sorted(pl_flat_df[PL_MAP_COLUMN].unique())

# Enhanced Multiselect with Select All/Clear
c1, c2, c3 = st.columns([4, 1, 1]) # Adjust ratios as needed

with c1:
    # Use st.session_state.chart_accounts_selection directly as default
    user_chart_selection = st.multiselect(
        "Select Account(s) to Plot:",
        options=chart_account_options,
        default=st.session_state.chart_accounts_selection,
        key="chart_multiselect_widget" # Assign a key
    )
    # Synchronize widget state back to session state if changed by user
    if user_chart_selection != st.session_state.chart_accounts_selection:
        st.session_state.chart_accounts_selection = user_chart_selection
        st.rerun() # Rerun to reflect the change immediately

with c2:
    st.markdown("<br>", unsafe_allow_html=True) # Align button vertically
    if st.button("Select All", key='select_all_chart', use_container_width=True):
        st.session_state.chart_accounts_selection = chart_account_options # Update state
        st.rerun() # Rerun to update multiselect

with c3:
    st.markdown("<br>", unsafe_allow_html=True) # Align button vertically
    if st.button("Clear All", key='clear_chart', use_container_width=True):
        st.session_state.chart_accounts_selection = [] # Update state
        st.rerun() # Rerun to update multiselect

# Plotting logic using the selection stored in session state
if st.session_state.chart_accounts_selection:
    # Filter data based on session state selection
    chart_data = pl_flat_df[pl_flat_df[PL_MAP_COLUMN].isin(st.session_state.chart_accounts_selection)].copy()

    # Determine x-axis (prefer datetime if available and clean)
    if 'Period_dt' in chart_data.columns and chart_data['Period_dt'].notna().all():
        x_axis_col_chart = 'Period_dt'
        x_axis_label = "Period (Date)"
        chart_data = chart_data.sort_values(by=x_axis_col_chart)
    elif 'Period_Str' in chart_data.columns: # Fallback to string period
        x_axis_col_chart = 'Period_Str'
        x_axis_label = "Period (YYYY-MM)"
        # Attempt to sort string periods if possible (assuming YYYY-MM format)
        try:
            chart_data = chart_data.sort_values(by=x_axis_col_chart)
        except TypeError:
            pass # Ignore sort error if mixed types prevent it
    else: # Should not happen if data prep is correct
        x_axis_col_chart = 'Period'
        x_axis_label = "Period"

    if not chart_data.empty and x_axis_col_chart in chart_data.columns:
        try:
            fig = px.line(
                chart_data,
                x=x_axis_col_chart,
                y='Amount',
                color=PL_MAP_COLUMN,
                markers=True, # Add markers to points
                title="Monthly Trend for Selected Accounts"
            )
            fig.update_layout(
                xaxis_title=x_axis_label,
                yaxis_title="Amount ($)",
                yaxis_tickformat=",.0f", # Format Y axis
                hovermode="x unified" # Improved hover experience
            )
            # Customize hover template
            fig.update_traces(
                hovertemplate="<b>%{fullData.name}</b><br>Period: %{x}<br>Amount: %{y:,.0f}<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate P&L trend chart: {e}")
    else:
        st.info("No data available for the selected accounts or period column issue.")
else:
    st.info("Select one or more accounts using the controls above to display the trend chart.")


# 2. JE Analysis Scatter Plot (Using ALL JEs)
st.markdown(f"<hr><h2>Total JE Analysis (Amount vs. Frequency)</h2>", unsafe_allow_html=True)
st.caption("Analyzes ALL Journal Entries in the dataset by the selected category.")

if isinstance(je_detail_df, pd.DataFrame) and not je_detail_df.empty:
    # Define potential columns for analysis (text-based usually)
    potential_analysis_cols = ['Customer', 'Memo']
    if JE_ACCOUNT_NAME_COL: # Add Account Name if it exists
        potential_analysis_cols.append(JE_ACCOUNT_NAME_COL)

    # Filter for columns that actually exist and have more than one unique value
    analysis_cols_options = [
        col for col in potential_analysis_cols if col in je_detail_df.columns and
        je_detail_df[col].nunique() > 1 and
        (pd.api.types.is_string_dtype(je_detail_df[col]) or pd.api.types.is_object_dtype(je_detail_df[col]))
    ]

    # Check if the amount column exists
    if JE_AMOUNT_COLUMN not in je_detail_df.columns:
        st.warning(f"Amount column '{JE_AMOUNT_COLUMN}' not found in JE data. Cannot perform analysis.")
        analysis_cols_options = [] # Disable analysis if no amount

    if analysis_cols_options:
        selected_analysis_col_all = st.selectbox(
            "Analyze ALL Journal Entries by:",
            options=analysis_cols_options,
            index=0, # Default to first option
            key="je_analysis_col_tab2_all"
        )

        if selected_analysis_col_all:
            st.write(f"Aggregating all {len(je_detail_df):,} journal entries by **{selected_analysis_col_all}**...")
            try:
                # Perform aggregation within a spinner
                with st.spinner(f"Aggregating by {selected_analysis_col_all}..."):
                    # Ensure Transaction Id exists or use another column for counting
                    count_col = 'Transaction Id' if 'Transaction Id' in je_detail_df.columns else JE_ID_COLUMN

                    agg_data_all = je_detail_df.groupby(selected_analysis_col_all).agg(
                        Count=(count_col, 'size'), # Count occurrences
                        Total_Amount=(JE_AMOUNT_COLUMN, 'sum') # Sum amounts
                    ).reset_index()

                    # Calculate absolute amount for bubble size
                    agg_data_all['Abs_Total_Amount'] = agg_data_all['Total_Amount'].abs()
                    # Sort for better visualization or table display
                    agg_data_all = agg_data_all.sort_values(by='Total_Amount', ascending=False)

                if not agg_data_all.empty:
                    st.write(f"Relationship between Frequency and Total Amount for **{selected_analysis_col_all}** categories across all data:")
                    fig_agg_all = px.scatter(
                        agg_data_all,
                        x='Count',
                        y='Total_Amount',
                        size='Abs_Total_Amount', # Size bubbles by absolute amount
                        hover_name=selected_analysis_col_all, # Show category on hover
                        color='Total_Amount', # Color bubbles by net amount
                        color_continuous_scale=px.colors.diverging.Picnic, # Example diverging scale
                        title=f'All JE Analysis by {selected_analysis_col_all} (Amount vs. Frequency)',
                        labels={'Count': 'Number of Entries (Frequency)', 'Total_Amount': 'Net Total Amount ($)'},
                        size_max=60 # Control max bubble size
                    )
                    fig_agg_all.update_layout(
                        xaxis_title="Number of Entries (Frequency)",
                        yaxis_title="Net Total Amount ($)",
                        yaxis_tickformat=",.0f",
                        xaxis_tickformat=",d" # Format frequency axis
                    )
                    # Customize hover template for scatter plot
                    fig_agg_all.update_traces(
                        hovertemplate=f"<b>%{{hovertext}}</b><br>Net Total Amount: %{{y:,.0f}}<br>Count: %{{x}}<extra></extra>",
                        hovertext=agg_data_all[selected_analysis_col_all] # Ensure hovertext matches category
                    )
                    st.plotly_chart(fig_agg_all, use_container_width=True)

                    # Display Top N categories table
                    N_TOP = 25
                    st.write(f"Top {min(N_TOP, len(agg_data_all))} categories by Net Amount (All JEs):")
                    st.dataframe(
                        agg_data_all.head(N_TOP),
                        use_container_width=True,
                        column_config={
                            selected_analysis_col_all : st.column_config.TextColumn(selected_analysis_col_all),
                            "Count": st.column_config.NumberColumn("Count", format="%d"),
                            "Total_Amount": st.column_config.NumberColumn("Net Total Amount ($)", format="$ {:,.0f}"),
                            "Abs_Total_Amount": None # Hide Abs Amount column
                        },
                        hide_index=True
                    )
                else:
                    st.info(f"No aggregated data to display for '{selected_analysis_col_all}' across all JEs.")
            except KeyError as e_agg_key:
                st.error(f"Error during aggregation: Column mismatch? Required columns might be missing. Details: {e_agg_key}")
            except Exception as e_agg:
                st.error(f"An error occurred during the full JE analysis: {e_agg}")
                st.exception(e_agg)
    else:
        st.info("No suitable text columns found in the full JE dataset for this type of analysis.")
else:
    st.warning("Full JE data (`je_detail_df`) is not available or empty. Cannot perform analysis.")

# 3. Placeholder for Other Visualizations
st.markdown(f"<hr><h2>Other Visualizations (Placeholders)</h2>", unsafe_allow_html=True)
st.info("Waterfall charts, histograms, or other analyses can be added here.")

# Example Placeholder: P&L Waterfall (requires specific logic based on account structure)
# st.subheader("P&L Waterfall Chart (Example)")
# selected_waterfall_period = st.selectbox("Select Period for Waterfall:", options=pl_flat_df['Period_Str'].unique())
# if selected_waterfall_period:
#    st.info("Waterfall chart logic needs to be implemented based on your specific P&L structure (e.g., identifying Revenue, COGS, Opex accounts).")
   # Add waterfall logic here using plotly.graph_objects.Waterfall

# Example Placeholder: JE Amount Distribution
# st.subheader("JE Amount Distribution (Histogram)")
# if isinstance(je_detail_df, pd.DataFrame) and not je_detail_df.empty and JE_AMOUNT_COLUMN in je_detail_df.columns:
#    st.info("Displaying distribution of non-zero JE amounts.")
#    fig_hist = px.histogram(je_detail_df[je_detail_df[JE_AMOUNT_COLUMN] != 0], x=JE_AMOUNT_COLUMN, title="Distribution of JE Amounts (Non-Zero)")
#    st.plotly_chart(fig_hist, use_container_width=True)