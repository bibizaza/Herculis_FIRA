# BondAnalysisApp.py
# Required packages: streamlit, pandas, plotly, scipy, numpy, statsmodels, openpyxl

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import numpy as np
from scipy.stats import shapiro, skew, kurtosis
import scipy.stats as stats
from scipy.stats import gaussian_kde
import os
import base64
from statsmodels.stats.stattools import durbin_watson

# =====================
# 1. Page Configuration
# =====================
st.set_page_config(page_title="📈 Bond Analysis Dashboard", layout="wide")

# =====================
# 2. CSS Styling for Expanders
# =====================
def inject_custom_css():
    """
    Inject custom CSS to style expanders for hover effects and border colors.
    """
    custom_css = """
    <style>
    /* Change background and text color on hover for expanders */
    .streamlit-expanderHeader:hover {
        background-color: #40E0D0 !important; /* Turquoise */
        color: white !important;
    }

    /* Change border color when expander is collapsed */
    .streamlit-expanderHeader[aria-expanded="false"] {
        border: 2px solid #40E0D0 !important; /* Turquoise border */
        border-radius: 5px;
        padding: 10px;
    }

    /* Remove default padding and margin */
    .streamlit-expanderHeader {
        padding: 10px !important;
        margin: 0 !important;
    }

    /* Change cursor to pointer when hovering over expander headers */
    .streamlit-expanderHeader {
        cursor: pointer;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# =====================
# 3. Data Processing Function
# =====================
@st.cache_data
def load_data(file):
    """
    Load bond data from an Excel file.

    Parameters:
    - file: Uploaded Excel file

    Returns:
    - df: Pandas DataFrame containing the bond data
    - required_columns: List of required columns
    """
    try:
        df = pd.read_excel(file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None, None

    required_columns = [
        'ISIN', 'Yield', 'G-Spread', 'Z-Spread', 'OAS',
        'Duration', 'Time To Maturity'
    ]
    if not all(column in df.columns for column in required_columns):
        st.error(f"Uploaded Excel file must contain the following columns with headers: {', '.join(required_columns)}")
        return None, None

    # Convert yields to float and check if they need to be converted to percentages
    df['Yield'] = df['Yield'].astype(float)
    if df['Yield'].max() < 1:
        df['Yield'] = df['Yield'] * 100

    # Handle missing Ratings by replacing NaN with "No Rating" if applicable
    if 'Rating' in df.columns:
        df['Rating'] = df['Rating'].fillna('No Rating')

    return df, required_columns

# =====================
# 4. NSS Model Function
# =====================
def nss_model(t, beta0, beta1, beta2, beta3, lambda1, lambda2):
    """
    Nelson-Siegel-Svensson (NSS) Model Function.

    Parameters:
    - t: Time to maturity
    - beta0, beta1, beta2, beta3: Coefficients
    - lambda1, lambda2: Decay factors

    Returns:
    - y(t): Yield or spread at time t
    """
    term1 = (1 - np.exp(-lambda1 * t)) / (lambda1 * t)
    term2 = term1 - np.exp(-lambda1 * t)
    term3 = ((1 - np.exp(-lambda2 * t)) / (lambda2 * t)) - np.exp(-lambda2 * t)
    return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3

def perform_nss_regression(t, y):
    """
    Perform NSS regression using nonlinear least squares.

    Parameters:
    - t: Independent variable (Time to Maturity)
    - y: Dependent variable (Yield or Spread)

    Returns:
    - y_pred: Predicted y values from the NSS model
    - params: Fitted parameters of the NSS model
    """
    # Adjust initial parameters based on the scale of y
    beta0_initial = np.mean(y)
    beta1_initial = -1.0
    beta2_initial = 1.0
    beta3_initial = -1.0
    lambda1_initial = 0.5
    lambda2_initial = 1.0

    initial_params = [beta0_initial, beta1_initial, beta2_initial, beta3_initial, lambda1_initial, lambda2_initial]
    bounds = (
        [-np.inf, -np.inf, -np.inf, -np.inf, 0.01, 0.01],
        [np.inf, np.inf, np.inf, np.inf, 10.0, 10.0]
    )

    try:
        params, covariance = curve_fit(
            nss_model,
            t,
            y,
            p0=initial_params,
            bounds=bounds,
            maxfev=100000
        )
        y_pred = nss_model(t, *params)
        return y_pred, params
    except RuntimeError as e:
        st.error(f"NSS regression failed to converge: {e}")
        return None, None  # Return None to indicate failure

# =====================
# 5. Introduction Section
# =====================
# 5. Introduction Section
def introduction():
    logo_path = get_logo_path()
    try:
        # Read the logo image in binary
        with open(logo_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()

        # Display logo, introduction text, and button all aligned to the left
        st.image(logo_path, width=200)
        st.title("Welcome to the Bond Analysis Dashboard")
        st.markdown("""
            This application allows you to visualize and analyze bond yields and spreads relative to a regression-based yield curve.
            Use the sidebar to upload your bond data and customize your analysis.

            ### How to Use:
            1. **Upload Your Excel File:** Ensure your Excel file contains the required columns.
            2. **Customize Your Analysis:** Choose metrics, axes, and filters.
            3. **Interpret the Results:** Explore the charts and residual analysis to identify potential investment opportunities.
        """)

        # Place the "Proceed to Dashboard" button below the text
        if st.button("Proceed to Dashboard", key="proceed_button_unique"):
            st.session_state['intro_done'] = True
            # Streamlit automatically reruns the script on interaction
    except FileNotFoundError:
        st.warning("Logo image not found. Please ensure the logo is in the app directory.")


def get_logo_path():
    """
    Determine the logo path based on the operating system.
    """
    return os.path.join(os.getcwd(), "logo.png")

# =====================
# 6. Main Application
# =====================
def main_app(df, actual_column, selected_horizontal_axis, filtered_df):
    # Implement the "Main Chart" section only

    if filtered_df.empty:
        st.warning("No bonds match the selected filters.")
        return

    # =====================
    # Residuals Calculation using NSS Regression
    # =====================
    independent_var = selected_horizontal_axis

    if len(filtered_df) < 6:
        st.warning("Not enough data points to perform NSS regression. At least 6 points are required.")
    else:
        t = filtered_df[independent_var].values
        y = filtered_df[actual_column].values

        # Data Validation: Ensure no zero or negative Duration
        if np.any(t <= 0):
            st.error(f"{independent_var} must be positive for NSS regression.")
        else:
            # Perform NSS regression
            y_pred, params = perform_nss_regression(t, y)

            if y_pred is not None:
                # Add regression predictions to the DataFrame
                filtered_df = filtered_df.copy()
                filtered_df['Regression_Predicted'] = y_pred

                # Calculate residuals
                filtered_df['Residual'] = filtered_df[actual_column] - filtered_df['Regression_Predicted']

                # Split data into overvalued and undervalued based on residuals
                overvalued_df = filtered_df[filtered_df['Residual'] < 0]  # Negative Residuals: Overvalued
                undervalued_df = filtered_df[filtered_df['Residual'] > 0]  # Positive Residuals: Undervalued

                # =====================
                # Plotting
                # =====================
                fig = go.Figure()

                # Add regression curve
                sorted_df = filtered_df.sort_values(by=independent_var)
                fig.add_trace(
                    go.Scatter(
                        x=sorted_df[independent_var],
                        y=sorted_df['Regression_Predicted'],
                        mode='lines',
                        name='Regression Curve',
                        line=dict(color='darkblue', width=4)  # Increased width and dark blue
                    )
                )

                # Define y_format_str based on the selected vertical axis
                if actual_column == 'Yield':
                    y_format_str = "<b>{}</b>: %{{y:.2f}}%<br>".format(actual_column)
                else:
                    y_format_str = "<b>{}</b>: %{{y:.0f}} bps<br>".format(actual_column)

                # Define rating information if available
                if 'Rating' in df.columns:
                    rating_info = "<b>Rating:</b> %{{customdata[1]}}<br>"
                else:
                    rating_info = ""

                # Construct the hovertemplate using str.format() and properly escaped placeholders
                hovertemplate = (
                    "<b>ISIN:</b> %{{customdata[0]}}<br>"
                    "{}"
                    "<b>{}</b>: %{{x:.2f}} yrs<br>"
                    "{}<extra></extra>"
                ).format(
                    y_format_str,
                    selected_horizontal_axis,
                    rating_info
                )

                # Add Bonds Potentially Undervalued (Green)
                fig.add_trace(
                    go.Scatter(
                        x=undervalued_df[independent_var],
                        y=undervalued_df[actual_column],
                        mode='markers',
                        name='Potentially Undervalued Bonds',
                        marker=dict(
                            color='green',
                            size=12,
                            symbol='circle-open',
                            line=dict(width=2, color='green')
                        ),
                        hovertemplate=hovertemplate,
                        customdata=undervalued_df[['ISIN', 'Rating']].values if 'Rating' in df.columns else undervalued_df[['ISIN']].values
                    )
                )

                # Add Bonds Potentially Overvalued (Red)
                fig.add_trace(
                    go.Scatter(
                        x=overvalued_df[independent_var],
                        y=overvalued_df[actual_column],
                        mode='markers',
                        name='Potentially Overvalued Bonds',
                        marker=dict(
                            color='red',
                            size=12,
                            symbol='circle-open',
                            line=dict(width=2, color='red')
                        ),
                        hovertemplate=hovertemplate,
                        customdata=overvalued_df[['ISIN', 'Rating']].values if 'Rating' in df.columns else overvalued_df[['ISIN']].values
                    )
                )

                # Update layout with corrected Y-axis formatting and larger fonts
                if actual_column == 'Yield':
                    fig.update_yaxes(tickformat=".2f")  # Display as is, e.g., 3.00%
                    y_axis_title = f"<b>{actual_column} (%)</b>"
                else:
                    fig.update_yaxes(tickformat=",.0f")  # For spreads, display without decimals
                    y_axis_title = f"<b>{actual_column} (bps)</b>"

                fig.update_layout(
                    height=600,  # Increase the height of the chart
                    title={
                        'text': "Issuer Curve",
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                    },
                    title_font_size=24,  # Increased font size
                    xaxis_title=f"<b>{selected_horizontal_axis} (Years)</b>",
                    yaxis_title=y_axis_title,
                    xaxis=dict(
                        title_font=dict(size=18),
                        tickfont=dict(size=14)
                    ),
                    yaxis=dict(
                        title_font=dict(size=18),
                        tickfont=dict(size=14)
                    ),
                    hovermode="closest",
                    template="plotly_dark",
                    legend=dict(
                        x=0,       # Top-left corner
                        y=1,       # Top position
                        bgcolor='white',  # White background for contrast
                        bordercolor='grey',  # Grey border
                        borderwidth=1
                    )
                )

                # Enhance hover effects
                fig.update_traces(marker=dict(line=dict(width=2)))

                # Display the plot
                st.plotly_chart(fig, use_container_width=True)

                # Add explanation below the chart
                st.markdown("""
                **Explanation:**

                - The curve is constructed using the Nelson-Siegel-Svensson (NSS) regression model.
                - Bonds plotted **below** the regression curve are **overvalued**, offering lower yields than expected.
                - Bonds plotted **above** the regression curve are **undervalued**, offering higher yields than expected.
                - **Hover over the points** to see detailed information about each bond.
                """)

                # =====================
                # Enhanced Bond Data Table with Conditional Coloring
                # =====================
                st.subheader("Bond Data")

                # Define a function to color rows based on residuals
                def highlight_residual(row):
                    if row['Residual'] > 0:
                        color = 'background-color: #d4edda'  # Light green for Undervalued
                    elif row['Residual'] < 0:
                        color = 'background-color: #f8d7da'  # Light red for Overvalued
                    else:
                        color = ''
                    return [color] * len(row)

                # Prepare the DataFrame for display
                display_columns = ['ISIN', actual_column, selected_horizontal_axis]
                if 'Rating' in df.columns:
                    display_columns.append('Rating')
                display_columns.append('Residual')

                styled_df = filtered_df.copy()
                # Round all numerical columns to two decimals
                numeric_cols = ['Yield', 'G-Spread', 'Z-Spread', 'OAS', 'Duration', 'Time To Maturity', 'Residual']
                for col in numeric_cols:
                    if col in styled_df.columns:
                        styled_df[col] = styled_df[col].round(2)

                # Apply conditional coloring to the DataFrame
                format_dict = {col: "{:.2f}" for col in numeric_cols if col in display_columns}
                st.dataframe(
                    styled_df[display_columns].style.apply(highlight_residual, axis=1).format(format_dict),
                    height=400
                )
            else:
                st.error("Regression did not produce any output. Please check your data and try again.")

# =====================
# 7. Residual Analysis Function
# =====================
def residual_analysis(df, actual_column, selected_horizontal_axis, filtered_df, std_threshold, analysis_type, vertical_axis_options):
    """
    Perform Residual Analysis.

    Parameters:
    - df: Original DataFrame
    - actual_column: Selected vertical axis column
    - selected_horizontal_axis: Selected horizontal axis column
    - filtered_df: Filtered DataFrame based on user selections
    - std_threshold: Number of standard deviations for classification
    - analysis_type: "Single Metric" or "Aggregate Metrics"
    - vertical_axis_options: Dictionary mapping vertical axis options
    """
    if filtered_df.empty:
        st.warning("No bonds match the selected filters.")
        return

    # Define metrics to analyze
    if analysis_type == "Single Metric":
        metrics_to_analyze = [actual_column]
    else:
        # Use all available metrics
        metrics_to_analyze = ['Yield', 'G-Spread', 'Z-Spread', 'OAS']
        # Filter out metrics not present in the DataFrame
        metrics_to_analyze = [metric for metric in metrics_to_analyze if metric in df.columns]

    residuals_df = pd.DataFrame(index=filtered_df.index)
    residuals_df['ISIN'] = filtered_df['ISIN']

    for metric in metrics_to_analyze:
        t = filtered_df[selected_horizontal_axis].values
        y = filtered_df[metric].values
        y_pred, params = perform_nss_regression(t, y)
        if y_pred is not None:
            residuals = y - y_pred
            # Standardize residuals
            standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
            residuals_df[metric + '_Residual'] = standardized_residuals
            # Store 'Regression_Predicted' values
            filtered_df[metric + '_Regression_Predicted'] = y_pred
        else:
            st.error(f"Regression failed for {metric}. Skipping this metric.")
            residuals_df[metric + '_Residual'] = np.nan

    if analysis_type == "Aggregate Metrics":
        # Calculate the aggregated residual
        residual_columns = [col for col in residuals_df.columns if '_Residual' in col]
        residuals_df['Aggregated_Residual'] = residuals_df[residual_columns].mean(axis=1)
        residual_column_name = 'Aggregated_Residual'
        filtered_df['Residual'] = residuals_df['Aggregated_Residual']
    else:
        residual_column_name = metrics_to_analyze[0] + '_Residual'
        filtered_df['Residual'] = residuals_df[metrics_to_analyze[0] + '_Residual']

    # Ensure residuals are valid
    residuals = filtered_df['Residual'].dropna()

    # Calculate thresholds
    residual_std = residuals.std()
    upper_threshold = std_threshold * residual_std
    lower_threshold = -std_threshold * residual_std

    # Classify bonds
    undervalued_bonds = filtered_df[filtered_df['Residual'] > upper_threshold]
    overvalued_bonds = filtered_df[filtered_df['Residual'] < lower_threshold]
    fairly_valued_bonds = filtered_df[
        (filtered_df['Residual'] <= upper_threshold) & (filtered_df['Residual'] >= lower_threshold)
    ]

    # =====================
    # Residual Analysis Sections
    # =====================
    with st.expander("1. Residuals by Bond (ISIN)", expanded=True):
        # Residuals by Bond (ISIN)
        # Sort the dataframe by Duration
        sorted_residuals_df = filtered_df.sort_values('Duration')

        # Determine colors based on residuals
        colors = ['green' if x > upper_threshold else 'red' if x < lower_threshold else 'gray' for x in sorted_residuals_df['Residual']]

        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=sorted_residuals_df['ISIN'],
                y=sorted_residuals_df['Residual'],
                marker_color=colors,
                hovertemplate=(
                    "<b>ISIN:</b> %{x}<br>"
                    "<b>Residual:</b> %{y:.2f}<br>"
                    "<extra></extra>"
                )
            )
        )
        fig_bar.update_layout(
            height=500,  # Adjust height if needed
            title="Residuals by Bond (ISIN)",
            xaxis_title="ISIN",
            yaxis_title="Standardized Residuals",
            template="plotly_dark",
            legend=dict(
                x=0.85,
                y=0.95,
                bgcolor='white',  # Changed to white
                bordercolor='grey',  # Grey border
                borderwidth=1
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Explanation
        st.markdown("""
        **Interpretation:**
        - This chart displays the standardized residuals for each bond identified by ISIN, sorted by Duration.
        - **Green bars** indicate bonds that are **undervalued** (positive residuals beyond the threshold).
        - **Red bars** indicate bonds that are **overvalued** (negative residuals beyond the threshold).
        - **Gray bars** indicate bonds that are **fairly valued** within the threshold.
        - **Hover over the bars** to see detailed information about each bond.
        """)

    with st.expander("2. Statistical Summary", expanded=False):
        # Statistical Summary
        st.subheader("Statistical Summary of Residuals")
        st.write(residuals.describe())

        # Calculate skewness and kurtosis
        res_skewness = skew(residuals)
        res_kurtosis = kurtosis(residuals)

        # Shapiro-Wilk Test
        stat_sw, p_value_sw = shapiro(residuals)
        if p_value_sw > 0.05:
            normality_interpretation_sw = "The residuals **appear** to be normally distributed (Shapiro-Wilk Test p-value > 0.05)."
        else:
            normality_interpretation_sw = "The residuals **do not** appear to be normally distributed (Shapiro-Wilk Test p-value ≤ 0.05)."

        # Anderson-Darling Test
        result_ad = stats.anderson(residuals)
        if result_ad.statistic > result_ad.critical_values[2]:  # 5% significance
            ad_interpretation = "The residuals **do not** appear to be normally distributed according to the Anderson-Darling test."
        else:
            ad_interpretation = "The residuals **appear** to be normally distributed according to the Anderson-Darling test."

        # Durbin-Watson Test
        dw_stat = durbin_watson(residuals)
        # Interpretation based on dw_stat
        if dw_stat < 1.5:
            dw_interpretation = "There is **strong evidence of positive autocorrelation** (Durbin-Watson Test)."
        elif 1.5 <= dw_stat <= 2.5:
            dw_interpretation = "There is **no evidence of autocorrelation** (Durbin-Watson Test)."
        else:
            dw_interpretation = "There is **strong evidence of negative autocorrelation** (Durbin-Watson Test)."

        # Display statistical results
        st.markdown(f"""
        - **Skewness:** {res_skewness:.4f}
        - **Kurtosis:** {res_kurtosis:.4f}
        - **Shapiro-Wilk Test p-value:** {p_value_sw:.4f}
        - **Anderson-Darling Test Statistic:** {result_ad.statistic:.4f}
        - **Durbin-Watson Test Statistic:** {dw_stat:.4f}

        **Interpretations:**
        - **Normality Tests:**
          - {normality_interpretation_sw}
          - {ad_interpretation}
        - **Autocorrelation Test:**
          - {dw_interpretation}
        """)

        # Provide actionable insights based on test results
        st.markdown("""
        **Actionable Insights:**
        
        - **If Residuals Are Not Normally Distributed:**
          - *Implication:* The predictive model may not be capturing all underlying patterns in the yield curve.
          - *Action:* Consider refining the model by including additional variables or exploring alternative modeling techniques to better fit the yield curve data.
        
        - **If Autocorrelation Is Detected:**
          - *Implication:* There may be patterns or trends over time in the yield curve that the model is not accounting for.
          - *Action:* Incorporate time-dependent variables or explore time series modeling approaches to better capture these dynamics.
        """)

    with st.expander("3. Histogram and Density Plot of Residuals", expanded=False):
        # Histogram and Density Plot
        st.subheader("Histogram and Density Plot of Residuals")

        # Compute KDE
        density = gaussian_kde(residuals)
        xs = np.linspace(residuals.min(), residuals.max(), 200)
        density_values = density(xs)

        fig_hist = go.Figure()

        # Histogram of residuals
        fig_hist.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=20,
                histnorm='probability density',
                marker_color='lightblue',
                opacity=0.6,
                name='Histogram',
                hovertemplate="<b>Residual:</b> %{x:.2f}<br><b>Count:</b> %{y}<extra></extra>"
            )
        )

        # KDE line
        fig_hist.add_trace(
            go.Scatter(
                x=xs,
                y=density_values,
                mode='lines',
                line=dict(color='red', width=2),
                name='Density',
                hovertemplate="<b>Density:</b> %{y:.4f}<extra></extra>"
            )
        )

        fig_hist.update_layout(
            title="Residuals Distribution with Density Curve",
            xaxis_title="Standardized Residuals",
            yaxis_title="Density",
            template="plotly_dark",
            legend=dict(
                x=0.85,
                y=0.95,
                bgcolor='white',  # Changed to white
                bordercolor='grey',  # Grey border
                borderwidth=1
            )
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Conclusion for the Distribution
        st.subheader("Conclusion on the Distribution of Residuals")

        # Interpretation based on skewness and kurtosis
        if abs(res_skewness) < 0.5:
            skewness_interpretation = "Residuals are approximately symmetric, indicating a balanced distribution of overvalued and undervalued bonds."
        elif res_skewness > 0:
            skewness_interpretation = "Residuals are positively skewed, suggesting more bonds are undervalued relative to the regression curve."
        else:
            skewness_interpretation = "Residuals are negatively skewed, indicating more bonds are overvalued relative to the regression curve."

        if res_kurtosis < 3:
            kurtosis_interpretation = "Residuals have light tails (platykurtic), implying fewer extreme values than a normal distribution."
        elif res_kurtosis > 3:
            kurtosis_interpretation = "Residuals have heavy tails (leptokurtic), indicating more extreme values and potential outliers."
        else:
            kurtosis_interpretation = "Residuals have a normal tail distribution (mesokurtic)."

        st.markdown(f"""
        Based on the histogram and density plot:
        - **{normality_interpretation_sw}**
        - **Skewness ({res_skewness:.4f}):** {skewness_interpretation}
        - **Kurtosis ({res_kurtosis:.4f}):** {kurtosis_interpretation}
        - **Anderson-Darling Test:** {ad_interpretation}
        """)

    with st.expander("4. Q-Q Plot of Residuals", expanded=False):
        # Q-Q Plot
        st.subheader("Q-Q Plot of Residuals")

        if len(residuals) < 5:
            st.warning("Not enough data points to generate a meaningful Q-Q plot.")
        else:
            osm, osr = stats.probplot(residuals, dist="norm")
            theoretical_quantiles = np.array(osm[0])
            ordered_residuals = np.array(osm[1])

            min_val = min(theoretical_quantiles.min(), ordered_residuals.min())
            max_val = max(theoretical_quantiles.max(), ordered_residuals.max())

            fig_qq = go.Figure()
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=ordered_residuals,
                    mode='markers',
                    marker=dict(color='purple', size=8),
                    name='Residuals',
                    hovertemplate="<b>Theoretical Quantile:</b> %{x:.2f}<br><b>Residual:</b> %{y:.2f}<extra></extra>"
                )
            )
            fig_qq.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='orange', dash='dash'),
                    name='y = x',
                    hovertemplate="<b>Reference Line:</b> y = x<extra></extra>"
                )
            )
            fig_qq.update_layout(
                title="Q-Q Plot",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Ordered Residuals",
                template="plotly_dark",
                legend=dict(
                    x=0.85,
                    y=0.95,
                    bgcolor='white',  # Changed to white
                    bordercolor='grey',  # Grey border
                    borderwidth=1
                )
            )
            st.plotly_chart(fig_qq, use_container_width=True)

            # Interpretation of Q-Q Plot
            st.markdown("""
            **Interpretation of Q-Q Plot:**
            - The Q-Q plot compares the quantiles of the residuals with the theoretical quantiles of a normal distribution.
            - **Points lying along the line** indicate that the residuals follow a normal distribution.
            - **Deviations from the line**, especially at the ends, suggest departures from normality.
            """)

    with st.expander("5. Residuals vs. Fitted Values", expanded=False):
        if analysis_type == "Single Metric":
            st.subheader("Residuals vs. Fitted Values")
            regression_predicted_column = metrics_to_analyze[0] + '_Regression_Predicted'
            x_values = filtered_df[regression_predicted_column]
            y_values = residuals
            # Plot the residuals vs. fitted values
            fig_res_fit = go.Figure()
            fig_res_fit.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    marker=dict(color='orange', size=8),
                    name='Residuals',
                    hovertemplate="<b>Fitted Value:</b> %{x:.2f}<br><b>Residual:</b> %{y:.2f}<extra></extra>"
                )
            )
            # Update layout
            fig_res_fit.update_layout(
                title="Residuals vs. Fitted Values",
                xaxis_title="Fitted Values",
                yaxis_title="Standardized Residuals",
                template="plotly_dark",
                legend=dict(
                    x=0.85,
                    y=0.95,
                    bgcolor='white',  # Changed to white
                    bordercolor='grey',  # Grey border
                    borderwidth=1
                )
            )
            st.plotly_chart(fig_res_fit, use_container_width=True)

            # Explanation for Residuals vs. Fitted Plot
            st.markdown("""
            **Interpretation:**
            - This plot checks for non-linearity, unequal error variances, and outliers.
            - **Random scatter** suggests a good fit.
            - **Patterns** (e.g., curves or funnels) indicate potential problems with the model.
            """)
        else:
            st.info("Residuals vs. Fitted Values plot is not available for Aggregate Metrics.")

    with st.expander("6. Comprehensive Conclusion", expanded=True):
        # Comprehensive Conclusion
        st.subheader("Comprehensive Conclusion")

        # Count of overvalued and undervalued bonds
        num_undervalued = len(undervalued_bonds)
        num_overvalued = len(overvalued_bonds)
        num_fairly_valued = len(fairly_valued_bonds)

        st.markdown(f"""
        Based on the residual analysis:

        - **Normality of Residuals:**
          - Skewness: {res_skewness:.4f} ({skewness_interpretation})
          - Kurtosis: {res_kurtosis:.4f} ({kurtosis_interpretation})
          - Shapiro-Wilk Test p-value: {p_value_sw:.4f} ({normality_interpretation_sw})
          - Anderson-Darling Test Statistic: {result_ad.statistic:.4f} ({ad_interpretation})
          - Durbin-Watson Test Statistic: {dw_stat:.4f} ({dw_interpretation})

        - **Potential Investment Opportunities:**
          - Number of potentially **undervalued bonds**: **{num_undervalued}**
          - Number of potentially **overvalued bonds**: **{num_overvalued}**
          - Number of **fairly valued bonds**: **{num_fairly_valued}**

        **Recommendations:**

        - **Undervalued Bonds:** Bonds with significant positive residuals (undervalued) may present **investment opportunities**. Consider these for potential inclusion in portfolios to achieve higher yields relative to the market benchmark.
            
        - **Overvalued Bonds:** Bonds with significant negative residuals (overvalued) may warrant **caution** or **further analysis**. These could indicate higher prices and lower yields, potentially offering less attractive returns.
            
        - **Fairly Valued Bonds:** Bonds within the residual threshold are considered **fairly valued** and align well with the market expectations set by the regression model.

        **Model Reliability Insights:**

        - **Normality:** {normality_interpretation_sw} {ad_interpretation}
            
        - **Autocorrelation:** {dw_interpretation}

        **Actionable Steps:**
        
        - If **residuals are not normally distributed** or if **autocorrelation is detected**, the classifications of overvalued or undervalued bonds **may be questionable**.
            - *Implication:* The regression model might not be fully capturing the complexities of the yield curve.
            - *Action:* Consider refining the model by incorporating additional variables, exploring alternative modeling techniques, or performing further analysis to ensure accurate bond valuation.
        
        - **Regular Model Review:** Periodically review and update the regression model with new data to maintain its accuracy and relevance in changing market conditions.

        **Note:** This analysis is based on statistical models and should be complemented with **fundamental analysis** and **market considerations** to make well-informed investment decisions.
        """)

        # Display tables for undervalued and overvalued bonds
        st.subheader("Potentially Undervalued Bonds")
        undervalued_bonds_display = undervalued_bonds.copy()
        if not undervalued_bonds_display.empty:
            display_columns = ['ISIN', actual_column, selected_horizontal_axis]
            if 'Rating' in df.columns:
                display_columns.append('Rating')
            display_columns.append('Residual')

            # Identify numeric columns in display_columns
            numeric_columns = ['Yield', 'G-Spread', 'Z-Spread', 'OAS', 'Duration', 'Time To Maturity', 'Residual']
            numeric_display_columns = [col for col in display_columns if col in numeric_columns]

            # Create format dictionary for numeric columns
            format_dict = {col: "{:.2f}" for col in numeric_display_columns}

            st.dataframe(
                undervalued_bonds_display[display_columns].style.format(format_dict),
                height=400
            )
        else:
            st.write("No potentially undervalued bonds identified.")

        st.subheader("Potentially Overvalued Bonds")
        overvalued_bonds_display = overvalued_bonds.copy()
        if not overvalued_bonds_display.empty:
            display_columns = ['ISIN', actual_column, selected_horizontal_axis]
            if 'Rating' in df.columns:
                display_columns.append('Rating')
            display_columns.append('Residual')

            # Identify numeric columns in display_columns
            numeric_columns = ['Yield', 'G-Spread', 'Z-Spread', 'OAS', 'Duration', 'Time To Maturity', 'Residual']
            numeric_display_columns = [col for col in display_columns if col in numeric_columns]

            # Create format dictionary for numeric columns
            format_dict = {col: "{:.2f}" for col in numeric_display_columns}

            st.dataframe(
                overvalued_bonds_display[display_columns].style.format(format_dict),
                height=400
            )
        else:
            st.write("No potentially overvalued bonds identified.")

        st.subheader("Fairly Valued Bonds")
        fairly_valued_bonds_display = fairly_valued_bonds.copy()
        if not fairly_valued_bonds_display.empty:
            display_columns = ['ISIN', actual_column, selected_horizontal_axis]
            if 'Rating' in df.columns:
                display_columns.append('Rating')
            display_columns.append('Residual')

            # Identify numeric columns in display_columns
            numeric_columns = ['Yield', 'G-Spread', 'Z-Spread', 'OAS', 'Duration', 'Time To Maturity', 'Residual']
            numeric_display_columns = [col for col in display_columns if col in numeric_columns]

            # Create format dictionary for numeric columns
            format_dict = {col: "{:.2f}" for col in numeric_display_columns}

            st.dataframe(
                fairly_valued_bonds_display[display_columns].style.format(format_dict),
                height=400
            )
        else:
            st.write("No fairly valued bonds identified.")

# =====================
# 8. Run the Application
# =====================
def run():
    # Inject custom CSS for hover effects
    inject_custom_css()

    # Initialize session state for 'intro_done' if not already set
    if 'intro_done' not in st.session_state:
        st.session_state['intro_done'] = False

    if not st.session_state['intro_done']:
        # Display introduction
        introduction()
    else:
        # Sidebar - File Uploader, Logo, and Navigation with Collapsible Sections
        logo_path = get_logo_path()
        try:
            st.sidebar.image(logo_path, width=100)
        except FileNotFoundError:
            st.sidebar.warning("Logo image not found. Please ensure the logo is in the app directory.")

        st.sidebar.header("Upload Your Bond Data")
        uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"], key="file_uploader_unique")

        # Navigation Tabs with unique key
        section = st.sidebar.radio("Navigate to", ["Main Chart", "Residual Analysis"], key="main_nav_radio_sidebar_unique")

        # Initialize selected_ratings
        selected_ratings = None

        # Load data only if a file is uploaded
        if uploaded_file is not None:
            df, required_columns = load_data(uploaded_file)
            if df is not None:
                # Collapsible Customize Axes Section (Moved Above Filters)
                with st.sidebar.expander("🎨 Customize Axes", expanded=True):
                    vertical_axis_options = {
                        'Yield': 'Yield',
                        'G-Spread': 'G-Spread',
                        'Z-Spread': 'Z-Spread',
                        'OAS': 'OAS'
                    }
                    horizontal_axis_options = {
                        'Duration': 'Duration',
                        'Time To Maturity': 'Time To Maturity'
                    }

                    selected_vertical_axis = st.selectbox(
                        "Select Vertical Axis",
                        options=list(vertical_axis_options.keys()),
                        index=0,  # Default to 'Yield'
                        key="vertical_axis_select_unique"
                    )
                    selected_horizontal_axis = st.selectbox(
                        "Select Horizontal Axis",
                        options=list(horizontal_axis_options.keys()),
                        index=1,  # Default to 'Time To Maturity'
                        key="horizontal_axis_select_unique"
                    )

                # Define actual_column so it's available throughout the main() function
                actual_column = vertical_axis_options[selected_vertical_axis]

                # Collapsible Filters Section
                with st.sidebar.expander("🔍 Filters", expanded=True):
                    if 'Rating' in df.columns:
                        unique_ratings = df['Rating'].unique()
                        selected_ratings = st.multiselect(
                            "Select Ratings",
                            options=unique_ratings,
                            default=unique_ratings,
                            key="rating_multiselect_unique"
                        )
                    else:
                        selected_ratings = None  # No rating filter if not present

                    min_duration = float(df['Duration'].min())
                    max_duration = float(df['Duration'].max())
                    selected_duration = st.slider(
                        "Select Duration Range (Years)",
                        min_duration,
                        max_duration,
                        (min_duration, max_duration),
                        key="duration_slider_unique"
                    )

                    min_maturity = float(df['Time To Maturity'].min())
                    max_maturity = float(df['Time To Maturity'].max())
                    selected_maturity = st.slider(
                        "Select Time To Maturity Range (Years)",
                        min_maturity,
                        max_maturity,
                        (min_maturity, max_maturity),
                        key="maturity_slider_unique"
                    )

                    min_yield = float(df['Yield'].min())
                    max_yield = float(df['Yield'].max())
                    selected_yield = st.slider(
                        "Select Yield Range (%)",
                        min_yield,
                        max_yield,
                        (min_yield, max_yield),
                        key="yield_slider_unique"
                    )

                    min_gspread = float(df['G-Spread'].min())
                    max_gspread = float(df['G-Spread'].max())
                    selected_gspread = st.slider(
                        "Select G-Spread Range (bps)",
                        min_gspread,
                        max_gspread,
                        (min_gspread, max_gspread),
                        key="gspread_slider_unique"
                    )

                    min_zspread = float(df['Z-Spread'].min())
                    max_zspread = float(df['Z-Spread'].max())
                    selected_zspread = st.slider(
                        "Select Z-Spread Range (bps)",
                        min_zspread,
                        max_zspread,
                        (min_zspread, max_zspread),
                        key="zspread_slider_unique"
                    )

                    min_oas = float(df['OAS'].min())
                    max_oas = float(df['OAS'].max())
                    selected_oas = st.slider(
                        "Select OAS Range (bps)",
                        min_oas,
                        max_oas,
                        (min_oas, max_oas),
                        key="oas_slider_unique"
                    )

                # Apply Filters
                filtered_df = df[
                    (df['Duration'] >= selected_duration[0]) &
                    (df['Duration'] <= selected_duration[1]) &
                    (df['Time To Maturity'] >= selected_maturity[0]) &
                    (df['Time To Maturity'] <= selected_maturity[1]) &
                    (df['Yield'] >= selected_yield[0]) &
                    (df['Yield'] <= selected_yield[1]) &
                    (df['G-Spread'] >= selected_gspread[0]) &
                    (df['G-Spread'] <= selected_gspread[1]) &
                    (df['Z-Spread'] >= selected_zspread[0]) &
                    (df['Z-Spread'] <= selected_zspread[1]) &
                    (df['OAS'] >= selected_oas[0]) &
                    (df['OAS'] <= selected_oas[1])
                ]

                if 'Rating' in df.columns and selected_ratings:
                    filtered_df = filtered_df[filtered_df['Rating'].isin(selected_ratings)]

                # =====================
                # Section Handling
                # =====================
                if section == "Main Chart":
                    main_app(df, actual_column, selected_horizontal_axis, filtered_df)
                elif section == "Residual Analysis":
                    st.header("Residual Analysis")
                    # Confirm Axis Selection
                    st.markdown("""
                        **Residual Analysis will be based on your current axis selections.**
                    """)

                    # Analysis Type Selection with unique key
                    analysis_type = st.radio(
                        "Select Analysis Type",
                        ("Single Metric", "Aggregate Metrics"),
                        key="analysis_type_radio_unique"
                    )

                    # Define standard deviation threshold with unique key
                    st.subheader("Define Over/Undervalued Threshold")
                    std_threshold = st.number_input(
                        "Number of Standard Deviations for Classification:",
                        min_value=0.0,
                        max_value=5.0,
                        value=1.0,
                        step=0.1,
                        key="std_threshold_input_unique"
                    )

                    residual_analysis(df, actual_column, selected_horizontal_axis, filtered_df, std_threshold, analysis_type, vertical_axis_options)
        else:
            st.info("Please upload an Excel file to proceed.")

# =====================
# 9. Run the Application
# =====================
if __name__ == "__main__":
    run()
