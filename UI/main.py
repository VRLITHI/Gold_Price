import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pandas import DataFrame
import itertools

from plotly.subplots import make_subplots
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10


# Analysis imports
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from prophet import Prophet

import warnings
warnings.filterwarnings('ignore') #this would remove any deprecated warning
import streamlit as st


st.title("Gold Price Forcasting")
data_gld = pd.read_csv('GLD.csv')
print("GLD Data: " + str(data_gld.shape))
st.markdown("Data Set form Kaggle")
st.write(data_gld.head(20))

st.markdown("Below is the brief description of the Data in the dataset")
st.write(data_gld.describe())

dateparser = lambda dates: pd.to_datetime(dates, format='%Y-%m-%d')
ffill_data = pd.read_csv('GLD.csv', parse_dates=['Date'], date_parser=dateparser)
print("GLD Data Before Forward Fill: " + str(ffill_data.shape))


cols = ffill_data.columns #store original column names
ffill_data.set_index("Date", inplace=True)
ffill_data = ffill_data.resample("D").ffill().reset_index()
# Sort by date in ascending order & export to new csv file
ffill_data = ffill_data[cols] #revert column order
ffill_data.sort_values(by='Date', inplace=True)
ffill_data.to_csv('gld-nomissing.csv', index=False)

dateparser = lambda dates: pd.to_datetime(dates,format='%Y-%m-%d')
data = pd.read_csv('gld-nomissing.csv', index_col='Date', parse_dates=['Date'], date_parser=dateparser)

sns.set(style="darkgrid")

# Create a Streamlit app
st.title('GLD Daily Opening Price')

# Create a figure using Matplotlib
fig, ax = plt.subplots(figsize=(16, 8))

# Plot the data
sns.lineplot(x=data.index, y='Open', data=data, linewidth=1.5, ax=ax)

# Display the plot using Streamlit
st.pyplot(fig)


data_diff = data.diff().dropna()
diff_adfuller_result = adfuller(data_diff['Open'])

data_sqrt = np.sqrt(data).dropna()
sqrt_adfuller_result = adfuller(data_sqrt['Open'])

data_difftwice = data.diff().diff().dropna()
difftwice_adfuller_result = adfuller(data_difftwice['Open'])

sns.set(style="darkgrid")

# Create a Streamlit app
st.header('Time Series Visualization')
st.markdown("The below graphs are the comparison before and after the stationary transformation ")

# Plot the time series before transformation
fig1, ax1 = plt.subplots(figsize=(16, 8))
sns.lineplot(x=data.index, y='Open', data=data, linewidth=1.5, label='Before transformation', ax=ax1)
ax1.set_title('Time Series - Before Transformation')
st.pyplot(fig1)

# Plot the time series after transformation
fig2, ax2 = plt.subplots(figsize=(16, 8))
sns.lineplot(x=data_difftwice.index, y='Open', data=data_difftwice, label='After transformation', color='green', ax=ax2)
ax2.set_title('Stationary Time Series - Diff Twice Transformation')
st.pyplot(fig2)

st.subheader('Autocorrelation and Partial Autocorrelation Plots')
st.markdown("ACF (Autocorrelation Function): It is a measure of the correlation between a time series and "
                "a lagged version of itself. It helps in selecting the right order for time series analysis.")

st.markdown("PACF (Partial Autocorrelation Function): It is the correlation between the time series and the lag version "
                "of itself after subtracting the effect of correlation at smaller lags. It is associated with just that particular lag.")

# Create a Matplotlib figure with subplots
fig_data_gld, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot ACF of data_gld
plot_acf(data_gld['Open'], lags=10, zero=False, ax=ax1, title='Autocorrelation Non-Stationary Data')

# Plot PACF of data_gld
plot_pacf(data_gld['Open'], lags=10, zero=False, ax=ax2, title='Partial Autocorrelation Non-Stationary Data')

# Display the plots using Streamlit
st.pyplot(fig_data_gld)


#------------------------------------------------------------

order_aic_bic = []

# Loop over AR order
for p in range(3):
    # Loop over MA order
    for q in range(3):
        try:
            # Fit model
            model = ARIMA(data_difftwice['Open'], order=(p, 0, q))
            results = model.fit()
            # Store the model order and the AIC/BIC values in order_aic_bic list
            order_aic_bic.append((p, q, results.aic, results.bic))
        except:
            # Print AIC and BIC as None when fails
            print(p, q, None, None)

# Create a DataFrame from the results

#------------------Traintestsplit

train_data = data.loc[:'2016']
test_data = data.loc['2017':]

sns.set(style="darkgrid")
st.subheader('The graph after train and test data')
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x=train_data.index, y='Open', data=train_data, linewidth=1.5, label='Training data', ax=ax)
sns.lineplot(x=test_data.index, y='Open', data=test_data, linewidth=1.5, label='Test data', ax=ax)
st.pyplot(fig)

#--------------------------------------------------------------
#DROP DOWN BOX

options = ["Please select the model","AR Model", "ARMA Model", "ARIMA Model"]
selected = st.selectbox('Select an option:', options)


if selected == "ARIMA Model":
    model = ARIMA(train_data['Open'], order=(0, 2, 1), trend=None)
    results = model.fit()

    pred_365_traindata = results.get_prediction(start=-365, dynamic=False)
    pred_mean_365_traindata = pred_365_traindata.predicted_mean
    confidence_intervals = pred_365_traindata.conf_int()
    lower_limits = confidence_intervals.loc[:, 'lower Open']
    upper_limits = confidence_intervals.loc[:, 'upper Open']

    pred_mean_365_traindata_df = pred_mean_365_traindata.to_frame(name='forecasted_mean')

    sns.set(style="darkgrid")
    st.header("ARIMA Model - Autoregressive Integrated Moving Average")
    st.markdown("model = ARIMA(df, order = (p,d,q))")

    st.markdown("Where p = number of autoregressive lags")
    st.markdown("      d = order of differencing")
    st.markdown("      q = number of moving average lags")

    st.subheader('One-step Ahead Forecast')
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(x=train_data['2014-01-01 00:00:00':].index, y='Open', data=train_data['2014-01-01 00:00:00':],
                 linewidth=4, label='observed', ax=ax)
    sns.lineplot(x=pred_mean_365_traindata_df.index, y=pred_mean_365_traindata_df['forecasted_mean'],
                 data=pred_mean_365_traindata_df, linewidth=1, label='forecast for 365 days', color='red', ax=ax)
    ax.fill_between(lower_limits.index, lower_limits, upper_limits, color='pink')
    st.pyplot(fig)




#-------------------------------------------------------------------------------------
    arima_data = data.drop(columns=['High', 'Low', 'Close', 'Adj Close', 'Volume'])
    results = auto_arima(arima_data,
                         seasonal=False,
                         start_p=1,
                         start_q=1,
                         max_p=4,
                         max_q=4,
                         information_criterion='aic',
                         trace=True,
                         stepwise=True)
    arima_train_data = arima_data.loc[:'2016']
    arima_test_data = arima_data.loc['2017':]

    # Create an ARIMA model
    arima_model = SARIMAX(arima_train_data,
                          order=(0, 1, 0),  # Adjust non-seasonal order (p, d, q) as needed
                          trend='c')

    # Fit the ARIMA model
    arima_results = arima_model.fit()

    # Make predictions for the last 365 days of the train data
    # dynamic=False ensures we produce one-step ahead forecasts, forecasts at each point are generated using the full history up to that point
    # start=-365, we want to start the prediction from one year back (365 days)
    arima_pred_365_traindata = arima_results.get_prediction(start=-365, dynamic=False)

    # Forecast mean for 365 days
    arima_pred_mean_365_traindata = arima_pred_365_traindata.predicted_mean

    # Get confidence intervals of forecast
    arima_confidence_intervals = arima_pred_365_traindata.conf_int()

    # Select lower and upper confidence limits
    arima_lower_limits = arima_confidence_intervals.loc[:, 'lower Open']
    arima_upper_limits = arima_confidence_intervals.loc[:, 'upper Open']

    # Convert arima_pred_mean_365_traindata series to a dataframe
    # Inspect arima_pred_mean_365_traindata_df
    arima_pred_mean_365_traindata_df = arima_pred_mean_365_traindata.to_frame(name='forecasted_mean')



    auto_arima_forecast = arima_results.get_forecast(steps=len(test_data))

    # Forecast mean
    auto_arima_mean_forecast = auto_arima_forecast.predicted_mean

    # Get confidence intervals of forecast
    # Assign it the same index at test data
    auto_arima_forecasted_confidence_intervals = auto_arima_forecast.conf_int()
    auto_arima_forecasted_confidence_intervals.index = test_data.index  # need to do this in order to plot

    # Select lower and upper confidence limits
    auto_arima_forecasted_lower_limits = auto_arima_forecasted_confidence_intervals.loc[:, 'lower Open']
    auto_arima_forecasted_upper_limits = auto_arima_forecasted_confidence_intervals.loc[:, 'upper Open']

    # Convert auto_arima_mean_forecast to a dataframe
    # Inspect auto_arima_mean_forecast
    auto_arima_mean_forecast_df = auto_arima_mean_forecast.to_frame(name='forecasted_mean')
    auto_arima_mean_forecast_df.index = test_data.index



    sns.set(style="darkgrid")

    # Create a Streamlit app
    st.subheader('Result for the ARIMA Model')

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot the train data
    sns.lineplot(x=train_data.index, y='Open', data=train_data, linewidth=4, label='observed train data', ax=ax)

    # Plot the test data
    sns.lineplot(x=test_data.index, y='Open', data=test_data, linewidth=4, label='observed test data', ax=ax)

    # Plot the forecast data
    sns.lineplot(x=auto_arima_mean_forecast_df.index, y=auto_arima_mean_forecast_df['forecasted_mean'],
                 data=auto_arima_mean_forecast_df, linewidth=1, label='forecast', color='red', ax=ax)

    # Shade the area between the confidence intervals
    ax.fill_between(auto_arima_forecasted_lower_limits.index, auto_arima_forecasted_lower_limits,
                    auto_arima_forecasted_upper_limits, color='pink')

    # Display the plot using Streamlit
    st.pyplot(fig)


#--------------------fbprophet-----------------
    pf_data = data.drop(['High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1)

    # Prophet requires 2 columns with variable names in the time series to be:
    # y – Target
    # ds – Datetime
    pf_data.rename(columns={'Open': 'y'}, inplace=True)
    pf_data['ds'] = pf_data.index

    # Create train and test data sets
    pf_train_data = pf_data.loc[:'2016']
    pf_test_data = pf_data.loc['2017':]

    # Fitting the prophet model
    # pf_model = Prophet(changepoint_prior_scale=0.1, daily_seasonality=True)
    pf_model = Prophet(daily_seasonality=True)
    pf_model.fit(pf_train_data)

    # Create future prices & predict prices
    pf_future_prices = pf_model.make_future_dataframe(
        periods=len(pf_test_data))  # 1056 days from 1/1/2017 - 11/22/2019, data in the test set
    pf_forecast = pf_model.predict(pf_future_prices)


    # Create a Prophet model (if not already created)
    # pf_model = Prophet()
    # pf_model.fit(train_data)

    # Plot the components using the plot_components method
    fig2 = pf_model.plot_components(pf_forecast)

    # Display the plot using Streamlit

    abc = pd.read_csv('GLD.csv')
    df = abc.drop(columns=['High', 'Low', 'Close', 'Adj Close', 'Volume'])
#---------
    import plotly.graph_objects as go  # Import 'go' module from Plotly
    from prophet import Prophet

    st.subheader('Prophet forecasting')

    # Create a DataFrame for future dates
    df_prophet = df[['Date', 'Open']]
    df_prophet.columns = ['ds', 'y']

    # Create and fit the Prophet model
    prophet_model = Prophet()
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=365)  # Adjust the number of periods as needed

    # Generate forecast
    prophet_forecast = prophet_model.predict(future)

    # Plot the forecast using Plotly
    fig = go.Figure()

    # Original data
    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Actual'))

    # Forecast
    fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat'], mode='lines', name='Forecast',line=dict(color='red')))

    # Upper and lower bounds
    fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_upper'], fill=None, mode='lines',
                             line=dict(color='gray'), name='Upper Bound'))
    fig.add_trace(go.Scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_lower'], fill='tonexty', mode='lines',
                             line=dict(color='gray'), name='Lower Bound'))

    # Customize layout if needed
    fig.update_layout(title='Prophet Forecast with Streamlit and Plotly',
                      xaxis_title='Date',
                      yaxis_title='Value')

    # Show the plot using Streamlit
    st.plotly_chart(fig)
#______
    def plotly_to_streamlit(fig):
        return fig.to_html(full_html=False)


    # Function to create a Plotly figure from Prophet model and forecast
    def plot_prophet(pf_model, pf_forecast):
        fig = make_subplots(rows=1, cols=1, subplot_titles=["Prophet Forecast"])
        fig.add_trace(go.Scatter(x=pf_model.history['ds'], y=pf_model.history['y'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=pf_forecast['ds'], y=pf_forecast['yhat'], mode='lines', name='Forecast'))
        fig.update_layout(title_text="Prophet Forecast")
        return fig


    # Function to create a Plotly figure from ARIMA forecast
    def plot_arima(arima_forecast_df, arima_lower_limits, arima_upper_limits):
        fig = make_subplots(rows=1, cols=1, subplot_titles=["ARIMA Forecast"])
        fig.add_trace(go.Scatter(x=arima_forecast_df.index, y=arima_forecast_df['forecasted_mean'], mode='lines',
                                 name='Forecast'))
        fig.add_trace(
            go.Scatter(x=arima_forecast_df.index, y=arima_lower_limits, fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                       mode='lines', name='Lower Limit'))
        fig.add_trace(
            go.Scatter(x=arima_forecast_df.index, y=arima_upper_limits, fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
                       mode='lines', name='Upper Limit'))
        fig.update_layout(title_text="ARIMA Forecast")
        return fig


    # Streamlit app
    def main():
        st.subheader("Forecasting the price for next 2 years with the help of prophet for the stationary data")

        # Load your data
        # arima_data = ...
        # pf_data = ...

        # ARIMA Model
        arima_model = SARIMAX(arima_data, order=(0, 1, 0), trend='c')
        arima_results = arima_model.fit()
        forecast_days = 365 * 2
        arima_forecast = arima_results.get_forecast(steps=forecast_days)
        arima_mean_forecast = arima_forecast.predicted_mean
        arima_mean_forecast_df = arima_mean_forecast.to_frame(name='forecasted_mean')
        arima_forecasted_confidence_intervals = arima_forecast.conf_int()
        arima_forecasted_confidence_intervals.index = arima_mean_forecast_df.index
        arima_forecasted_lower_limits = arima_forecasted_confidence_intervals.loc[:, 'lower Open']
        arima_forecasted_upper_limits = arima_forecasted_confidence_intervals.loc[:, 'upper Open']

        # Prophet Model
        all_pf_model = Prophet(changepoint_prior_scale=0.1, daily_seasonality=True)
        all_pf_model.fit(pf_data)
        all_pf_future_prices = all_pf_model.make_future_dataframe(periods=forecast_days)
        all_pf_forecast = all_pf_model.predict(all_pf_future_prices)

        # Streamlit UI
        st.markdown("Forecasting graph for next two years of the given stationary data")
        st.plotly_chart(
            plot_arima(arima_mean_forecast_df, arima_forecasted_lower_limits, arima_forecasted_upper_limits))

        st.markdown("Forecasting price graph for the complete data with next 2 years price")
        st.plotly_chart(plot_prophet(all_pf_model, all_pf_forecast))


    if __name__ == "__main__":
        main()

   

    #----------------------------------arma-model ----------------
elif selected == "ARMA Model":
    st.title("ARMA MODEL")
    from statsmodels.tsa.arima.model import ARIMA

    # Fit
    ar_order = (1, 0, 0)  # Replace p with the desired order for autoregressive (AR) component
    ar_model = ARIMA(train_data['Open'], order=ar_order)
    ar_results = ar_model.fit()

    # Make predictions for the last 365 days of the train data
    pred_365_traindata_arma = ar_results.get_prediction(start=-365, dynamic=False)

    # Forecast mean for these 365 days
    pred_mean_365_traindata_arma = pred_365_traindata_arma.predicted_mean

    # Get confidence intervals of forecast
    confidence_intervals_arma = pred_365_traindata_arma.conf_int()

    # Select lower and upper confidence limits
    lower_limits_arma = confidence_intervals_arma.loc[:, 'lower Open']
    upper_limits_arma = confidence_intervals_arma.loc[:, 'upper Open']
    pred_mean_365_traindata_df = pred_mean_365_traindata_arma.to_frame(name='forecasted_mean')
    st.subheader("ARMA Model Summary")
    st.text(ar_results.summary())
    auto_arima_forecast = ar_results.get_forecast(steps=len(test_data))

    # Forecast mean
    auto_arima_mean_forecast = auto_arima_forecast.predicted_mean

    # Get confidence intervals of forecast
    # Assign it the same index at test data
    auto_arima_forecasted_confidence_intervals = auto_arima_forecast.conf_int()
    auto_arima_forecasted_confidence_intervals.index = test_data.index  # need to do this in order to plot

    # Select lower and upper confidence limits
    auto_arima_forecasted_lower_limits = auto_arima_forecasted_confidence_intervals.loc[:, 'lower Open']
    auto_arima_forecasted_upper_limits = auto_arima_forecasted_confidence_intervals.loc[:, 'upper Open']

    # Convert auto_arima_mean_forecast to a dataframe
    # Inspect auto_arima_mean_forecast
    auto_arima_mean_forecast_df = auto_arima_mean_forecast.to_frame(name='forecasted_mean')
    auto_arima_mean_forecast_df.index = test_data.index

    st.header('Result for the ARMA Model')

    # Create a figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot the train data
    sns.lineplot(x=train_data.index, y='Open', data=train_data, linewidth=4, label='observed train data', color='green',
                 ax=ax)

    # Plot the test data
    sns.lineplot(x=test_data.index, y='Open', data=test_data, linewidth=4, label='observed test data', color='green',
                 ax=ax)

    # Plot the forecast data
    sns.lineplot(x=auto_arima_mean_forecast_df.index, y=auto_arima_mean_forecast_df['forecasted_mean'],
                 data=auto_arima_mean_forecast_df, linewidth=2, label='forecast', color='blue', ax=ax)

    # Shade the area between the confidence intervals
    ax.fill_between(auto_arima_forecasted_lower_limits.index, auto_arima_forecasted_lower_limits,
                    auto_arima_forecasted_upper_limits, color='pink')

    # Display the plot in the Streamlit app
    st.pyplot(fig)

elif selected == "AR Model":
    st.title("AR Model")
    from statsmodels.tsa.ar_model import AutoReg

    ar_model = AutoReg(train_data['Open'], lags=1)
    ar_results = ar_model.fit()

    st.subheader("Autoregressive Model Summary")
    st.text(ar_results.summary())

    auto_arima_forecast = ar_results.forecast(steps=len(test_data))

    # Forecast mean
    auto_arima_mean_forecast = auto_arima_forecast

    # Get confidence intervals of forecast
    # Assign it the same index at test data

    # Convert auto_arima_mean_forecast to a dataframe
    # Inspect auto_arima_mean_forecast
    auto_arima_mean_forecast_df = auto_arima_mean_forecast.to_frame(name='forecasted_mean')
    auto_arima_mean_forecast_df.index = test_data.index

    st.header('Result for the AR Model')

    # Create a figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot the train data
    sns.lineplot(x=train_data.index, y='Open', data=train_data, linewidth=4, label='observed train data', color='green',
                 ax=ax)

    # Plot the test data
    sns.lineplot(x=test_data.index, y='Open', data=test_data, linewidth=4, label='observed test data', color='green',
                 ax=ax)

    # Plot the forecast data
    sns.lineplot(x=auto_arima_mean_forecast_df.index, y=auto_arima_mean_forecast_df['forecasted_mean'],
                 data=auto_arima_mean_forecast_df, linewidth=2, label='forecast', color='blue', ax=ax)

    # Display the plot in the Streamlit app
    st.pyplot(fig)