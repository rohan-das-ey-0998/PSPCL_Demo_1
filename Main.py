import streamlit as st
import pandas as pd
import numpy as np
import plotly as pl
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import warnings
import datetime

from Actual_Forecast.Actual_Vs_Forecast import fetch_data_from_tables_1
from Forecasting_results.Create_Forecast_DB import setup_database_forecast
from Forecasting_results.Forecasting_till_date import train_and_forecast_total, save_dataframe_to_table, \
    fetch_data_from_table

warnings.filterwarnings("ignore")

st.set_page_config(page_title="PSPCL",
                   page_icon="🧊",
                   layout="wide")

st.sidebar.image("Logo/img.png", width=270)

from Data_Format.Data_Upload import process_file
from DB_Table_Creation.DB_Table_Present import setup_database
from Load_Into_DB.Load_Data_Into_Database import process_dataframe_and_store
from Load_Into_DB.Fetch_Data_From_DB import get_table_as_dataframe
from Load_From_DB_Table.Load_from_Table import fetch_data
from Model_Training.Data_Preparation import preprocess_data, add_lag_features, preprocess_outliers, stationarity_check
from Model_Training.Model_Training_Hyparameter_Tuning import evaluate_models, evaluate_forecast, find_best_model, \
    train_and_forecast


def create_line_chart(df, col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df[col], mode='lines', name='Value'))
    fig.update_layout(xaxis_title='Datetime', yaxis_title=col)
    st.plotly_chart(fig)


def create_line_chart_my(df, col, selected_month, selected_year):
    # Convert 'Datetime' column to datetime type if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Datetime']):
        df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Filter dataframe based on selected month and year
    filtered_df = df[(df['Datetime'].dt.month == selected_month) & (df['Datetime'].dt.year == selected_year)]

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['Datetime'], y=filtered_df[col], mode='lines', name='Value'))

    fig.update_layout(xaxis_title='Datetime',
                      yaxis_title=col)

    st.plotly_chart(fig)


def draw_line_graph(dataframe):
    # Set the Datetime column as the index
    dataframe['Datetime'] = pd.to_datetime(dataframe['Datetime'])
    dataframe.set_index('Datetime', inplace=True)

    # Create a figure with plotly
    fig = go.Figure()

    # Add Forecasted Value trace with red color
    fig.add_trace(go.Scatter(x=dataframe.index, y=dataframe['Forecasted_Value'], mode='lines', name='Forecasted Value',
                             line=dict(color='red')))

    # Add Actual Value trace with blue color
    fig.add_trace(
        go.Scatter(x=dataframe.index, y=dataframe['Value'], mode='lines', name='Actual Value', line=dict(color='blue')))

    # Update layout
    fig.update_layout(xaxis_title='Datetime',
                      yaxis_title='Value',
                      xaxis=dict(tickangle=-45),
                      legend=dict(x=0, y=1.1))

    # Display the plot
    st.plotly_chart(fig)


def main():
    st.title("Load Forecasting Dashboard")
    uploaded_file = st.sidebar.file_uploader("Upload Data", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        df = process_file(uploaded_file)
        db_path, table_name = setup_database()
        # st.write(db_path)
        # st.write(table_name)
        process_dataframe_and_store(df, db_path, table_name)
        print("Success in DB ingestion")

        data_df = fetch_data(db_path, table_name)
        df_new_1 = data_df.copy()
        df_new = preprocess_data(df_new_1)

        col1, col2, col3 = st.columns([.3, .3, .4])
        with col1:
            st.metric(label="Total Records", value=data_df.shape[0])
        with col2:
            starting_datetime = data_df['Datetime'].min()
            ending_datetime = data_df['Datetime'].max()
            st.metric(label="Start Date", value=starting_datetime)
        with col3:
            st.metric(label="End Date", value=ending_datetime)

        col5, col6 = st.columns([.5, .5])
        with col5:
            st.subheader("Actual Load Data")
            col5_1, col5_2 = st.columns([.5, .5])
            with col5_1:
                selected_year1 = st.selectbox("Select Year",
                                              sorted(pd.to_datetime(data_df['Datetime']).dt.year.unique()))
            with col5_2:
                selected_month1 = st.selectbox("Select Month",
                                               sorted(pd.to_datetime(data_df['Datetime']).dt.month.unique()))

            create_line_chart_my(data_df, "Value", selected_month1, selected_year1)

        with col6:
            st.subheader("Actual vs Forecasted ")

            r2 = fetch_data_from_tables_1()
            r2['Datetime'] = pd.to_datetime(r2['Datetime'])

            col6_1, col6_2, col6_3 = st.columns([.3, .3, .3])
            with col6_1:
                selected_year = st.selectbox("Select Year", sorted(r2['Datetime'].dt.year.unique()))
            with col6_2:
                selected_month = st.selectbox("Select Month", sorted(r2['Datetime'].dt.month.unique()))
            with col6_3:
                selected_day = st.selectbox("Select Day", sorted(r2['Datetime'].dt.day.unique()))

            filtered_data = r2[
                (r2['Datetime'].dt.year == selected_year) &
                (r2['Datetime'].dt.month == selected_month) &
                (r2['Datetime'].dt.day == selected_day)
                ]

            # Sort the filtered data
            sorted_data = filtered_data.sort_values(by='Datetime')
            draw_line_graph(sorted_data)

        st.subheader("Forecasting : ")

        # Pick a date
        d = st.date_input("Select a Date for Load Forecasting")

        # Disable the Proceed button if a date is not selected
        proceed = st.button("Proceed", disabled=(d is None))

        ## Initializing session state
        if "proceed_state" not in st.session_state:
            st.session_state.proceed_state = False

        if proceed or st.session_state.proceed_state:
            st.session_state.proceed_state = True

            # Display a spinner
            with st.spinner('Forecasting in progress...'):
                df_new = add_lag_features(df_new, col="Value")
                df_new_without_outliers = preprocess_outliers(df_new)
                processed_df = stationarity_check(df_new_without_outliers)

                col7, col8 = st.columns([.6, .4])
                with col7:
                    st.subheader("Training Results")
                    m = evaluate_models(processed_df)
                    st.table(m)

                with col8:
                    st.subheader(f"Forecasting Result for {d}")
                    if d is not None:
                        start_datetime = pd.to_datetime(str(d) + ' 00:00:00')
                        end_datetime = pd.to_datetime(str(d) + ' 23:45:00')
                        # Create date range
                        date_range = pd.date_range(start=start_datetime, end=end_datetime, freq='15T')
                        date_len = len(date_range)

                        best_model = find_best_model(m)
                        result_df = train_and_forecast(date_len, processed_df, best_model)

                    forecast_df_next_day = pd.DataFrame(
                        {'Datetime': date_range, 'Forecasted_Value': np.round(result_df, 0)})
                    forecast_df_next_day = forecast_df_next_day.reset_index(drop=True)
                    st.dataframe(forecast_df_next_day)

                st.write(f"Best Model Selected {best_model}")

                create_line_chart(forecast_df_next_day, "Forecasted_Value")

                try:
                    setup_database_forecast()
                    # st.write("Forecast DB created successfully")
                except Exception as e:
                    print(e)

                # st.write(processed_df)
                # st.write(processed_df.columns)
                # st.write(best_model)
                r = train_and_forecast_total(processed_df, best_model)
                # st.write(r)

                try:
                    save_dataframe_to_table(r)
                    # st.write("Data saved successfully in the forecasting table")
                except Exception as e:
                    print(e)

                try:
                    # st.write("Fetching Data........Forecasting")
                    r1 = fetch_data_from_tables_1()
                    # st.write(r1)
                    # print(r1)

                    # draw_line_graph(r1)
                    # st.line_chart(data=r1)
                except Exception as e:
                    print(e)

    else:
        # Display error message for unsupported file format
        st.error("Please upload a CSV or Excel file.")


# Run the main function to start the app
if __name__ == '__main__':
    main()
