
import streamlit as st
import pandas as pd
import os
import csv
import warnings
import tempfile
import io 
from io import StringIO
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math

warnings.filterwarnings('ignore')

# streamlit run .\CV_difference_V1.py --server.port 8888

st.set_page_config(page_title="BIPM Time Analyser", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title(":chart_with_upwards_trend: CGGTTS data analyser v1.0")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
# Display the Start MJD and End MJD input fields
col1, col2 = st.columns(2)
# Display the Start MJD and End MJD input fields
col3, col4 = st.columns(2)

# Initialize df to an empty DataFrame
df = pd.DataFrame()
df1 = pd.DataFrame()
df2 = pd.DataFrame()


def find_header_end(lines):
    for idx, line in enumerate(lines):
        if "hhmmss  s  .1dg .1dg    .1ns" in line:
            return idx + 2
    return None


df1_mjd=pd.DataFrame()
Avg_refsys_CV = pd.DataFrame()

combined_Colm_data_01 = pd.DataFrame()

# File uploader and data processing
with st.form("my-form1", clear_on_submit=True):
    files_01 = st.file_uploader(":file_folder: Upload the CGGTTS files of receiver 1", accept_multiple_files=True)
    submitted1 = st.form_submit_button("Submit1")


# Initialize selected_df_01 in session state if not present
if 'sel_MJD_df_01' not in st.session_state:
    st.session_state['sel_MJD_df_01'] = pd.DataFrame()

Required_Colm_data_01 = []

def process_data1(files_01):
    if files_01:
        col1.empty()
        col2.empty()
        unique_mjd_values = set()  # To store unique MJD values
        unique_sv_id = set()  # To store the Unique SV ID values
        unique_FRC =set()
        df_01 =pd.DataFrame()
        combined_Colm_data_01 = pd.DataFrame()
        # A list to store cleaned data across multiple files
        
        
         
        for each_file in files_01:
            all_dataframes = []
            filename = each_file.name
            st.write(f"File: {filename}")

            # Read uploaded file contents directly into memory
            file_content = each_file.read().decode()

            # Split the file into lines
            lines = file_content.split('\n')

            data_after_head = []
            # Flag to indicate if we are currently inside a header block
            inside_header = False

            for line in lines:
                # Start of the header
                if line.startswith("CGGTTS")or line.startswith("GGTTS"):
                    inside_header = True

                # If we're not inside a header, process the line as data
                if not inside_header:
                    data_after_head.append(line)

                # End of the header
                if "hhmmss  s  .1dg .1dg    .1ns" in line:
                    inside_header = False

            # Create DataFrame from the data list
            df = pd.DataFrame(data_after_head, columns=['data'])
            
            # Split the 'data' column based on spaces
            df_split = df['data'].str.split(expand=True)
            
            # Rename columns for better understanding
            # column_names = ["SAT", "CL", "MJD", "STTIME", "TRKL", "ELV", "AZTH", "REFSV", "SRSV", "REFSYS", "SRSYS", "DSG", "IOE", "MDTR", "SMDT", "MDIO", "SMDI", "MSIO", "SMSI", "ISG", "FR", "HC", "FRC", "CK", "REFUTC", "DUTC"]
            column_names = ["SAT", "CL", "MJD", "STTIME", "TRKL", "ELV", "AZTH", "REFSV", "SRSV", "REFSYS", "SRSYS", "DSG", "IOE", "MDTR", "SMDT", "MDIO", "SMDI", "MSIO", "SMSI", "ISG", "FR", "HC", "FRC"]
            # df_split.columns = column_names
            # column_names = ["SAT", "CL", "MJD", "STTIME", "TRKL", "ELV", "AZTH", "REFSV", "SRSV", "REFSYS", "FRC"]
            if len(df_split.columns) < len(column_names):
                st.error(f"Error in file {filename}: The number of columns in the data does not match the expected count.")
                break  # Skip further processing for this file
            # Trim the DataFrame to only the required columns
            df_split = df_split.iloc[:, :len(column_names)]

            # Set the column names
            df_split.columns = column_names

            df_split = df_split[df_split['SAT'].notna()] # Skip the lines where SAT column is missing 
            # print(f"File read is :\n {df_split}")
            # print(f"Sv ids in the Data: \n {df_split['SAT']}")
            # Convert to appropriate datatypes
            # df_01['SAT'] = df_split['SAT'].astype(int)
            df_01['SAT'] = df_split['SAT']
            # df_01['MJD'] = df_split['MJD'].astype(float)
            # unique_mjd_values = set(df_split['MJD'])  # Unique MJD values in the list 
            df_split['STTIME'] = df_split['STTIME']  # Keep as string for hhmmss processing

            # Combine hhmmss into MJD
            # df_01['MJD'] += df_split['STTIME'].apply(lambda x: (int(x[0:2]) * 3600 + int(x[2:4]) * 60 + int(x[4:6]) * 1) * 0.00001)

            df_split['MJD'] = df_split['MJD'].astype(str).str.replace('"', '').astype(float)

            # Process STTIME and combine it with MJD
            def convert_sttime_to_seconds(sttime_str):
                # Extract hours, minutes, seconds and convert to total seconds
                hours, minutes, seconds = map(int, [sttime_str[:2], sttime_str[2:4], sttime_str[4:6]])
                return (hours * 3600 + minutes * 60 + seconds)/86400

            # Apply the conversion to STTIME and add it to MJD
            df_split['MJD'] += df_split['STTIME'].apply(lambda x: convert_sttime_to_seconds(x))
            df_01['MJD'] = df_split['MJD']
            # Convert other relevant columns to desired datatypes
            df_01['ELV'] = df_split['ELV'].astype(float)
            df_01['REFSV'] = df_split['REFSV'].astype(float)
            df_01['SRSV'] = df_split['SRSV'].astype(float)
            df_01['REFSYS'] = df_split['REFSYS'].astype(float)
            df_01['FRC'] = df_split['FRC'].astype(str)
            # unique_frc_values = df_split['FRC'].unique()
            # df_split['FRC'] = list(unique_frc_values)

            Required_Colm_data_01.append(df_01)
            # st.write(f"Required data columns : \n {Required_Colm_data_01}")
            unique_FRC.update(df_01['FRC'].unique())
            unique_mjd_values.update(df_01['MJD'].unique())
            unique_sv_id.update(df_01['SAT'].unique())
            
            combined_Colm_data_01 = pd.concat([combined_Colm_data_01, df_01])
            # combined_Colm_data_01 = pd.concat(Required_Colm_data_01, ignore_index=True)

                # Update the "Start MJD" and "End MJD" select boxes
        unique_mjd_values = sorted(unique_mjd_values)
        # unique_mjd_int_values1 = sorted(set(int(mjd) for mjd in unique_mjd_values))
        unique_mjd_int_values1 = sorted(set(int(mjd) for mjd in unique_mjd_values if not pd.isna(mjd)))

        # st.write(f"combined columns data_01: \n {combined_Colm_data_01}")
        return combined_Colm_data_01,unique_mjd_int_values1, unique_FRC

    else:
        return pd.DataFrame()


if files_01:
    processed_data1, unique_mjd_values1, unique_FRC1 = process_data1(files_01)
    # unique_mjd_int_values1 = sorted(set(int(mjd) for mjd in unique_mjd_values))
    # st.write("All MJD values from files:", unique_mjd_values1)
    unique_mjd_int_values1 = sorted(set(int(mjd) for mjd in unique_mjd_values1 if not pd.isna(mjd)))
    st.session_state['df1_total'] = processed_data1
    st.session_state['start_mjd_01'] = unique_mjd_int_values1[0]
    st.session_state['end_mjd_01'] = unique_mjd_int_values1[-1]
    st.session_state['unique_FRC1'] = unique_FRC1
    st.session_state['show_plot1'] = False  # Reset plot visibility


def process_4_plot1(given_data1, start_mjd, end_mjd):
    # Ensure MJD values are of the correct type for comparison
    given_data1["MJD"] = pd.to_numeric(given_data1["MJD"], errors='coerce')
    
        # Check if start and end MJD are the same (data for a single day)
    if start_mjd == end_mjd:
        # Filter data for that specific day
        filtered_df = given_data1[(given_data1["MJD"] < float(start_mjd)+1) & (given_data1["MJD"] > float(start_mjd)-1)]
    else:
        # Filter the data based on the MJD range
        filtered_df = given_data1[
            (given_data1["MJD"].notnull()) &
            (given_data1["MJD"] >= float(start_mjd)) &
            (given_data1["MJD"] <= float(end_mjd+1))
        ]
    
    # st.write("Selected Start MJD:", start_mjd)
    # st.write("Selected End MJD:", end_mjd)
    
    return filtered_df



def plot_data1(frequency1):
    # Filter the MJD-filtered data based on the frequency
    df1_data_filtered = st.session_state['sel_MJD_df_01'][st.session_state['sel_MJD_df_01']['FRC'] == frequency1]
    # st.write(f"Filtered data: \n {df1_data_filtered}")
    st.session_state["sel_MJD_FRC_01"] = df1_data_filtered
    

    if not df1_data_filtered.empty:
        Avg_refsys_CV = (df1_data_filtered.groupby("MJD")["REFSYS"].mean().reset_index())
        Avg_refsys_CV["REFSYS"] = (Avg_refsys_CV["REFSYS"]*0.1).round(2)
        mean_value = (Avg_refsys_CV["REFSYS"].mean())

        # st.markdown(f"## Receiver 1 Average REFSYS: {frequency1}")

        # Create a scatter plot with Plotly
        fig = go.Figure()

        # Add scatter plot of data points
        fig.add_trace(go.Scatter(
            x=Avg_refsys_CV["MJD"], 
            y=Avg_refsys_CV["REFSYS"], 
            mode='markers',
            name='REFSYS'
        ))

        # Add a dashed mean line
        fig.add_hline(y=mean_value, line_dash="dash", line_color="red",
                    annotation_text=f"Mean: {mean_value:.2f}", 
                    annotation_position="bottom right",
                     annotation_font=dict(size=18, color="black"))

        # Update layout for better presentation
        fig.update_layout(
            title=f"Receiver 1 Average REFSYS: {frequency1}",
            xaxis_title="MJD",
            yaxis_title="REFSYS",
            yaxis=dict(tickmode='auto', nticks =10),
            xaxis =dict(tickfont= dict(size=14, color ="black"), exponentformat ='none')
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Selected frequency data is not available in the selected MJD range ")


# MJD Selection
if 'df1_total' in st.session_state and unique_mjd_int_values1:
    cols = st.columns(2)  # Creates two columns

    # "Start MJD" in the first column
    with cols[0]:
        start_mjd_input1 = st.selectbox("Start MJD", options=unique_mjd_int_values1, key='start_mjd01')

    # st.session_state['start_mjd_input1'] = start_mjd_input1
    # "End MJD" in the second column
    with cols[1]:
        # Update the end_values1 list based on the selected start_mjd_input
        end_values1 = [mjd for mjd in unique_mjd_int_values1 if mjd >= start_mjd_input1]
        # end_values1 = [mjd for mjd in all_mjd_float_values if mjd >= start_mjd_input1]
        end_mjd_input1 = st.selectbox("End MJD", options=end_values1, index=len(end_values1)-1  if end_values1 else 0, key='end_mjd01')
        # end_mjd_input1 = st.selectbox("End MJD", options=end_values1, index=0 if end_values1 else -1, key='end_mjd01')

    st.session_state['sel_MJD_df_01'] = process_4_plot1(st.session_state['df1_total'], start_mjd_input1, end_mjd_input1)
    
    # If unique frequencies are available, update the plot
    if 'unique_FRC1' in st.session_state and 'sel_MJD_df_01' in st.session_state:
        filtered_unique_frequencies = [freq for freq in st.session_state['unique_FRC1'] if pd.notna(freq)]
        if filtered_unique_frequencies:
            # Re-select the frequency after MJD filtering
            selected_frequency1 = st.radio("Select Frequency", filtered_unique_frequencies, index=0, key='Frequency1', horizontal=True)
            st.session_state.selected_frequency1= selected_frequency1
            plot_data1(selected_frequency1)
        else:
            st.error("No valid frequencies available to process the data")


# Initialize the variable outside of the conditional blocks
unique_mjd_int_values2 = []
processed_data1= None
selected_df_01 = None
# processed_data = Selected_MJD_data1(uploaded_files)
# Initialize variables
unique_mjd_int_values = []

if 'show_plot2' not in st.session_state:
    st.session_state['show_plot2'] = False

# File uploader and data processing
with st.form("my-form2", clear_on_submit=True):
    files_02 = st.file_uploader(":file_folder: Upload the CGGTTS files of receiver 2", accept_multiple_files=True)
    submitted1 = st.form_submit_button("Submit2")

Required_Colm_data_02 = []

def process_data2(files_02):
    if files_02:
        col3.empty()
        col4.empty()
        unique_mjd_values = set()  # To store unique MJD values
        unique_sv_id = set()  # To store the Unique SV ID values
        unique_FRC =set()
        df_02 =pd.DataFrame()
        combined_Colm_data_02 = pd.DataFrame()
        # A list to store cleaned data across multiple files
        
                 
        for each_file in files_02:
            all_dataframes = []
            filename = each_file.name
            st.write(f"File: {filename}")

            # Read uploaded file contents directly into memory
            file_content = each_file.read().decode()

            # Split the file into lines
            lines = file_content.split('\n')

            data_after_head = []
            # Flag to indicate if we are currently inside a header block
            inside_header = False

            for line in lines:
                # Start of the header
                if line.startswith("CGGTTS")or line.startswith("GGTTS"):
                    inside_header = True

                # If we're not inside a header, process the line as data
                if not inside_header:
                    data_after_head.append(line)

                # End of the header
                if "hhmmss  s  .1dg .1dg    .1ns" in line:
                    inside_header = False

            # Create DataFrame from the data list
            df = pd.DataFrame(data_after_head, columns=['data'])
            
            # Split the 'data' column based on spaces
            df_split = df['data'].str.split(expand=True)
            
            # Rename columns for better understanding
            # column_names = ["SAT", "CL", "MJD", "STTIME", "TRKL", "ELV", "AZTH", "REFSV", "SRSV", "REFSYS", "SRSYS", "DSG", "IOE", "MDTR", "SMDT", "MDIO", "SMDI", "MSIO", "SMSI", "ISG", "FR", "HC", "FRC", "CK", "REFUTC", "DUTC"]
            column_names = ["SAT", "CL", "MJD", "STTIME", "TRKL", "ELV", "AZTH", "REFSV", "SRSV", "REFSYS", "SRSYS", "DSG", "IOE", "MDTR", "SMDT", "MDIO", "SMDI", "MSIO", "SMSI", "ISG", "FR", "HC", "FRC"]
            # df_split.columns = column_names
            # column_names = ["SAT", "CL", "MJD", "STTIME", "TRKL", "ELV", "AZTH", "REFSV", "SRSV", "REFSYS", "FRC"]
            if len(df_split.columns) < len(column_names):
                st.error(f"Error in file {filename}: The number of columns in the data does not match the expected count.")
                break  # Skip further processing for this file
            # Trim the DataFrame to only the required columns
            df_split = df_split.iloc[:, :len(column_names)]

            # Set the column names
            df_split.columns = column_names

            df_split = df_split[df_split['SAT'].notna()] # Skip the lines where SAT column is missing 
            # print(f"File read is :\n {df_split}")
            # print(f"Sv ids in the Data: \n {df_split['SAT']}")
            # Convert to appropriate datatypes
            # df_02['SAT'] = df_split['SAT'].astype(int)
            df_02['SAT'] = df_split['SAT']
            # df_01['MJD'] = df_split['MJD'].astype(float)
            # unique_mjd_values = set(df_split['MJD'])  # Unique MJD values in the list 
            df_split['STTIME'] = df_split['STTIME']  # Keep as string for hhmmss processing

            # Combine hhmmss into MJD
            # df_01['MJD'] += df_split['STTIME'].apply(lambda x: (int(x[0:2]) * 3600 + int(x[2:4]) * 60 + int(x[4:6]) * 1) * 0.00001)

            df_split['MJD'] = df_split['MJD'].astype(str).str.replace('"', '').astype(float)

            # Process STTIME and combine it with MJD
            def convert_sttime_to_seconds(sttime_str):
                # Extract hours, minutes, seconds and convert to total seconds
                hours, minutes, seconds = map(int, [sttime_str[:2], sttime_str[2:4], sttime_str[4:6]])
                return (hours * 3600 + minutes * 60 + seconds)/86400

            # Apply the conversion to STTIME and add it to MJD
            df_split['MJD'] += df_split['STTIME'].apply(lambda x: convert_sttime_to_seconds(x) )
            df_02['MJD'] = df_split['MJD']
            # Convert other relevant columns to desired datatypes
            df_02['ELV'] = df_split['ELV'].astype(float)
            df_02['REFSV'] = df_split['REFSV'].astype(float)
            df_02['SRSV'] = df_split['SRSV'].astype(float)
            df_02['REFSYS'] = df_split['REFSYS'].astype(float)
            df_02['FRC'] = df_split['FRC'].astype(str)
            # unique_frc_values = df_split['FRC'].unique()
            # df_split['FRC'] = list(unique_frc_values)

            Required_Colm_data_02.append(df_02)
            unique_FRC.update(df_02['FRC'].unique())
            unique_mjd_values.update(df_02['MJD'].unique())
            unique_sv_id.update(df_02['SAT'].unique())
        
            combined_Colm_data_02 = pd.concat(Required_Colm_data_02, ignore_index=True)

                # Update the "Start MJD" and "End MJD" select boxes
        unique_mjd_values = sorted(unique_mjd_values)
        unique_mjd_int_values2 = sorted(set(int(mjd) for mjd in unique_mjd_values))

        # st.write(f"combined columns data: \n {combined_Colm_data_02}")
        return combined_Colm_data_02,unique_mjd_int_values2, unique_FRC

    else:
        return pd.DataFrame()


if files_02:
    processed_data2, unique_mjd_values, unique_FRC2 = process_data1(files_02)
    # unique_mjd_int_values1 = sorted(set(int(mjd) for mjd in unique_mjd_values))
    unique_mjd_int_values2 = sorted(set(int(mjd) for mjd in unique_mjd_values if not pd.isna(mjd)))
    st.session_state['df2_total'] = processed_data2
    st.session_state['start_mjd_02'] = unique_mjd_int_values2[0]
    st.session_state['end_mjd_02'] = unique_mjd_int_values2[-1]
    st.session_state['unique_FRC2'] = unique_FRC2
    st.session_state['show_plot2'] = False  # Reset plot visibility


def process_4_plot2(given_data2, start_mjd, end_mjd):
    # Ensure MJD values are of the correct type for comparison
    given_data2["MJD"] = pd.to_numeric(given_data2["MJD"], errors='coerce')
    
        # Check if start and end MJD are the same (data for a single day)
    if start_mjd == end_mjd:
        # Filter data for that specific day
        filtered_df = given_data2[(given_data2["MJD"] < float(start_mjd)+1) & (given_data2["MJD"] > float(start_mjd)-1)]
    else:
        # Filter the data based on the MJD range
        filtered_df = given_data2[
            (given_data2["MJD"].notnull()) &
            (given_data2["MJD"] >= float(start_mjd)) &
            (given_data2["MJD"] <= float(end_mjd+1))
        ]
    
    # st.write("Selected Start MJD:", start_mjd)
    # st.write("Selected End MJD:", end_mjd)
    
    return filtered_df


def plot_data(frequency2):
    # Filter the MJD-filtered data based on the frequency
    df2_data_filtered = st.session_state['sel_MJD_df_02'][st.session_state['sel_MJD_df_02']['FRC'] == frequency2]
    st.session_state["sel_MJD_FRC_02"] = df2_data_filtered
    # st.write(f"Filtered data: \n {df1_data_filtered}")

    #     st.line_chart(Avg_refsys_CV.set_index("MJD")[["REFSYS", "Avg"]])
    if not df2_data_filtered.empty:
        Avg_refsys_CV = (df2_data_filtered.groupby("MJD")["REFSYS"].mean().reset_index())
        Avg_refsys_CV["REFSYS"] = (Avg_refsys_CV["REFSYS"]*0.1).round(2)
        mean_value = Avg_refsys_CV["REFSYS"].mean()

        # st.markdown(f"## Receiver 2 Average REFSYS: {frequency2}")

        # Create a scatter plot with Plotly
        fig = go.Figure()

        # Add scatter plot of data points
        fig.add_trace(go.Scatter(
            x=Avg_refsys_CV["MJD"], 
            y=Avg_refsys_CV["REFSYS"], 
            mode='markers',
            name='REFSYS'
        ))

        # Add a dashed mean line
        fig.add_hline(y=mean_value, line_dash="dash", line_color="red",
                    annotation_text=f"Mean: {mean_value:.2f}", 
                    annotation_position="bottom right",
                     annotation_font=dict(size=18, color="black"))

        # Update layout for better presentation
        fig.update_layout(
            title=f"Receiver 2 Average REFSYS: {frequency2}",
            xaxis_title="MJD",
            yaxis_title="REFSYS",
            yaxis=dict(tickmode='auto', nticks =10),
            xaxis =dict(tickfont= dict(size=14, color ="black"), exponentformat ='none')
        )
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("No valid frequencies available to process the data")


# MJD Selection
if 'df2_total' in st.session_state and unique_mjd_int_values2:
    cols = st.columns(2)  # Creates two columns

    # "Start MJD" in the first column
    with cols[0]:
        start_mjd_input2 = st.selectbox("Start MJD", options=unique_mjd_int_values2, key='start_mjd02')

    # "End MJD" in the second column
    with cols[1]:
        # Update the end_values1 list based on the selected start_mjd_input
        end_values2 = [mjd for mjd in unique_mjd_int_values2 if mjd >= start_mjd_input2]
        end_mjd_input2 = st.selectbox("End MJD", options=end_values2, index=len(end_values2) - 1 if end_values2 else 0, key='end_mjd02')
    
    # Filter the DataFrame based on the MJD selection

    st.session_state['sel_MJD_df_02'] = process_4_plot2(st.session_state['df2_total'], start_mjd_input2, end_mjd_input2)
    
        # st.session_state['sel_MJD_df_01'] = process_4_plot1(st.session_state['df1_total'], start_mjd_input1, end_mjd_input1)
    
    # If unique frequencies are available, update the plot
    if 'unique_FRC2' in st.session_state and 'sel_MJD_df_02' in st.session_state:
        filtered_unique_frequencies = [freq for freq in st.session_state['unique_FRC2'] if pd.notna(freq)]
        if filtered_unique_frequencies:
            # Re-select the frequency after MJD filtering
            selected_frequency2 = st.radio("Select Frequency", filtered_unique_frequencies, index=0, key='Frequency2', horizontal=True)
            st.session_state.selected_frequency2= selected_frequency2
            plot_data(selected_frequency2)
        else:
            st.error("No valid frequencies available for selection.")
    

# Function to create the DataFrame for CSV
def create_csv_data_CV(starting_mjd, ending_mjd, SVids, frequency1, frequency2, selected_data):
    # Creating DataFrame for data section
    # x=df3_filtered["MJD_time"], 
    #                 y=df3_filtered["CV_avg_diff"]
    selected_data["MJD_time"] = selected_data["MJD_time"].apply(lambda x: f"{x:.5f}")
    data_df = pd.DataFrame({
        'MJD': selected_data["MJD_time"],
        'CV_difference (ns)': selected_data['CV_avg_diff']
    })

    # Creating header information
    header_CV_info = (
        f"Common View Time Transfer Link Performance \n"
        f"Start MJD: {starting_mjd}\n"
        f"End MJD: {ending_mjd}\n"
        f"Frequency selected for comparision in receiver 1: {frequency1}\n"
        f"Frequency selected for comparision in receiver 2: {frequency2}\n"
        f"Selected satellites for time transfer: {', '.join(SVids)}\n"
    )

    return header_CV_info, data_df

# Function to convert header and DataFrame to CSV for download
def convert_to_csv(header, df):
    output = StringIO()
    output.write(header)
    df.to_csv(output,sep='\t', index=False, header=True)
    return output.getvalue()


# Function to create the DataFrame for CSV
def create_csv_data_AV(starting_mjd, ending_mjd, SVids, frequency1, frequency2, selected_data):
    # Creating DataFrame for data section
    # Format the 'MJD' column to have only 5 digits after the decimal
    selected_data["MJD_time"] = selected_data["MJD_time"].apply(lambda x: f"{x:.5f}")
    
    data_AV_df = pd.DataFrame({
        'MJD': selected_data["MJD_time"],
        'AV_difference (ns)': selected_data['AV_diff']
    })

    # Creating header information
    header_AV_info = (
        f"ALL in View Time Transfer Link Performance \n"
        f"Start MJD: {starting_mjd}\n"
        f"End MJD: {ending_mjd}\n"
        f"Frequency selected for comparision in receiver 1: {frequency1}\n"
        f"Frequency selected for comparision in receiver 2: {frequency2}\n"
        f"Selected Satellites for time transfer: {', '.join(SVids)}\n"
    )

    return header_AV_info, data_AV_df
    

data1_avail =0
data2_avail =0

# st.sidebar.header("Lab 2 Data")
# plot_option2 = st.sidebar.button("Plot Avg RefSys",key=10)
st.sidebar.header("Pricipals of CV & AV time transfer")
plot_CV = st.sidebar.button("PDF material", key= 'Material')

st.sidebar.header("Common View Performance")
plot_CV = st.sidebar.button("Plot CV", key= 'Common_view')
# plot_button = st.sidebar.button("CV Performance", key=5)

st.sidebar.header("All in View Performance")
plot_AV = st.sidebar.button("Plot AV", key= 'All_in_view')
# plot_button = st.sidebar.button("CV Performance", key=5)



df3 = pd.DataFrame(columns=['MJD_time', 'CV_avg_diff'])
CV_data =[]

# Initialize session variables 

if 'df1_mjd' not in st.session_state:
    st.session_state.df1_mjd = None

if 'df2_mjd' not in st.session_state:
    st.session_state.df2_mjd = None

if 'selected_svids' not in st.session_state:
    st.session_state.selected_svids = ['ALL']

unique_SVIDs = []


if 'sel_MJD_FRC_01' in st.session_state and 'sel_MJD_FRC_02' in st.session_state:
    
    st.session_state.df1_mjd = st.session_state.sel_MJD_FRC_01
    st.session_state.df2_mjd = st.session_state.sel_MJD_FRC_02
    # st.write("Hello world")

    # if not df1_mjd.empty and not df2_mjd.empty:
    if st.session_state.df1_mjd is not None and st.session_state.df2_mjd is not None:
        
        # Extract unique values
        unique_svids_df1 = st.session_state.df1_mjd['SAT'].unique()
        unique_svids_df2 = st.session_state.df2_mjd['SAT'].unique()
        
        unique_SVIDs = []
        unique_MJD_times = sorted(set(st.session_state.df1_mjd["MJD"]).union(set(st.session_state.df2_mjd["MJD"])))
        # st.write(f"MJD selected Data 01: \n {st.session_state.df1_mjd}")
        # st.write(f"MJD selected Data 02: \n {st.session_state.df2_mjd}")

        all_common_svids = set()
        df1_mjd_01 =[]
        df2_mjd_02 =[]
        missing_session =[]

        for mjd_time in unique_MJD_times:
            # Filter dataframes for the current mjd_time
            df1_mjd_01 = st.session_state.df1_mjd[st.session_state.df1_mjd["MJD"] == mjd_time]
            df2_mjd_02 = st.session_state.df2_mjd[st.session_state.df2_mjd["MJD"] == mjd_time]
            
            common_svids = set(df1_mjd_01["SAT"]) & set(df2_mjd_02["SAT"])            

            # Add to the all_common_svids set
            all_common_svids.update(common_svids)
            

        # Convert set to list
        unique_SVIDs = list(all_common_svids)

        if unique_SVIDs:
            # print(f"Unique SAT in combined data: \n{unique_SVIDs}")
            # If the session_state.plot_data doesn't exist, initialize it to None
            if 'plot_CV_data' not in st.session_state:
                st.session_state.plot_CV_data = None                       

            if 'plot_AV_data' not in st.session_state:
                st.session_state.plot_AV_data = None 
            # Sidebar options
            # st.sidebar.header("Common View Data")

                # Initialize selected_svids in session_state if not present
            if 'selected_svids' not in st.session_state:
                st.session_state.selected_svids = ['ALL']

            selected_svids = st.sidebar.multiselect(
                "Select SV_ids:",
                options=['ALL'] + list(unique_SVIDs),
                # default=st.session_state.selected_svids,
                default =['ALL'],
                key= 12)  # Use the unique key here
            # Update the session state
            # Handle ALL option
               # Update the session state
            if 'ALL' in selected_svids or len(selected_svids) == len(unique_SVIDs):
                svids_to_use = unique_SVIDs
            else:
                svids_to_use = selected_svids

            # Update session state for selected svids
            st.session_state.selected_svids = svids_to_use
            # plot_button = st.sidebar.button("Plot CV")

            if plot_CV:

                if 'ALL' in st.session_state.selected_svids:
                    st.session_state.selected_svids = list(unique_SVIDs)

                # print(f"Selected SV IDs are: {st.session_state.selected_svids}")
                unique_MJD_times = sorted(set(st.session_state.df1_mjd["MJD"]).intersection(set(st.session_state.df2_mjd["MJD"])))
                # unique_MJD_times = sorted(set(df1_mjd_01["MJD"]).union(set(df2_mjd_02["MJD"])))
                # print ("Unique MJD values:",unique_MJD_times)
                if unique_MJD_times: # If there are common MJD exists between the files 
                    for unique_time in unique_MJD_times: # For each unique MJD time in the selected timing range 
                        refsys_value_df1 = []
                        refsys_value_df2 = []
                        avg_CV_diff = 0
                        if unique_time in st.session_state.df1_mjd["MJD"].values and unique_time in st.session_state.df2_mjd["MJD"].values:
                            for svid in st.session_state.selected_svids:
                                
                                condition1 = (st.session_state.df1_mjd["SAT"] == svid) & (st.session_state.df1_mjd["MJD"] == unique_time)
                                
                                if condition1.any():
                                    refsys_value_df1.extend(st.session_state.df1_mjd[condition1]["REFSYS"].tolist())
                                    # refsys_value_df1 = df1_mjd[df1_mjd["sv_id"] == svid]["refsys"].tolist()
                                    
                                condition2 = (st.session_state.df2_mjd["SAT"] == svid) & (st.session_state.df2_mjd["MJD"] == unique_time)
                                
                                if condition2.any():
                                    refsys_value_df2.extend(st.session_state.df2_mjd[condition2]["REFSYS"].tolist())
                                
                            # Calculate all the CV_diff values for the given svid and mjd_time
                            CV_diffs = [val1 - val2 for val1, val2 in zip(refsys_value_df1, refsys_value_df2)]

                            if CV_diffs:  # CV_difference can be zero also 
                                avg_CV_diff = (sum(CV_diffs) *0.1)/ len(CV_diffs) # 0.1 ns is the unit of REFSYS in CGGTTS data format 
                                new_row = {'MJD_time': unique_time, 'CV_avg_diff': round(avg_CV_diff,2)}
                            else:
                                # Handle the case where there are no valid diffs (e.g., one or both lists are empty)
                                avg_CV_diff = None  # Use None to represent missing data rather than zero
                                new_row = {'MJD_time': unique_time, 'CV_avg_diff': avg_CV_diff}

                            # else:
                            #     print (f"CV difference is zero at session: {unique_time}")
                            CV_data.append(new_row)

                        else:
                            missing_session.append(unique_time)
                            # st.write("")
                else: 
                    st.error("Files doesn't belong to same time period ")

                st.session_state.plot_CV_data = pd.DataFrame(CV_data, columns=['MJD_time', 'CV_avg_diff'])
                
            # if  not st.session_state.plot_data.empty:
            # Plotting 
            
            if st.session_state.plot_CV_data is not None and not st.session_state.plot_CV_data.empty:
                df3 = st.session_state.plot_CV_data

                # User inputs for the y-axis range
                col1, col2 = st.columns(2)
                with col1:
                    user_start_y = st.number_input("Lower Outlier limit", min_value=float(df3["CV_avg_diff"].min()), max_value=float(df3["CV_avg_diff"].max()), value=float(df3["CV_avg_diff"].min()))
                with col2:
                    user_end_y = st.number_input("Upper Outlier limit", min_value=float(df3["CV_avg_diff"].min()), max_value=float(df3["CV_avg_diff"].max()), value=float(df3["CV_avg_diff"].max()))

                # Filter the data based on user selection and calculate mean
                df3_filtered = df3[(df3["CV_avg_diff"] >= user_start_y) & (df3["CV_avg_diff"] <= user_end_y)]
                user_mean_val = df3_filtered["CV_avg_diff"].mean()

                # Display the user selected mean
                # st.write(f"Mean value of data with in selected limit ({user_start_y} - {user_end_y}): {user_mean_val:.2f} ns")
              
                # Set x-axis range and filter rows of the dataframe
                min_x = math.floor(min(df3["MJD_time"]))
                max_x = math.ceil(max(df3["MJD_time"]))
                              

                # Create scatter plot
                fig = go.Figure()

                # Add scatter plot of data points
                fig.add_trace(go.Scatter(
                    x=df3_filtered["MJD_time"], 
                    y=df3_filtered["CV_avg_diff"], 
                    mode='markers',
                    name='CV_avg_diff',
                    marker=dict(size=10)  # Increase marker size
                ))

                # Add a thicker horizontal line for the mean
                fig.add_hline(y=user_mean_val, line_dash="dash", line_color="red", line_width=3,
                            annotation_text=f"Mean: {user_mean_val:.2f} ns", 
                            annotation_position="top right",
                            annotation_font=dict(size=18, color="black"))

                # Set plot titles and labels with increased font size and black color
                fig.update_layout(
                    title=f"CV performance (MJD: {min_x} - {max_x-1})",
                    title_font=dict(size=20, color="black"),
                    xaxis_title="MJD time",
                    xaxis_title_font=dict(size=16, color="black"),
                    yaxis_title="Time difference (ns)",
                    yaxis_title_font=dict(size=16, color="black"),
                    xaxis=dict(
                        tickmode='array',
                        # tickvals=[i for i in range(int(min_x), int(max_x) + 1) if i % 1 == 0],
                        # tickformat="05d",
                        tickfont=dict(size=14, color="black"),
                        exponentformat='none' 
                    ),
                    yaxis=dict(
                        tickmode='auto', nticks =10,
                        tickfont=dict(size=14, color="black")
                    ),
                    # yaxis=dict(tickmode='auto', nticks =10)
                    autosize=False,
                    width=800,
                    height=600
                )

                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Data file processing 
                                
                if st.sidebar.button('Get CV file of this data'): 
                    # Create the CSV data
                    # Create the CSV header and data
                    header, data_df = create_csv_data_CV(min_x, max_x-1, 
                                                    st.session_state.selected_svids, st.session_state.selected_frequency1,
                                                    st.session_state.selected_frequency2, df3_filtered)

                    # Convert to CSV
                    csv = convert_to_csv(header, data_df)

                    # Create download button
                    st.sidebar.download_button(
                        label="Download CV data",
                        data=csv,
                        file_name="common_view_performance.csv",
                        mime="text/csv",
                    )
                        

            if 'selected_svids' not in st.session_state:
                st.session_state.selected_svids = ['ALL']

            if plot_AV:

                if 'ALL' in st.session_state.selected_svids or len(st.session_state.selected_svids) == len(unique_SVIDs):
                    svids_to_use = unique_SVIDs
                else:
                    svids_to_use = st.session_state.selected_svids
                # If individual SV_ids are selected/deselected, remove ALL
                if len(st.session_state.selected_svids) < len(unique_SVIDs) + 1:
                    st.session_state.selected_svids = [svid for svid in st.session_state.selected_svids if svid != 'ALL']

                st.session_state.selected_svids = svids_to_use

                if 'ALL' in st.session_state.selected_svids:
                    st.session_state.selected_svids = list(unique_SVIDs)        

                # This is just an example and may not reflect your actual formula
                st.session_state.df1_mjd['inv_cos2'] = 1 / pow(np.cos(np.radians(st.session_state.df1_mjd['ELV']*0.1)),2)
                st.session_state.df2_mjd['inv_cos2'] = 1 / pow(np.cos(np.radians(st.session_state.df2_mjd['ELV']*0.1)),2)
        

                # Define a function to calculate k weight based on the inverse of the square of the cosine of the elevation angle
                def calculate_k(group):
                    sum_inv_cos2 = group['inv_cos2'].sum()
                    k = 1 / sum_inv_cos2 if sum_inv_cos2 != 0 else float('inf')  # Use infinity to represent undefined k
                    return k

                # print(f"Selected SV IDs are: {st.session_state.selected_svids}")
                unique_MJD_times = sorted(set(st.session_state.df1_mjd["MJD"]).union(set(st.session_state.df2_mjd["MJD"])))
                # print ("Unique MJD values:",unique_MJD_times)
                if unique_MJD_times: # If there are common MJD exists between the files 
                    AV_data = []
                    for unique_time in unique_MJD_times: # For each unique MJD time in the selected timing range 
                        refsys_value_df1 = []
                        refsys_value_df2 = []
                        Elv_value_df1 = []
                        Elv_value_df2 = []
                        weighted_means_df1 = []
                        weighted_means_df2 = []
                        avg_CV_diff = 0
                        # Initialize a dictionary to store the Avg_AV_refsys for each unique MJD
                        avg_AV_refsys_dict = {}

                        if unique_time in st.session_state.df1_mjd["MJD"].values and unique_time in st.session_state.df2_mjd["MJD"].values:
                            # THE LOGIC CHANGES FROM HERE FOR AV
                            df1_filtered = st.session_state.df1_mjd[st.session_state.df1_mjd["MJD"] == unique_time]
                            df2_filtered = st.session_state.df2_mjd[st.session_state.df2_mjd["MJD"] == unique_time]
                            
                            # Calculate k values outside the satellite loop since it's constant for each MJD
                            k1_value = df1_filtered.groupby('MJD').apply(calculate_k).iloc[0]
                            k2_value = df2_filtered.groupby('MJD').apply(calculate_k).iloc[0]


                            condition1 = df1_filtered["SAT"].isin(st.session_state.selected_svids) # Boolean list when the condition is satisfied 
                            # st.write(f"Satellite: {np.sum(condition1)}")
                            condition2 = df2_filtered["SAT"].isin(st.session_state.selected_svids) # 
                            
                            if condition1.any():
                                weighted_mean_df1 = ((df1_filtered.loc[condition1, 'REFSYS'] * k1_value * df1_filtered.loc[condition1, 'inv_cos2']).sum())*0.1
                                # st.write(f"Weights: {weighted_mean_df1}")
                                # weighted_means_df1.append(weighted_mean_df1)
                            
                            if condition2.any():
                                weighted_mean_df2 = ((df2_filtered.loc[condition2, 'REFSYS'] * k2_value * df2_filtered.loc[condition2, 'inv_cos2']).sum())*0.1
                                
                            
                            # Only compute AV_diff_refsys if both conditions are met
                            if condition1.any() and condition2.any():
                                AV_diff_refsys = weighted_mean_df1 - weighted_mean_df2

                                if AV_diff_refsys:  # AV_difference can be zero also 
                                     
                                    new_row = {'MJD_time': unique_time, 'AV_diff': round(AV_diff_refsys,2)}
                                else:
                                    # Handle the case where there are no valid diffs (e.g., one or both lists are empty)
                                    AV_diff_refsys = None  # Use None to represent missing data rather than zero
                                    new_row = {'MJD_time': unique_time, 'AV_diff': round(AV_diff_refsys,2)}
                                
                                AV_data.append(new_row)

                        else:
                            missing_session.append(unique_time)
                   
                    # Assuming st.session_state is a Streamlit state object
                    st.session_state.plot_AV_data = pd.DataFrame(AV_data, columns=['MJD_time', 'AV_diff'])
                else: 
                    st.error("Files doesn't belong to the same time period ")

            if st.session_state.plot_AV_data is not None and not st.session_state.plot_AV_data.empty:
                df4 = st.session_state.plot_AV_data

                # User inputs for the y-axis range
                col1, col2 = st.columns(2)
                with col1:
                    user_start_y = st.number_input("Lower Outlier limit", min_value=float(df4["AV_diff"].min()), max_value=float(df4["AV_diff"].max()), value=float(df4["AV_diff"].min()))
                with col2:
                    user_end_y = st.number_input("Upper Outlier limit", min_value=float(df4["AV_diff"].min()), max_value=float(df4["AV_diff"].max()), value=float(df4["AV_diff"].max()))

                # Filter the data based on user selection and calculate mean
                df4_filtered = df4[(df4["AV_diff"] >= user_start_y) & (df4["AV_diff"] <= user_end_y)]
                user_mean_val = df4_filtered["AV_diff"].mean()

                # Set x-axis range
                min_x = math.floor(min(df4["MJD_time"]))
                max_x = math.ceil(max(df4["MJD_time"]))

                # Create scatter plot using Plotly
                fig = go.Figure()

                # Add scatter plot of data points
                fig.add_trace(go.Scatter(
                    x=df4_filtered["MJD_time"], 
                    y=df4_filtered["AV_diff"], 
                    mode='markers',
                    name='AV_diff',
                    marker=dict(size=10)  # Increase marker size
                ))

                # Add a thicker horizontal line for the mean and customize annotation text
                fig.add_hline(y=user_mean_val, line_dash="dash", line_color="red", line_width=3,
                            annotation_text=f"Mean: {user_mean_val:.2f} ns", 
                            annotation_position="top right",
                            annotation_font=dict(size=16, color="black"))

                # Set plot titles and labels with increased font size and black color
                fig.update_layout(
                    title=f"AV performance ( MJD: {min_x} - {max_x-1}))",
                    title_font=dict(size=20, color="black"),
                    xaxis_title="MJD time",
                    xaxis_title_font=dict(size=16, color="black"),
                    yaxis_title="Time difference (ns)",
                    yaxis_title_font=dict(size=16, color="black"),
                    xaxis=dict(
                        tickmode='array',
                        # tickvals=[i for i in range(int(min_x), int(max_x) + 1) if i % 1 == 0],
                        # tickformat="05d",
                        tickfont=dict(size=14, color="black"), 
                        exponentformat='none'
                    ),
                    yaxis=dict(
                        tickmode='auto',nticks =10,
                        tickfont=dict(size=16, color="black")
                    ),
                    autosize=False,
                    width=800,
                    height=600
                )

                # Display the plot
                st.plotly_chart(fig, use_container_width=True)

                               
                if st.sidebar.button('Get AV file of this data'): 
                    # Create the CSV data
                    # Create the CSV header and data
                    header, data_df = create_csv_data_AV(min_x, max_x-1, 
                                                    st.session_state.selected_svids, st.session_state.selected_frequency1,
                                                    st.session_state.selected_frequency2, df4_filtered)

                    # Convert to CSV
                    csv_AV = convert_to_csv(header, data_df)

                    # Create download button
                    st.sidebar.download_button(
                        label="Download AV result",
                        data=csv_AV,
                        file_name="All_in_view_result.csv",
                        mime="text/csv",
                    )

    else:
        st.error("No overlap of selected data to work up on")

# Add a spacer to push the contact info to the bottom
st.sidebar.write("")  # This line adds some space
st.sidebar.write("")  # Add as many as needed to push the content down
st.sidebar.write("")   

# contact information at the bottom of the sidebar
st.sidebar.markdown('---')  # Add a horizontal line for separation
st.sidebar.markdown('**Contact Information**')
st.sidebar.text('Mr/Ms XYZ')
st.sidebar.text('Email: XYZ@bipm.org')
   
