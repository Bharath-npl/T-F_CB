
import streamlit as st
import pandas as pd
import os
import csv
import warnings
# import cv2
import io 
from io import StringIO
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
import base64

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
Avg_refsys_Rx1 = pd.DataFrame()
Avg_refsys_Rx2 = pd.DataFrame()


st.sidebar.image("https://www.bipm.org/documents/20126/27072194/Logo+BIPM+blue.png/797bb4e6-8cfb-5d78-b480-e460ad5eeec2?t=1583920059345", width=200)
#One line of gap 
st.sidebar.write("")
#IEEE UFFSC logo
st.sidebar.image("https://www.fusfoundation.org/images/IEEE-UFFC.jpg", width=200)

st.sidebar.header("Time & Frequency Capacity Building")


st.sidebar.header("CV & AV Time Transfer")


def display_material(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
    # Embedding PDF in HTML using iframe
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1400" height="1000" type="application/pdf"></iframe>'
    
    st.markdown(pdf_display, unsafe_allow_html=True)



def display_manual(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
    # Embedding PDF in HTML using iframe
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1400" height="1000" type="application/pdf"></iframe>'
    
    st.markdown(pdf_display, unsafe_allow_html=True)

# if st.sidebar.checkbox('Time transfer through GNSS'):
    # display_material('https://github.com/Bharath-npl/T-F_CB/blob/main/Expert_presentation.pdf')
# else:
#     st.write("PDF is hidden. Check the box to view it.")

# if st.sidebar.checkbox('User manual'):
    # display_material('https://github.com/Bharath-npl/T-F_CB/blob/main/User_manual_cggtts.pdf')


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
        valid_filenames01 = []
         
        for each_file in files_01:
            all_dataframes = []
            filename = each_file.name
            # Read uploaded file contents directly into memory
            file_content = each_file.read().decode()

            # Split the file into lines
            lines = file_content.split('\n')

            # Check if the first line starts with CGGTTS or GGTTS
            if not (lines[0].startswith("CGGTTS") or lines[0].startswith("GGTTS")):
                st.error(f"File format not supported for file: {filename}")
                continue  # Skip to the next file
            
            # st.write(f"File: {filename}")
            # Add the valid filename to the list
            valid_filenames01.append(filename)

            data_after_head = []
            # Flag to indicate if we are currently inside a header block
            inside_header = False
            frc_is_at = None
            prev_line = None  # Variable to keep track of the previous line
            
            for line in lines:
                # find the position of the FRC in the line 
                if "hhmmss  s  .1dg .1dg    .1ns" in line and prev_line:
                    frc_position = prev_line.find('FRC')
                    if frc_position != -1:
                        frc_is_at = frc_position
                
                # Start of the header
                if line.startswith("CGGTTS")or line.startswith("GGTTS"):
                    inside_header = True
                    if "=" in line:
                        Rx1_version = line.split('=')[1].strip()
                        # Do something with Rx1_version
                    else:
                        print("Problem in reading CGGTTS version, please add '=' before version number as per standard format")
                
                if line.startswith("REF=") or line.startswith("REF ="):
                    Receiever1 = line.split('=')[1].strip()
                
                if line.startswith("LAB=") or line.startswith("LAB ="):
                    LAB = line.split('=')[1].strip()

                # If we're not inside a header, process the line as data
                elif not inside_header:
                    data_after_head.append(line)

                # End of the header
                if "hhmmss  s  .1dg .1dg    .1ns" in line:
                    inside_header = False
                    
                prev_line = line  # Update the prev_line with the current line

            # Create DataFrame from the data list
            data_rows = []

            for line in data_after_head:
                if line.strip():  # Skip empty lines
                    # Extract the columns based on their fixed positions
                    data_row = {
                        'SAT': line[0:3].strip(),
                        'CL': line[4:6].strip(),
                        'MJD': line[7:12].strip(),
                        'STTIME': line[13:19].strip(),
                        'TRKL': line[20:24].strip(),
                        'ELV': line[25:28].strip(),
                        'AZTH': line[29:33].strip(),
                        'REFSV': line[34:45].strip(),
                        'SRSV': line[46:52].strip(),
                        'REFSYS': line[53:64].strip(),
                        'REF': Receiever1,
                        'Version':Rx1_version,
                        'LAB': LAB                                                
                    }

                    # Use the 'FRC' position if found
                    if frc_is_at is not None and len(line) > frc_is_at + 2:
                        data_row['FRC'] = line[frc_is_at:frc_is_at + 3].strip()
                    else:
                        # if it is CGGTTS version 1.0 there is no FRC column in the data format but the data is of L1C
                        data_row['FRC'] = "L1C"

                    data_rows.append(data_row)

            # Create DataFrame from the data list
            df_split = pd.DataFrame(data_rows)

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
            df_01['REF'] = df_split['REF'].astype(str)
            df_01['Version'] = df_split['Version'].astype(str)
            df_01['LAB'] = df_split['LAB'].astype(str)
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

        # Write the list of valid filenames in a row
        if valid_filenames01:
            st.write(f"Files uploaded: {', '.join(valid_filenames01)}")
        else:
            st.write("No valid files found.")

        
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
    st.session_state['REF01'] = ', '.join(map(str, processed_data1['REF'].dropna().unique()))
    st.session_state['LAB1'] = ' '.join(map(str, processed_data1['LAB'].dropna().unique()))

    
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


def create_csv_data_Rx1(starting_mjd, ending_mjd, selected_data, frequency1):
    # Creating DataFrame for data section
    # Format the 'MJD' column to have only 5 digits after the decimal
    selected_data["MJD"] = selected_data["MJD"].apply(lambda x: f"{x:.5f}")
    
    data_Rx1_df = pd.DataFrame({
        'MJD': selected_data["MJD"],
        'Refsys (ns)': selected_data['REFSYS']
    })

    # Creating header information
    header_Rx1_info = (
        f"# {st.session_state['REF01']} REFSYS data: Each point corresponds to average of all visible satellite REFSYS values at each epoch  \n"
        f"# Start MJD: {starting_mjd}\n"
        f"# End MJD: {ending_mjd}\n"
        f"# Frequency: {frequency1}\n"
        f"# Lab:{st.session_state['LAB1']}\n")
                

    return header_Rx1_info, data_Rx1_df


def create_csv_data_Rx2(starting_mjd, ending_mjd, selected_data, frequency2):
    # Creating DataFrame for data section
    # Format the 'MJD' column to have only 5 digits after the decimal
    selected_data["MJD"] = selected_data["MJD"].apply(lambda x: f"{x:.5f}")
    
    data_Rx2_df = pd.DataFrame({
        'MJD': selected_data["MJD"],
        'Refsys (ns)': selected_data['REFSYS']
    })

    # Creating header information
    header_Rx2_info = (
        f"# {st.session_state['REF02']} REFSYS data: Each point corresponds to average of all visible satellite REFSYS values at each epoch  \n"
        f"# Start MJD: {starting_mjd}\n"
        f"# End MJD: {ending_mjd}\n"
        f"# Frequency: {frequency2}\n"
        f"# Lab:{st.session_state['LAB2']}\n")

    return header_Rx2_info, data_Rx2_df
    

# Function to convert header and DataFrame to CSV for download
def convert_to_csv(header, df):
    output = StringIO()
    output.write(header)
    df.to_csv(output,sep='\t', index=False, header=True)
    return output.getvalue()



def plot_data1(frequency1):
    # Filter the MJD-filtered data based on the frequency
    df1_data_filtered = st.session_state['sel_MJD_df_01'][st.session_state['sel_MJD_df_01']['FRC'] == frequency1]
    # st.write(f"Filtered data: \n {df1_data_filtered}")
    st.session_state["sel_MJD_FRC_01"] = df1_data_filtered
    

    if not df1_data_filtered.empty:
        Avg_refsys_Rx1 = (df1_data_filtered.groupby("MJD")["REFSYS"].mean().reset_index())
        Avg_refsys_Rx1["REFSYS"] = (Avg_refsys_Rx1["REFSYS"]*0.1).round(2)
        Data01_stdev = Avg_refsys_Rx1["REFSYS"].std()

        # Select the start and end MJD from the user selection 
        min_Rx1 = math.floor(min(Avg_refsys_Rx1["MJD"]))
        max_Rx1 = math.ceil(max(Avg_refsys_Rx1["MJD"]))
        
        # Create a scatter plot with Plotly
        fig = go.Figure()

        # Add scatter plot of data points
        fig.add_trace(go.Scatter(
            x=Avg_refsys_Rx1["MJD"], 
            y=Avg_refsys_Rx1["REFSYS"], 
            mode='markers',
            name='REFSYS'
        ))

        # Add a standard deviation annotation to the plot 
        fig.add_annotation(xref='paper', yref='paper', x=1, y=1, text=f"Std Dev: {Data01_stdev:.2f} ns",
        showarrow=False, font=dict(size=18, color="black"),
        xanchor='right', yanchor='top')

           
        # Update layout for better presentation

        fig.update_layout(
            title=f"{st.session_state['REF01']} - GNSS(time) at Lab: {st.session_state['LAB1']} through {st.session_state.selected_frequency1}. (Each point correponds to Average of all satellite refsys per epoch)",
            xaxis_title="MJD",
            yaxis_title="REFSYS (ns)",
            yaxis=dict(tickmode='auto', nticks =10),
            xaxis =dict(tickformat=".2f",  # Set the tick values as original MJD values
                    # ticktext=formatted_mjd,  # Set the formatted MJD values as tick labels,
                    tickfont= dict(size=14, color ="black"), 
                    exponentformat ='none'))
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Create the CSV header and data
        headerRx1, data_dfRx1 = create_csv_data_Rx1(min_Rx1, max_Rx1, Avg_refsys_Rx1,
                                        st.session_state.selected_frequency1)
 
        
        csv_Rx1 = convert_to_csv(headerRx1, data_dfRx1)
                
        # Create a download button
        # csv = convert_Rx1_to_csv(Avg_refsys_Rx1)
        st.download_button(
            label="Download REFSYS data",
            data=csv_Rx1,
            file_name='Refsys01.csv',
            mime='text/csv',
        )


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
    submitted2 = st.form_submit_button("Submit2")

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
        valid_filenames02 = []
                 
        for each_file in files_02:
            all_dataframes = []
            filename = each_file.name

            # Read uploaded file contents directly into memory
            file_content = each_file.read().decode()

            # Split the file into lines
            lines = file_content.split('\n')

            # Check if the first line starts with CGGTTS or GGTTS
            if not (lines[0].startswith("CGGTTS") or lines[0].startswith("GGTTS")):
                st.error(f"File format not supported for file: {filename}")
                continue  # Skip to the next file
            
            # st.write(f"File: {filename}")
            valid_filenames02.append(filename)


            data_after_head = []
            # Flag to indicate if we are currently inside a header block
            inside_header = False
            prev_line = None
            frc_is_at =None

            for line in lines:
                               
                # Start of the header
                if line.startswith("CGGTTS")or line.startswith("GGTTS"):
                    inside_header = True
                    if "=" in line:
                        Rx2_version = line.split('=')[1].strip()
                        # Do something with Rx1_version
                    else:
                        print("Problem in reading CGGTTS version, please add '=' before version number as per standard format")
                    
                # Find the position of the FRC in the line 
                if "hhmmss  s  .1dg .1dg    .1ns" in line and prev_line:
                    frc_position = prev_line.find('FRC')
                    if frc_position != -1:
                        frc_is_at = frc_position

                if line.startswith("REF=") or line.startswith("REF ="):
                    Receiever2 = line.split('=')[1].strip()

                if line.startswith("LAB=") or line.startswith("LAB ="):
                    LAB = line.split('=')[1].strip()

                # If we're not inside a header, process the line as data
                elif not inside_header:
                    data_after_head.append(line)

                # End of the header
                if "hhmmss  s  .1dg .1dg    .1ns" in line:
                    inside_header = False
                prev_line = line  # Update the prev_line with the current line

            # Create DataFrame from the data list
            data_rows = []

            for line in data_after_head:
                if line.strip():  # Skip empty lines
                    # Extract the columns based on their fixed positions
                    data_row = {
                        'SAT': line[0:3].strip(),
                        'CL': line[4:6].strip(),
                        'MJD': line[7:12].strip(),
                        'STTIME': line[13:19].strip(),
                        'TRKL': line[20:24].strip(),
                        'ELV': line[25:28].strip(),
                        'AZTH': line[29:33].strip(),
                        'REFSV': line[34:45].strip(),
                        'SRSV': line[46:52].strip(),
                        'REFSYS': line[53:64].strip(),
                        'REF': Receiever2,
                        'Version':Rx1_version, 
                        'LAB': LAB}
                    
                    # Use the 'FRC' position if found
                    if frc_is_at is not None and len(line) > frc_is_at + 2:
                        data_row['FRC'] = line[frc_is_at:frc_is_at + 3].strip()
                    else:
                        data_row['FRC'] = "No_FRC"

                    data_rows.append(data_row)

            # Create DataFrame from the data list
            df_split = pd.DataFrame(data_rows)

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
            df_02['Version'] = df_split['Version'].astype(str)
            df_02['LAB'] = df_split['LAB'].astype(str)
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

        # Write the list of valid filenames in a row
        if valid_filenames02:
            st.write(f"Files uploaded: {', '.join(valid_filenames02)}")
        else:
            st.write("No valid files found.")

       
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
    st.session_state['REF02'] = ', '.join(map(str, processed_data2['REF'].dropna().unique()))
    st.session_state['LAB2'] = ' '.join(map(str, processed_data2['LAB'].dropna().unique()))


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


def plot_data2(frequency2):
    # Filter the MJD-filtered data based on the frequency
    df2_data_filtered = st.session_state['sel_MJD_df_02'][st.session_state['sel_MJD_df_02']['FRC'] == frequency2]
    st.session_state["sel_MJD_FRC_02"] = df2_data_filtered
    # st.write(f"Filtered data: \n {df1_data_filtered}")

    #     st.line_chart(Avg_refsys_CV.set_index("MJD")[["REFSYS", "Avg"]])
    if not df2_data_filtered.empty:
        Avg_refsys_Rx2 = (df2_data_filtered.groupby("MJD")["REFSYS"].mean().reset_index())
        Avg_refsys_Rx2["REFSYS"] = (Avg_refsys_Rx2["REFSYS"]*0.1).round(2)
        Data02_stdev = Avg_refsys_Rx2["REFSYS"].std()

        # st.markdown(f"## Receiver 2 Average REFSYS: {frequency2}")
        # Select the start and end MJD from the user selection 
        min_Rx2 = math.floor(min(Avg_refsys_Rx2["MJD"]))
        max_Rx2 = math.ceil(max(Avg_refsys_Rx2["MJD"]))

        # Create a scatter plot with Plotly
        fig = go.Figure()

        # Add scatter plot of data points
        fig.add_trace(go.Scatter(
            x=Avg_refsys_Rx2["MJD"], 
            y=Avg_refsys_Rx2["REFSYS"], 
            mode='markers',
            name='REFSYS'
        ))

        # Add a standard deviation annotation to the plot 
        fig.add_annotation(xref='paper', yref='paper', x=1, y=1, text=f"Std Dev: {Data02_stdev:.2f} ns",
        showarrow=False, font=dict(size=18, color="black"),
        xanchor='right', yanchor='top')

        # Update layout for better presentation
        fig.update_layout(
            title=f"{st.session_state['REF02']} - GNSS(time) at Lab: {st.session_state['LAB2']} through {st.session_state.selected_frequency2} . (Each point correponds to Average of all satellite refsys per epoch)",
            xaxis_title="MJD",
            yaxis_title="REFSYS (ns)",
            yaxis=dict(tickmode='auto', nticks =10),
            xaxis =dict(tickformat=".2f", tickfont= dict(size=14, color ="black"), exponentformat ='none')
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Create the CSV header and data
        headerRx2, data_dfRx2 = create_csv_data_Rx2(min_Rx2, max_Rx2, Avg_refsys_Rx2,
                                        st.session_state.selected_frequency2)
 
        
        csv_Rx2 = convert_to_csv(headerRx2, data_dfRx2)
                
        # Create a download button
        # csv = convert_Rx1_to_csv(Avg_refsys_Rx1)
        st.download_button(
            label="Download REFSYS data",
            data=csv_Rx2,
            file_name='Refsys02.csv',
            mime='text/csv',
        )

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
            plot_data2(selected_frequency2)
        else:
            st.error("No valid frequencies available for selection.")
    

# Function to create the DataFrame for CSV
def create_CVSV_data_CSV(starting_mjd, ending_mjd, SVids, frequency1, frequency2, Elv_mask, selected_data):
    # Convert 'MJD' to float and then to string with 5 decimal places
    selected_data['MJD'] = selected_data['MJD'].astype(float).apply(lambda x: f"{x:.5f}")

    # Function to concatenate values into a comma-separated string
    concat_values = lambda x: ','.join(x.astype(str))

    # Function to concatenate rounded values
    concat_rounded_values = lambda x: ','.join(x.round(2).astype(str))
    
    # Function to count the number of satellites
    count_satellites = lambda x: len(x)

    # Group by 'MJD' and aggregate SAT and CV_diff columns
    aggregated_data = selected_data.groupby('MJD').agg({
        'SAT': [count_satellites, concat_values],
        'CV_diff': concat_rounded_values
    }).reset_index()

    # Rename columns for clarity
    aggregated_data.columns = ['MJD', 'Num_CV_Satellites', 'PRNs', 'Refsys_difference(ns)']

        # Creating header information
    header_CV_info = (
        f"# Common satellites in view and their Time Transfer between {st.session_state['LAB1']} and {st.session_state['LAB2']}  \n"
        f"# Start MJD: {starting_mjd}\n"
        f"# End MJD: {ending_mjd}\n"
        f"# Frequency selected for comparision in receiver 1: {frequency1}\n"
        f"# Frequency selected for comparision in receiver 2: {frequency2}\n"
        f"# Elevation mask applied: {Elv_mask} degrees\n"
        f"# Selected satellites for time transfer: {', '.join(sorted(SVids))}\n")
    

    return header_CV_info, aggregated_data


# Function to create the DataFrame for CSV
def create_csv_data_CV(starting_mjd, ending_mjd, SVids, frequency1, frequency2, Elv_mask, selected_data):
    # Creating DataFrame for data section
    # x=df3_filtered["MJD_time"], 
    #                 y=df3_filtered["CV_avg_diff"]
    selected_data["MJD"] = selected_data["MJD"].apply(lambda x: f"{x:.5f}")
    data_df = pd.DataFrame({
        'MJD': selected_data["MJD"],
        'CV_difference (ns)': selected_data['CV_avg_diff'].round(2)
    })

    # Creating header information
    header_CV_info = (
        f"#Common View Time Transfer Link Performance \n"
        f"#Start MJD: {starting_mjd}\n"
        f"#End MJD: {ending_mjd}\n"
        f"#Frequency selected for comparision in receiver 1: {frequency1}\n"
        f"#Frequency selected for comparision in receiver 2: {frequency2}\n"
        f"#Elevation mask applied: {Elv_mask} degrees\n"
        f"#Selected satellites for time transfer: {', '.join(sorted(SVids))}\n"
    )

    return header_CV_info, data_df


# Function to create the DataFrame for CSV
def create_csv_data_AV(starting_mjd, ending_mjd, SVids, frequency1, frequency2, Elv_mask, selected_data):
    # Creating DataFrame for data section
    # Format the 'MJD' column to have only 5 digits after the decimal
    selected_data["MJD_time"] = selected_data["MJD_time"].apply(lambda x: f"{x:.5f}")
    
    data_AV_df = pd.DataFrame({
        'MJD': selected_data["MJD_time"],
        'AV_difference (ns)': selected_data['AV_diff'].round(2)
    })

    # Creating header information
    header_AV_info = (
        f"#All-in-View link between [{st.session_state['REF01']} - {st.session_state['REF02']}] \n"
        f"#Start MJD: {starting_mjd}\n"
        f"#End MJD: {ending_mjd}\n"
        f"#Frequency selected for comparision in receiver 1: {frequency1}\n"
        f"#Frequency selected for comparision in receiver 2: {frequency2}\n"
        f"#Elevation mask applied: {Elv_mask} degrees\n"
        f"#Selected Satellites for time transfer: {', '.join(SVids)}\n"
    )

    return header_AV_info, data_AV_df
    

data1_avail =0
data2_avail =0
# BIPM Logo

st.sidebar.header("Common-View Analysis")
plot_CV = st.sidebar.button("Plot Common-View", key= 'Common_view')

# CV_PRNS = st.sidebar.button("Plot Common PRNs", key= 'Common_PRNs')


st.sidebar.header("All-in-View Analysis")
plot_AV = st.sidebar.button("Plot All-in-View", key= 'All_in_view')


# Check conditions only when plot buttons are clicked
# if (st.session_state.plot_CV_clicked or st.session_state.plot_AV_clicked):
#     if not submitted1 or not submitted2:
#         st.error("One of the receiver data is not uploaded yet. Don't forget to press the Submit button after uploading the files.")



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

def process_plot_CV(df1, df2, unique_MJD_times, selected_svids, unique_SVIDs, Elv_Mask):
    # Filter based on ELV values
    df1 = df1[df1['ELV'] >= (Elv_Mask * 10)]
    df2 = df2[df2['ELV'] >= (Elv_Mask * 10)]
    
    # Ensure 'ALL' in selected_svids is handled correctly
    if 'ALL' in selected_svids:
        selected_svids = unique_SVIDs

    # Filter the DataFrames
    df1_filtered = df1[df1["MJD"].isin(unique_MJD_times) & df1["SAT"].isin(selected_svids)]
    df2_filtered = df2[df2["MJD"].isin(unique_MJD_times) & df2["SAT"].isin(selected_svids)]

    # Merge the filtered DataFrames
    merged_df = pd.merge(df1_filtered, df2_filtered, on=['SAT', 'MJD'], suffixes=('_df1', '_df2'))

    # Compute differences
    merged_df['CV_diff'] = (merged_df['REFSYS_df1'] - merged_df['REFSYS_df2']) * 0.1

    # Group by 'MJD' for the result
    result = merged_df.groupby('MJD').agg({'CV_diff': ['mean', 'count']})
    result.columns = ['CV_avg_diff', 'count']
    result.reset_index(inplace=True)

    # Handle missing MJD times
    missing_session = list(set(unique_MJD_times) - set(result['MJD']))

    # Create a new dataframe for CV_SV
    group = merged_df.groupby('MJD')
    CV_SV = pd.DataFrame({
        'MJD': group.groups.keys(),
        'SAT_list': group['SAT'].apply(list),
        'CV_diff_list': group['CV_diff'].apply(list)
    })

    return result, missing_session, CV_SV


def process_plot_AV(df1, df2, selected_svids, unique_SVIDs, unique_MJD_times, Elv_Mask):
    global print_once
     # Filter based on ELV values
    df1 = df1[df1['ELV'] >= Elv_Mask*10]
    df2 = df2[df2['ELV'] >= Elv_Mask*10]
    
    # Handling 'ALL' selection
    svids_to_use = unique_SVIDs if 'ALL' in selected_svids or len(selected_svids) == len(unique_SVIDs) else selected_svids

    # Remove 'ALL' if individual SV_ids are selected
    if len(selected_svids) < len(unique_SVIDs) + 1:
        svids_to_use = [svid for svid in selected_svids if svid != 'ALL']
    
    # Calculate inverse cosine squared values
    df1['sin2'] =  np.sin(np.radians(df1['ELV'] * 0.1))**2
    df2['sin2'] =  np.sin(np.radians(df2['ELV'] * 0.1))**2

    if unique_MJD_times:
        AV_data = []
        for unique_time in unique_MJD_times:
            df1_filtered = df1[df1["MJD"] == unique_time]
            df2_filtered = df2[df2["MJD"] == unique_time]

            # Apply outlier filter for REFSYS values
            mean_df1 = df1_filtered['REFSYS'].mean()
            std_df1 = df1_filtered['REFSYS'].std()
            mean_df2 = df2_filtered['REFSYS'].mean()
            std_df2 = df2_filtered['REFSYS'].std()

            df1_filtered = df1_filtered[np.abs(df1_filtered['REFSYS'] - mean_df1) <= 1.5 * std_df1]
            df2_filtered = df2_filtered[np.abs(df2_filtered['REFSYS'] - mean_df2) <= 1.5 * std_df2]

            if not df1_filtered.empty and not df2_filtered.empty:
                
                # Filter data based on selected satellites
                condition1 = df1_filtered["SAT"].isin(svids_to_use)
                condition2 = df2_filtered["SAT"].isin(svids_to_use)

                # Normalisation of weights values
                Norm_weigth1 = 1/np.sum(df1_filtered.loc[condition1, 'sin2'])
                Norm_weight2 = 1/np.sum(df2_filtered.loc[condition2, 'sin2'])


                if condition1.any() and condition2.any():
                    
                    weighted_sum_df1 = (df1_filtered.loc[condition1, 'REFSYS'] * Norm_weigth1 * df1_filtered.loc[condition1, 'sin2']).sum() * 0.1
                    weighted_sum_df2 = (df2_filtered.loc[condition2, 'REFSYS'] * Norm_weight2 * df2_filtered.loc[condition2, 'sin2']).sum() * 0.1


                    res1 = df1_filtered['REFSYS'] - weighted_sum_df1
                    std_df1 = np.sqrt(np.sum(res1**2)/len(res1))
                    res2 = df2_filtered['REFSYS'] - weighted_sum_df2
                    std_df2 = np.sqrt(np.sum(res2**2)/len(res2))


                    df1_filtered = df1_filtered[np.abs(res1) <= 1.5 * std_df1]
                    df2_filtered = df2_filtered[np.abs(res2) <= 1.5 * std_df2]

                    Norm_weigth1 = 1/np.sum(df1_filtered.loc[condition1, 'sin2'])
                    Norm_weight2 = 1/np.sum(df2_filtered.loc[condition2, 'sin2'])

                    weighted_sum_df1 = (df1_filtered.loc[condition1, 'REFSYS'] * Norm_weigth1 * df1_filtered.loc[condition1, 'sin2']).sum() * 0.1
                    weighted_sum_df2 = (df2_filtered.loc[condition2, 'REFSYS'] * Norm_weight2 * df2_filtered.loc[condition2, 'sin2']).sum() * 0.1

                    AV_diff_refsys = weighted_sum_df1 - weighted_sum_df2
                    new_row = {'MJD_time': unique_time, 'AV_diff': round(AV_diff_refsys, 2) if AV_diff_refsys else None}
                    AV_data.append(new_row)
                    
                    # Start of the code for printnting the first epoch of AV
                    #**********************************************
                    # if print_once ==1:
                    #     data = {
                    #         'SAT': df1_filtered.loc[condition1, 'SAT'],
                    #         'MJD': [unique_time] * len(df1_filtered.loc[condition1]),
                    #         'Refsys (ns)': df1_filtered.loc[condition1, 'REFSYS']*0.1,
                    #         'Elv': df1_filtered.loc[condition1, 'ELV']/10,
                    #         'Residual':  np.abs(df1_filtered.loc[condition1, 'REFSYS'] - mean_df1),  
                    #         # 'sine square': df1_filtered.loc[condition1, 'sin2'],
                    #         'Weight': [Norm_weigth1] * df1_filtered.loc[condition1, 'sin2'],
                    #         'Weighted refsys (ns)': df1_filtered.loc[condition1, 'REFSYS']*Norm_weigth1*df1_filtered.loc[condition1, 'sin2']*0.1
                    #     }

                    #     # Create a DataFrame
                    #     # st.write(f"Length of the column : {condition1}")
                    #     df_to_display = pd.DataFrame(data)
                    #     # Display the DataFrame as a table in Streamlit
                    #     st.write("Overview of All_in_View at FIRST epoch of 1st receiver")
                    #     st.table(df_to_display)
                    #     print_once = 2
                    #**************************************
                    # End of the code for printing the first epoch of AV

            else:
                # Handle the case when one of the filtered DataFrames is empty
                AV_data.append({'MJD_time': unique_time, 'AV_diff': None})

                               
            # Print the required caluclated infromatio in the screen
                            
            # st.write(f"Sum of the weights after normalisation at the above (1st) epoch: {round(sum([Norm_weigth1] * df1_filtered.loc[condition1, 'sin2']),2)}")
            # st.write(f"Sum of the weighted refsys of 1st data set at the above (1st) epoch: {round(weighted_sum_df1,2)}")
            # st.write(f"sum of the weighted refsys of 2nd data set at the above (1st) epoch: {round(weighted_sum_df2,2)}")
            # st.write(f"AV difference at the above (1st) epoch: {round(weighted_sum_df1 - weighted_sum_df2,2)}")
            
            # st.write(f"sum of the weights: {K_nd_Inverse_cos.sum()}")


        return pd.DataFrame(AV_data, columns=['MJD_time', 'AV_diff'])
    else:
        st.error("Files don't belong to the same time period")
        return pd.DataFrame()




if 'sel_MJD_FRC_01' in st.session_state and 'sel_MJD_FRC_02' in st.session_state:
    
    if 'df1_mjd' not in st.session_state:
        st.session_state.df1_mjd = pd.DataFrame() 

    if 'df2_mjd' not in st.session_state:
        st.session_state.df2_mjd = pd.DataFrame()

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
        Common_MJD_times = sorted(set(st.session_state.df1_mjd["MJD"]).intersection(set(st.session_state.df2_mjd["MJD"])))

        all_common_svids = set()
        df1_mjd_01 =[]
        df2_mjd_02 =[]
        missing_session =[]
        filtered_data01 = pd.DataFrame()
        filtered_data02 = pd.DataFrame()
        st.session_state.df3_filtered = pd.DataFrame()
        st.session_state.df4_filtered = pd.DataFrame()


        for mjd_time in unique_MJD_times:
            # Filter dataframes for the current mjd_time
            df1_mjd_01 = st.session_state.df1_mjd[st.session_state.df1_mjd["MJD"] == mjd_time]
            df2_mjd_02 = st.session_state.df2_mjd[st.session_state.df2_mjd["MJD"] == mjd_time]
            
                # Accumulate the filtered data
            filtered_data01 = pd.concat([filtered_data01, df1_mjd_01])
            filtered_data02 = pd.concat([filtered_data02, df2_mjd_02])
            # common_svids = set(df1_mjd_01["SAT"]) & set(df2_mjd_02["SAT"]) 

            all_svids = set(df1_mjd_01["SAT"]).union( set(df2_mjd_02["SAT"]))
          
            all_common_svids.update(all_svids)
            
         # Store the accumulated data in the session state
        st.session_state.df1_mjd_01 = filtered_data01
        st.session_state.df2_mjd_02 = filtered_data02

        # Convert set to list
        unique_SVIDs = list(all_common_svids)

        if unique_SVIDs:
            # print(f"Unique SAT in combined data: \n{unique_SVIDs}")
            # If the session_state.plot_data doesn't exist, initialize it to None
            if 'plot_CV_data' not in st.session_state:
                st.session_state.plot_CV_data = None    

            if 'CV_SV_data' not in st.session_state:
                st.session_state.CV_SV_data = None                     

            if 'plot_AV_data' not in st.session_state:
                st.session_state.plot_AV_data = None 
            # Sidebar options
            # st.sidebar.header("Common View Data")

                # Initialize selected_svids in session_state if not present
            if 'selected_svids' not in st.session_state:
                st.session_state.selected_svids = ['ALL']

            st.sidebar.markdown('---')  # Add a horizontal line for separation
            st.sidebar.header("Filters for CV & AV analysis")
            selected_svids = st.sidebar.multiselect(
                "Choose Satellites (PRN's)",
                options=['ALL'] + list(unique_SVIDs),
                # default=st.session_state.selected_svids,
                default =['ALL'],
                key= 12)  # Use the unique key here

               # Update the session state
            if 'ALL' in selected_svids or len(selected_svids) == len(unique_SVIDs):
                svids_to_use = unique_SVIDs
            else:
                svids_to_use = selected_svids
            
            # Set the default elevation mask in session_state 
            if 'elevation_mask' not in st.session_state:
                st.session_state.elevation_mask = 15.0

            # Elevation mask input
            elevation_mask = st.sidebar.number_input('Elevation Mask (0 to 90 degrees)',
                                                    min_value=0.0, 
                                                    max_value=90.0, 
                                                    value=15.0,  # default value
                                                    step=0.5)  # step size for increment/decrement

            # Update session state for elevation mask
            st.session_state.elevation_mask = elevation_mask

            # Update session state for selected svids
            st.session_state.selected_svids = svids_to_use
            # plot_button = st.sidebar.button("Plot CV")

            if plot_CV :
                # Replace the following with your actual DataFrame and variable names
                df1_CV = st.session_state.df1_mjd_01  # Replace with your actual DataFrame
                df2_CV = st.session_state.df2_mjd_02  # Replace with your actual DataFrame
                selected_svids = st.session_state.selected_svids  # Replace with your actual list of selected svids

                result_df, missing_sessions, cv_sv_df = process_plot_CV(df1_CV, df2_CV, unique_MJD_times, selected_svids, unique_SVIDs, st.session_state.elevation_mask)

                if not result_df.empty:
                    st.session_state.plot_CV_data = result_df[['MJD', 'CV_avg_diff']]
                else:
                    st.error("No COMMON data available for processing. Check if the two data sets belong to the same time period and same code of frequency selection ")               
                
                if not cv_sv_df.empty:
                        st.session_state.CV_SV_data = cv_sv_df
                # else:
                #     st.error("No COMMON data available for processing. Check if the two data sets belong to the same time period and same code of frequency selection ") 
                # if  not st.session_state.plot_data.empty:
                # Plotting 
            
            if st.session_state.CV_SV_data is not None and not st.session_state.CV_SV_data.empty:
                # Use the correct dataframe
                st.markdown('---')  # Add a horizontal line for separation
                
                df_cv_sv = st.session_state.CV_SV_data
                long_form = []

                # Unpacking the lists in df_cv_sv
                for _, row in df_cv_sv.iterrows():
                    for sat, diff in zip(row['SAT_list'], row['CV_diff_list']):
                        long_form.append({'MJD': row['MJD'], 'SAT': sat, 'CV_diff': diff})

                long_df = pd.DataFrame(long_form)

                # Initialize a figure
                fig = go.Figure()

                # Add scatter plot for each satellite
                for sat in long_df['SAT'].unique():
                    df_sat = long_df[long_df['SAT'] == sat]
                    fig.add_trace(go.Scatter(
                        x=df_sat['MJD'],
                        y=df_sat['CV_diff'],
                        mode='markers',
                        name=sat,  # Satellite name as legend entry
                        marker=dict(size=10)  # Increase marker size
                    ))

                # Set plot titles and labels
                fig.update_layout(
                    title=f"Common satellites in view between {st.session_state['LAB1']} ({st.session_state.selected_frequency1}) and {st.session_state['LAB2']} ({st.session_state.selected_frequency2}) <br> (Each point corresponds to time difference between refsys values for each COMMON Satellite in view at each epoch)",
                    title_font=dict(size=20, color="black"),
                    xaxis_title="MJD",
                    xaxis_title_font=dict(size=16, color="black"),
                    yaxis_title="Time difference (ns)",
                    yaxis_title_font=dict(size=16, color="black"),
                    xaxis=dict(tickformat=".2f",
                        tickfont=dict(size=14, color="black"),
                        exponentformat='none' 
                    ),
                    yaxis=dict(
                        tickmode='auto', nticks=10,
                        tickfont=dict(size=14, color="black")
                    ),
                    autosize=False,
                    width=800,
                    height=600
                )
                

                # Display the plot in Streamlit
                st.plotly_chart(fig, use_container_width=True)

                # Set x-axis range and filter rows of the dataframe
                min_mjd_time = long_df["MJD"].dropna().min()
                max_mjd_time = long_df["MJD"].dropna().max()

                # Now apply math.floor() and math.ceil()
                min_x = math.floor(min_mjd_time) if pd.notna(min_mjd_time) else None
                max_x = math.ceil(max_mjd_time) if pd.notna(max_mjd_time) else None

                header, data_df = create_CVSV_data_CSV(min_x, max_x-1, 
                                                    st.session_state.selected_svids, st.session_state.selected_frequency1,
                                                    st.session_state.selected_frequency2,st.session_state.elevation_mask, long_df)

                # Convert to CSV
                csv_CV = convert_to_csv(header, data_df)

                # Create download button
                st.download_button(
                    label="Download CV Satellite data",
                    data=csv_CV,
                    file_name="Common_View_Sat_data.csv",
                    mime="text/csv",
                )

                                
            if st.session_state.plot_CV_data is not None and not st.session_state.plot_CV_data.empty:
                df3 = st.session_state.plot_CV_data
                st.markdown('---')  # Add a horizontal line for separation
                
                if st.session_state.selected_frequency1 != st.session_state.selected_frequency2:
                    st.error("Caution: The selected frequenices are different")

                # User inputs for the y-axis range
                col1, col2 = st.columns(2)
                with col1:
                    user_start_y = st.number_input("Lower Outlier limit", min_value=float(df3["CV_avg_diff"].min()), max_value=float(df3["CV_avg_diff"].max()), value=float(df3["CV_avg_diff"].min()))
                with col2:
                    user_end_y = st.number_input("Upper Outlier limit", min_value=float(df3["CV_avg_diff"].min()), max_value=float(df3["CV_avg_diff"].max()), value=float(df3["CV_avg_diff"].max()))

                # Filter the data based on user selection and calculate mean
                df3_filtered = df3[(df3["CV_avg_diff"] >= user_start_y) & (df3["CV_avg_diff"] <= user_end_y)]
                std_dev = df3_filtered["CV_avg_diff"].std()

                            
                # Set x-axis range and filter rows of the dataframe
                min_mjd_time = df3["MJD"].dropna().min()
                max_mjd_time = df3["MJD"].dropna().max()

                # Now apply math.floor() and math.ceil()
                min_x = math.floor(min_mjd_time) if pd.notna(min_mjd_time) else None
                max_x = math.ceil(max_mjd_time) if pd.notna(max_mjd_time) else None

                # Add a check if min_x and max_x are None
                if min_x is None or max_x is None:
                    print("Error: MJD_time column contains only NaN values")
                    # Handle the error appropriately
                else:
                # Create scatter plot
                    fig = go.Figure()
                            
                    # Add scatter plot of data points
                    fig.add_trace(go.Scatter(
                        x=df3_filtered["MJD"], 
                        y=df3_filtered["CV_avg_diff"], 
                        mode='markers',
                        name='CV_avg_diff',
                        marker=dict(size=10)  # Increase marker size
                    ))
        
                    # Add a standard deviation annotation to the plot 
                    fig.add_annotation(xref='paper', yref='paper', x=1, y=1, text=f"Std Dev: {std_dev:.2f} ns",
                    showarrow=False, font=dict(size=18, color="black"),
                    xanchor='right', yanchor='top')

                    # Set plot titles and labels with increased font size and black color
                    fig.update_layout(
                        # title=f"Common - View link between [{st.session_state['REF01']} - {st.session_state['REF02']}] during (MJD: {min_x} - {max_x-1})",
                        title=f"Common - View link between {st.session_state['LAB1']} ({st.session_state.selected_frequency1}) and {st.session_state['LAB2']}({st.session_state.selected_frequency1}) <br> (Each point is average of differences between refsys values of all common satellites at each epoch)",
                        title_font=dict(size=20, color="black"),
                        xaxis_title="MJD",
                        xaxis_title_font=dict(size=16, color="black"),
                        yaxis_title="Time difference (ns)",
                        yaxis_title_font=dict(size=16, color="black"),
                        xaxis=dict(tickformat=".2f",
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
                                
                # if st.sidebar.button('Get CV file of this data'): 
                    # Create the CSV data
                    # Create the CSV header and data
                    header, data_df = create_csv_data_CV(min_x, max_x-1, 
                                                    st.session_state.selected_svids, st.session_state.selected_frequency1,
                                                    st.session_state.selected_frequency2,st.session_state.elevation_mask, df3_filtered)

                    # Convert to CSV
                    csv_CV = convert_to_csv(header, data_df)

                    # Create download button
                    st.download_button(
                        label="Download CV data",
                        data=csv_CV,
                        file_name="Common_View_data.csv",
                        mime="text/csv",
                    )
                        

            if 'selected_svids' not in st.session_state:
                st.session_state.selected_svids = ['ALL']

            if plot_AV:

                df1_AV = st.session_state.df1_mjd_01  # Replace with your actual DataFrame
                df2_AV = st.session_state.df2_mjd_02  # Replace with your actual DataFrame
                
                # Convert 'MJD' columns to sets
                set_mjd_df1 = set(df1_AV['MJD'])
                set_mjd_df2 = set(df2_AV['MJD'])

                # Find the intersection of the two sets
                common_mjd = set_mjd_df1.intersection(set_mjd_df2)

                # Check if there is any common 'MJD' and process
                if common_mjd:
                    result_df02 = process_plot_AV(df1_AV, df2_AV, st.session_state.selected_svids, unique_SVIDs, unique_MJD_times, st.session_state.elevation_mask)
                else:
                    # Handle the case where there is no common MJD
                    result_df02 = pd.DataFrame() 
                
                if not result_df02.empty:
                    st.session_state.plot_AV_data = result_df02
                    
                else:
                    st.error("No COMMON data available for processing. Check if the two data sets belong to the same time period and same code of frequency selection ") 
                    
            
            if 'plot_AV_data' in st.session_state and st.session_state.plot_AV_data is not None: 
                
                df4 = st.session_state.plot_AV_data
                                
                st.markdown('---')  # Add a horizontal line for separation
                # print(df4)
                # User inputs for the y-axis range
                col1, col2 = st.columns(2)
                with col1:
                    user_start_y = st.number_input("Lower Outlier limit", min_value=float(df4["AV_diff"].min()), max_value=float(df4["AV_diff"].max()), value=float(df4["AV_diff"].min()))
                with col2:
                    user_end_y = st.number_input("Upper Outlier limit", min_value=float(df4["AV_diff"].min()), max_value=float(df4["AV_diff"].max()), value=float(df4["AV_diff"].max()))

                # Filter the data based on user selection and calculate mean
                df4_filtered = df4[(df4["AV_diff"] >= user_start_y) & (df4["AV_diff"] <= user_end_y)]
                
                std_dev = df4_filtered["AV_diff"].std()

                # Set x-axis range
                min_x = math.floor(min(df4["MJD_time"]))
                max_x = math.ceil(max(df4["MJD_time"]))
               
                if min_x is not None and max_x is not None:
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

                    # Add a standard deviation annotation to the plot 
                    fig.add_annotation(xref='paper', yref='paper', x=1, y=1, text=f"Std Dev: {std_dev:.2f} ns",
                    showarrow=False, font=dict(size=18, color="black"),
                    xanchor='right', yanchor='top')

                    # Set plot titles and labels with increased font size and black color
                    fig.update_layout(
                        title=f"All-in-View link between {st.session_state['REF01']} ({st.session_state.selected_frequency1}) and {st.session_state['REF02']} ({st.session_state.selected_frequency2}) <br> (Each point is the difference of the average refsys of all satellites in view)",
                        title_font=dict(size=20, color="black"),
                        xaxis_title="MJD",
                        xaxis_title_font=dict(size=16, color="black"),
                        yaxis_title="Time difference (ns)",
                        yaxis_title_font=dict(size=16, color="black"),
                        xaxis=dict(tickformat=".2f",
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
                                
                    # if st.sidebar.button('Get AV file of this data') : 
                        # Create the CSV data
                        # Create the CSV header and data
                    header, data_df = create_csv_data_AV(min_x, max_x-1, 
                                                    st.session_state.selected_svids, st.session_state.selected_frequency1,
                                                    st.session_state.selected_frequency2, st.session_state.elevation_mask , df4_filtered)

                    # Convert to CSV
                    csv_AV = convert_to_csv(header, data_df)

                    # Create download button
                    st.download_button(
                        label="Download AV data",
                        data=csv_AV,
                        file_name="All_in_view_data.csv",
                        mime="text/csv",
                    )

    else:
        st.error("One of the session data is not available. Either frequency or MJD is not selected properly")
# else: 
#     st.error("Only one data set is available, cannot process either CV or AV analysis")
# Add a spacer to push the contact info to the bottom
st.sidebar.write("")  # This line adds some space
# st.sidebar.write("")  # Add as many as needed to push the content down
# st.sidebar.write("")   

# contact information at the bottom of the sidebar
st.sidebar.markdown('---')  # Add a horizontal line for separation
st.sidebar.markdown('**Contact Information**')
st.sidebar.text('Mr/Ms XYZ')
st.sidebar.text('Email: XYZ@bipm.org')
   
