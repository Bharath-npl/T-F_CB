
import streamlit as st
import pandas as pd
import os
import csv
import warnings
import tempfile
import io 
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

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



def Selected_MJD_data1(files_01, data_set):
    if files_01:
        col1.empty()
        col2.empty()
        unique_mjd_values = set()  # To store unique MJD values
        unique_sv_id = set()  # To store the Unique SV ID values

        # A list to store cleaned data across multiple files
        Required_Colm_data_01 = []
        combined_Colm_data_01 = pd.DataFrame()
        
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
            # df_split.columns = column_names
            column_names = ["SAT", "CL", "MJD", "STTIME", "TRKL", "ELV", "AZTH", "REFSV", "SRSV", "REFSYS"]
            
            # Trim the DataFrame to only the required columns
            df_split = df_split.iloc[:, :len(column_names)]

            # Set the column names
            df_split.columns = column_names

            df_split = df_split[df_split['SAT'].notna()] # Skip the lines where SAT column is missing 
            # print(f"File read is :\n {df_split}")
            # print(f"Sv ids in the Data: \n {df_split['SAT']}")
            # Convert to appropriate datatypes
            df_split['SAT'] = df_split['SAT'].astype(int)
            df_split['MJD'] = df_split['MJD'].astype(float)
            # unique_mjd_values = set(df_split['MJD'])  # Unique MJD values in the list 
            df_split['STTIME'] = df_split['STTIME']  # Keep as string for hhmmss processing

            # Combine hhmmss into MJD
            df_split['MJD'] += df_split['STTIME'].apply(lambda x: (int(x[0:2]) * 3600 + int(x[2:4]) * 60 + int(x[4:6]) * 1) * 0.00001)

            # Convert other relevant columns to desired datatypes
            df_split['ELV'] = df_split['ELV'].astype(float)
            df_split['REFSV'] = df_split['REFSV'].astype(float)
            df_split['SRSV'] = df_split['SRSV'].astype(float)
            df_split['REFSYS'] = df_split['REFSYS'].astype(float)

            Required_Colm_data_01.append(df_split)
            

            unique_mjd_values.update(df_split['MJD'].unique())
            unique_sv_id.update(df_split['SAT'].unique())
        
        combined_Colm_data_01 = pd.concat(Required_Colm_data_01, ignore_index=True)

        # Update the "Start MJD" and "End MJD" select boxes
        unique_mjd_values = sorted(unique_mjd_values)
        unique_mjd_int_values = sorted(set(int(mjd) for mjd in unique_mjd_values))

        # Initialize session state for start_mjd and end_mjd if not already set
        if 'start_mjd_01' not in st.session_state:
            st.session_state['start_mjd_01'] = unique_mjd_int_values[0]
        
        start_mjd_input = st.selectbox(
            "Start MJD",
            options=unique_mjd_int_values,
            key='start_mjd_01',
        )

        # End MJD selectbox, options are constrained based on start_mjd
        end_values = [mjd for mjd in unique_mjd_int_values if mjd >= st.session_state['start_mjd_01']]

        if 'end_mjd_01' not in st.session_state or st.session_state['end_mjd_01'] not in end_values:
            st.session_state['end_mjd_01'] = end_values[-1] if end_values else st.session_state['start_mjd_01']

        end_mjd_input = st.selectbox(
            "End MJD",
            options=end_values,
            key='end_mjd_01'
        )

        # Filtering the dataframe based on selected MJD values
        selected_df_01 = combined_Colm_data_01[
            (combined_Colm_data_01["MJD"].notnull()) &
            (combined_Colm_data_01["MJD"] >= st.session_state['start_mjd_01']) &
            (combined_Colm_data_01["MJD"] < st.session_state['end_mjd_01']+1)
        ]

        # Update session state when the selection changes
        def update_session_state():
            st.session_state['start_mjd_01'] = start_mjd_input
            st.session_state['end_mjd_01'] = end_mjd_input

        # If either start_mjd_input or end_mjd_input has changed, update the session state
        if st.session_state['start_mjd_01'] != start_mjd_input or st.session_state['end_mjd_01'] != end_mjd_input:
            update_session_state()

        # Display filtered dataframe or a warning message if it's empty
        if not selected_df_01.empty:
            st.session_state.filtered_data = selected_df_01
            # st.dataframe(filtered_df)  # Display the filtered dataframe
        else:
            st.warning("No data available for the selected MJD range.")

        # Compute average refsys values grouped by MJD_time
        Avg_refsys_CV = selected_df_01.groupby("MJD")["REFSYS"].mean().reset_index()
        Avg_refsys_CV["REFSYS"] = Avg_refsys_CV["REFSYS"].round(2)

        # Calculate the overall average for df1 and df2
        avg_value_refsys = Avg_refsys_CV["REFSYS"].mean()

        # If plot button is pressed, show the graph
        if plot_option1:
            # Add a new column for the average value
            Avg_refsys_CV["Avg_refsys"] = avg_value_refsys
            st.line_chart(Avg_refsys_CV.set_index("MJD")[["REFSYS", "Avg_refsys"]])

        return selected_df_01

    else:
        return pd.DataFrame()
    

# Fro second set of CGGTTS data 

def Selected_MJD_data2(files_02, data_set):
    if files_02:
        col3.empty()
        col4.empty()
        unique_mjd_values = set()  # To store unique MJD values
        unique_sv_id = set()  # To store the Unique SV ID values

        # A list to store cleaned data across multiple files
        Required_Colm_data_02 = []
        combined_Colm_data_02 = pd.DataFrame()
        
        for each_file in files_02:
            # all_dataframes = []
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
                if line.startswith("CGGTTS") or line.startswith("GGTTS"):
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
            # df_split.columns = column_names
            column_names = ["SAT", "CL", "MJD", "STTIME", "TRKL", "ELV", "AZTH", "REFSV", "SRSV", "REFSYS"]
            
            # Trim the DataFrame to only the required columns
            df_split = df_split.iloc[:, :len(column_names)]

            # Set the column names
            df_split.columns = column_names

            df_split = df_split[df_split['SAT'].notna()] # Skip the lines where SAT column is missing 

            # Convert to appropriate datatypes
            df_split['SAT'] = df_split['SAT'].astype(int)
            df_split['MJD'] = df_split['MJD'].astype(float)
            # unique_mjd_values = set(df_split['MJD'])  # Unique MJD values in the list 
            df_split['STTIME'] = df_split['STTIME']  # Keep as string for hhmmss processing

            # Combine hhmmss into MJD
            df_split['MJD'] += df_split['STTIME'].apply(lambda x: (int(x[0:2]) * 3600 + int(x[2:4]) * 60 + int(x[4:6]) * 1) * 0.00001)

            # Convert other relevant columns to desired datatypes
            df_split['ELV'] = df_split['ELV'].astype(float)
            df_split['REFSV'] = df_split['REFSV'].astype(float)
            df_split['SRSV'] = df_split['SRSV'].astype(float)
            df_split['REFSYS'] = df_split['REFSYS'].astype(float)

            Required_Colm_data_02.append(df_split)
            

            unique_mjd_values.update(df_split['MJD'].unique())
            unique_sv_id.update(df_split['SAT'].unique())
        
        combined_Colm_data_02 = pd.concat(Required_Colm_data_02, ignore_index=True)

        # Update the "Start MJD" and "End MJD" select boxes
        unique_mjd_values = sorted(unique_mjd_values)
        unique_mjd_int_values = sorted(set(int(mjd) for mjd in unique_mjd_values))

        # Initialize session state for start_mjd and end_mjd if not already set
        if 'start_mjd_02' not in st.session_state:
            st.session_state['start_mjd_02'] = unique_mjd_int_values[0]
        
        start_mjd_input = st.selectbox(
            "Start MJD",
            options=unique_mjd_int_values,
            key='start_mjd_02',
        )

        # End MJD selectbox, options are constrained based on start_mjd
        end_values = [mjd for mjd in unique_mjd_int_values if mjd >= st.session_state['start_mjd_02']]

        if 'end_mjd_02' not in st.session_state or st.session_state['end_mjd_02'] not in end_values:
            st.session_state['end_mjd_02'] = end_values[-1] if end_values else st.session_state['start_mjd_02']

        end_mjd_input = st.selectbox(
            "End MJD",
            options=end_values,
            key='end_mjd_02'
        )

        # Filtering the dataframe based on selected MJD values
        Selected_df_02 = combined_Colm_data_02[
            (combined_Colm_data_02["MJD"].notnull()) &
            (combined_Colm_data_02["MJD"] >= st.session_state['start_mjd_02']) &
            (combined_Colm_data_02["MJD"] < st.session_state['end_mjd_02']+1)
        ]

        # Update session state when the selection changes
        def update_session_state():
            st.session_state['start_mjd_02'] = start_mjd_input
            st.session_state['end_mjd_02'] = end_mjd_input

        # If either start_mjd_input or end_mjd_input has changed, update the session state
        if st.session_state['start_mjd_02'] != start_mjd_input or st.session_state['end_mjd_02'] != end_mjd_input:
            update_session_state()

        # Display filtered dataframe or a warning message if it's empty
        if not Selected_df_02.empty:
            st.session_state.filtered_data = Selected_df_02
            # st.dataframe(Selected_df_02)  # Display the filtered dataframe
        else:
            st.warning("No data available for the selected MJD range.")

        # Compute average refsys values grouped by MJD_time
        Avg_refsys_AV = Selected_df_02.groupby("MJD")["REFSYS"].mean().reset_index()
        Avg_refsys_AV["REFSYS"] = Avg_refsys_AV["REFSYS"].round(2)

        # Calculate the overall average for df1 and df2
        avg_value_refsys = Avg_refsys_AV["REFSYS"].mean()

        # If plot button is pressed, show the graph
        if plot_option2:
            # Add a new column for the average value
            Avg_refsys_AV["Avg_refsys"] = avg_value_refsys
            st.line_chart(Avg_refsys_AV.set_index("MJD")[["REFSYS", "Avg_refsys"]])

        return Selected_df_02

    else:
        return pd.DataFrame()


with st.form("my-form1", clear_on_submit=True):
        files_01 = st.file_uploader(":file_folder: Upload the CGGTTS files of Lab 1", accept_multiple_files=True)
        submitted1 = st.form_submit_button("Submit1")


st.sidebar.header("Lab 1 Data")
plot_option1 = st.sidebar.button("Plot Avg RefSys",key=55)

st.sidebar.header("Lab 2 Data")
plot_option2 = st.sidebar.button("Plot Avg RefSys",key=10)

st.sidebar.header("Common View Performance")
plot_CV = st.sidebar.button("Plot CV", key= 35)
# plot_button = st.sidebar.button("CV Performance", key=5)

st.sidebar.header("All in View Performance")
plot_AV = st.sidebar.button("Plot AV", key= 36)
# plot_button = st.sidebar.button("CV Performance", key=5)

if submitted1:
    st.session_state['df1_data'] = Selected_MJD_data1(files_01,1)
        
with st.form("my-form2", clear_on_submit=True):
        files_02 = st.file_uploader(":file_folder: Upload the CGGTTS files of Lab 2", accept_multiple_files=True)
        submitted2 = st.form_submit_button("Submit2")

if submitted2:
    # Selected_MJD_data2(files_02,2)
    st.session_state['df2_data'] = Selected_MJD_data2(files_02,2)

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


if 'df1_data' in st.session_state and 'df2_data' in st.session_state:
    
    st.session_state.df1_mjd = st.session_state.df1_data
    st.session_state.df2_mjd = st.session_state.df2_data
    # st.write("Hello world")

    # if not df1_mjd.empty and not df2_mjd.empty:
    if st.session_state.df1_mjd is not None and st.session_state.df2_mjd is not None:
        
        # Extract unique values
        unique_svids_df1 = st.session_state.df1_mjd['SAT'].unique()
        unique_svids_df2 = st.session_state.df2_mjd['SAT'].unique()
        
        unique_SVIDs = []
        unique_MJD_times = sorted(set(st.session_state.df1_mjd["MJD"]).union(set(st.session_state.df2_mjd["MJD"])))

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
            if 'plot_data' not in st.session_state:
                st.session_state.plot_data = None                       

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

            # plot_button = st.sidebar.button("Plot CV")

            if plot_CV:

                if 'ALL' in selected_svids or len(selected_svids) == len(unique_SVIDs):
                    svids_to_use = unique_SVIDs
                else:
                    svids_to_use = selected_svids
                # If individual SV_ids are selected/deselected, remove ALL
                if len(selected_svids) < len(unique_SVIDs) + 1:
                    selected_svids = [svid for svid in selected_svids if svid != 'ALL']

                st.session_state.selected_svids = svids_to_use

                if 'ALL' in st.session_state.selected_svids:
                    st.session_state.selected_svids = list(unique_SVIDs)

                # print(f"Selected SV IDs are: {st.session_state.selected_svids}")
                unique_MJD_times = sorted(set(st.session_state.df1_mjd["MJD"]).union(set(st.session_state.df2_mjd["MJD"])))
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
                
                st.write(f"Data Missing for the sessions: {missing_session}")

                st.session_state.plot_data = pd.DataFrame(CV_data, columns=['MJD_time', 'CV_avg_diff'])
                
            # if  not st.session_state.plot_data.empty:
            if st.session_state.plot_data is not None and not st.session_state.plot_data.empty:

                df3 =st.session_state.plot_data
                # Set x-axis range
                min_x = math.floor(min(df3["MJD_time"]))
                max_x = math.ceil(max(df3["MJD_time"]))

                # Filter rows of the dataframe based on the x-axis range
                df_filtered = df3[(df3["MJD_time"] >= min_x) & (df3["MJD_time"] <= max_x)]
                # st.write(f"Filtered data: \n {df_filtered}")

                # Calculate the mean of CV_avg_diff only for the filtered data
                mean_val = df_filtered["CV_avg_diff"].mean()
                
                 
                fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figsize as per requirement

                
                # Plot only the filtered data points
                ax.scatter(df_filtered["MJD_time"], df_filtered["CV_avg_diff"])

                # ax.scatter(df3.set_index("MJD_time").index, df3.set_index("MJD_time")["CV_avg_diff"])
                # Plot the mean line
                ax.axhline(mean_val, color='r', linestyle='--', label=f"Mean: {mean_val:.2f} ns")
                # ax.set_title("CV performance between two labs ( MJD: ", min_x , "to MJD: ", max_x)
                ax.set_title(f"CV performance between two labs ( MJD: {min_x} to MJD: {max_x})")
                ax.legend()
                ax.set_xlabel("MJD time")
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
                ax.xaxis.get_major_formatter().set_scientific(False)
                ax.xaxis.get_major_formatter().set_useOffset(False)
                ax.set_ylabel("Time difference (ns) ")
                ax.grid(True)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                
                # Display the plot
                st.pyplot(fig)
                
            if 'selected_svids' not in st.session_state:
                st.session_state.selected_svids = ['ALL']
            if plot_AV:

                if 'ALL' in selected_svids or len(selected_svids) == len(unique_SVIDs):
                    svids_to_use = unique_SVIDs
                else:
                    svids_to_use = selected_svids
                # If individual SV_ids are selected/deselected, remove ALL
                if len(selected_svids) < len(unique_SVIDs) + 1:
                    selected_svids = [svid for svid in selected_svids if svid != 'ALL']

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
                                weighted_mean_df1 = (df1_filtered.loc[condition1, 'REFSYS'] * k1_value * df1_filtered.loc[condition1, 'inv_cos2']).mean()
                                # st.write(f"Weights: {weighted_mean_df1}")
                                # weighted_means_df1.append(weighted_mean_df1)
                            
                            if condition2.any():
                                weighted_mean_df2 = (df2_filtered.loc[condition2, 'REFSYS'] * k2_value * df2_filtered.loc[condition2, 'inv_cos2']).mean()
                                
                            
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

            if st.session_state.plot_AV_data is not None and not st.session_state.plot_AV_data.empty:

                df4 =st.session_state.plot_AV_data
                # Set x-axis range
                min_x = math.floor(min(df4["MJD_time"]))
                # max_x = math.floor(max(df3["MJD_time"]))
                max_x = math.ceil(max(df4["MJD_time"]))

                # Filter rows of the dataframe based on the x-axis range
                df4_filtered = df4[(df4["MJD_time"] >= min_x) & (df4["MJD_time"] <= max_x)]
                mean_val = df4_filtered["AV_diff"].mean()

                fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figsize as per requirement
                # ax.plot(df3.set_index("MJD_time")["CV_avg_diff"])
                # Calculate the mean of CV_avg_diff
                
                ax.scatter(df4_filtered.set_index("MJD_time").index, df4_filtered.set_index("MJD_time")["AV_diff"])
                # Plot the mean line
                ax.axhline(mean_val, color='r', linestyle='--', label=f"Mean: {mean_val:.2f} ns")
                # ax.set_title("CV performance between two labs ( MJD: ", min_x , "to MJD: ", max_x)
                ax.set_title(f"AV performance between two labs ( MJD: {min_x} to MJD: {max_x})")
                ax.legend()
                ax.set_xlabel("MJD time")
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
                ax.xaxis.get_major_formatter().set_scientific(False)
                ax.xaxis.get_major_formatter().set_useOffset(False)
                ax.set_ylabel("Time difference (ns) ")
                ax.grid(True)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                st.pyplot(fig)

    else:
        st.write("No selected data to work up on")

   
