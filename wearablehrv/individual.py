#########################INDIVIDUAL######################################
#########################INDIVIDUAL######################################
#########################INDIVIDUAL######################################
############################## Wearablehrv ##############################

############################Importing Modules############################

import datetime
import os
import json
import pickle
import pandas as pd
import tkinter as tk
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import ipywidgets as widgets
import matplotlib.dates as mdates
from ipywidgets import IntText, Dropdown, Output, HBox
from IPython.display import display, clear_output, Markdown
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis import get_time_domain_features
from hrvanalysis import get_frequency_domain_features
from avro.datafile import DataFileReader
from avro.io import DatumReader

###########################################################################
###########################################################################
###########################################################################

def labfront_conversion (path, pp, file_name, device_name, date):

    """
    Converts Labfront data into a standardized CSV format, filtering by a specific date.

    This function processes CSV data from Labfront. It first reads the data, then focuses on 
    the 'isoDate' and 'bbi' columns. The data is then filtered based on the provided date, and 
    relevant columns are renamed for standardization. The processed data is then saved into a new CSV file.

    Parameters:
    -----------
    path : str
        The directory path pointing to the location of the Labfront data.
    pp : str
        The unique ID of the participant for which the data is being processed.
    file_name : str
        The name of the Labfront file (with its extension) to be processed.
    device_name : str
        The name of the device used to collect the data. This will be used in the resulting CSV's filename.
    date : str
        The specific date for which data should be extracted, provided in a format that can be parsed by pandas' to_datetime function (e.g., 'YYYY-MM-DD').

    Returns:
    --------
    None. The function saves the output directly to a CSV file in the specified path.
    """

    labfront = pd.read_csv (path+file_name, skiprows=5)
    # convert isoDate to datetime column
    labfront['isoDate'] = pd.to_datetime(labfront['isoDate'])
    # filter rows for the selected date
    labfront = labfront[labfront['isoDate'].dt.date == pd.to_datetime(date).date()]
    labfront = labfront [["unixTimestampInMs", "bbi"]]
    labfront = labfront.rename (columns = {"unixTimestampInMs":"timestamp", "bbi": "rr"})
    labfront.to_csv (path+pp+"_"+device_name+".csv", index = False)

    print ("Dataset Successfully Converted and Saved in Your Path!")

###########################################################################
###########################################################################
###########################################################################

def empatica_conversion (path, pp):
    
    """
    Converts Empatica data from Avro format into a CSV file, focusing on the 'systolicPeaks' field.

    This function processes data files associated with a participant's Empatica device data stored 
    in Avro format. It specifically reads the 'systolicPeaks' field from these Avro files. 
    The extracted peak times (in nanoseconds) are converted to milliseconds, and the interbeat 
    intervals (IBIs) are then calculated. The resulting data is saved to a CSV file.

    Parameters:
    -----------
    path : str
        The directory path pointing to the location of the participant's Empatica data.
    pp : str
        The unique ID of the participant whose Empatica data is to be converted.

    Note:
    -----
    The expected directory structure is: 
    <path>/<participant_id>_empatica/raw_data/v6
    with Avro files containing the 'systolicPeaks' field.

    Returns:
    --------
    None. The function saves the output directly to a CSV file in the specified path.
    """

    avrofiles_path = path + "/" +  pp + "_empatica" + "/raw_data"  + "/v6"
    file_name = "empatica"

    # Function to read systolicPeaks data from a single Avro file
    def read_systolic_peaks_from_file(avro_file):
        reader = DataFileReader(open(avro_file, "rb"), DatumReader())
        data = []
        for datum in reader:
            data = datum
        reader.close()

        systolic_peaks = data['rawData']['systolicPeaks']
        return systolic_peaks

    # Iterate through all Avro files in the given directory and combine the 'systolicPeaks' data
    combined_systolic_peaks_ns = []
    for file in os.listdir(avrofiles_path):
        if file.endswith(".avro"):
            avro_file = os.path.join(avrofiles_path, file)
            systolic_peaks = read_systolic_peaks_from_file(avro_file)
            combined_systolic_peaks_ns.extend(systolic_peaks['peaksTimeNanos'])

    # convert each nanosecond value to millisecond
    combined_systolic_peaks_ms = [x // 1000000 for x in combined_systolic_peaks_ns]
    # calculating the interbeat intervals
    ibis = np.diff (combined_systolic_peaks_ms)
    # create a DataFrame
    data = {"timestamp": combined_systolic_peaks_ms[:-1],"rr": ibis} # exclude the last timestamp since there's no corresponding rr_interval
    # turn the dataframe into a pandas dataframe
    df = pd.DataFrame(data)
    # saving the file
    df.to_csv (path+pp+"_empatica.csv", index = False)

    print ("Data saved succesfully in your path")

###########################################################################
###########################################################################
###########################################################################

def define_events(path, pp, conditions, already_saved= True, save_as_csv= False):

    """
    This function defines and saves events that occurred during a task for a specific participant.

    Parameters:
    -----------
    path : str
        The path to the directory where the events file should be saved.
    pp : str
        The ID of the participant for whom the events are being defined.
    conditions : list
        A list of strings that represent the different conditions in the task.
    already_saved : bool, optional
        A boolean variable that indicates if the events file has already been saved previously. If True, the function reads the file from the specified path. If False, the function opens a GUI to allow the user to define the events interactively. Default is True.
    save_as_csv : bool, optional
        A boolean variable that indicates if the events DataFrame should be saved as a CSV file. Default is False.

    Returns:
    --------
    events : pandas DataFrame
        A DataFrame that contains the events data for the participant.
    """

    # Define the path to the events file
    path_events = path + pp + "_events.csv" # creathing the path

    if already_saved == True: # If there is already a pp_events.csv file with the spesified format saved

        # Read the events file into a DataFrame
        events = pd.read_csv(path_events,names =['timestamp','conditions','datapoint','remove'],skiprows=1)
        events = events.drop(events.columns[3], axis=1) #removing the fourth column which is useless
        events = events.sort_values(by='timestamp', ascending=True) #sorting the dataframe based on the times
        events = events.reset_index(drop=True) #restting the indexing

    elif already_saved == False: # Opening the GUI to enter the conditions

        # Define a function to handle the button click
        def submit():
            # Get the values from the user inputs
            start_time = start_time_var.get()
            end_time = end_time_var.get()
            condition = condition_var.get()

            # Check if all fields are filled
            if start_time and end_time and condition:
                # Add the event to the DataFrame
                events_df.loc[len(events_df)] = [start_time, condition, "start"]
                events_df.loc[len(events_df)] = [end_time, condition, "end"]

                # Clear the input fields
                start_time_var.set('')
                end_time_var.set('')
                condition_menu.selection_clear()
                start_time_entry.focus()

                # Update the labels to show which fields are filled
                condition_index = conditions.index(condition)
                start_status_labels[condition_index].config(text=start_time)
                end_status_labels[condition_index].config(text=end_time)

            else:
                # Display an error message if not all fields are filled
                error_label.config(text="Please fill all fields")

        def save_all():
            # Change the column names
            events_df.columns = ['timestamp', 'conditions', 'datapoint']
            # Save the final file in a variable called "events"
            global events
            events = events_df
            # Close the GUI window
            window.destroy()

        # Create the GUI window
        window = tk.Tk()
        window.title("Event Tracker")

        # Create the input fields and labels
        start_time_label = tk.Label(window, text="Start Time (HH:MM:SS)")
        start_time_label.grid(row=0, column=0)
        start_time_var = tk.StringVar()
        start_time_entry = tk.Entry(window, textvariable=start_time_var)
        start_time_entry.grid(row=0, column=1)

        end_time_label = tk.Label(window, text="End Time (HH:MM:SS)")
        end_time_label.grid(row=1, column=0)
        end_time_var = tk.StringVar()
        end_time_entry = tk.Entry(window, textvariable=end_time_var)
        end_time_entry.grid(row=1, column=1)

        condition_label = tk.Label(window, text="Condition")
        condition_label.grid(row=2, column=0)
        condition_var = tk.StringVar()
        condition_menu = tk.OptionMenu(window, condition_var, *conditions)
        condition_menu.grid(row=2, column=1)

        submit_button = tk.Button(window, text="Submit", command=submit)
        submit_button.grid(row=3, column=1)

        # Create the status and error labels
        status_label = tk.Label(window, text="Fields Filled:")
        status_label.grid(row=4, column=0, columnspan=2, sticky="W")

        # Create the status labels for each condition
        start_status_labels = []
        end_status_labels = []
        for i, condition in enumerate(conditions):
            condition_label = tk.Label(window, text=condition)
            condition_label.grid(row=i+5, column=0)

            start_status_label = tk.Label(window, text="NOT FILLED", fg="red")
            start_status_label.grid(row=i+5, column=1)
            start_status_labels.append(start_status_label)

            end_status_label = tk.Label(window, text="NOT FILLED", fg="red")
            end_status_label.grid(row=i+5, column=2)
            end_status_labels.append(end_status_label)

        error_label = tk.Label(window, fg="red")
        error_label.grid(row=len(conditions)+6, column=0, columnspan=2, sticky="W")

        # Create the "Save All" button
        save_all_button = tk.Button(window, text="Save All", command=save_all)
        save_all_button.grid(row=len(conditions)+6, column=0, columnspan=2, pady=10)

        # Create the DataFrame to hold the events
        events_df = pd.DataFrame(columns=['timestamp', 'conditions', 'datapoint'])
        events = events_df

        # Run the GUI loop
        window.mainloop()

    if save_as_csv == True:
        events.to_csv(path_events, index=False)
        print ("The event.csv is saved for futute use")

    print ("Events imported succesfully")
    return events

###########################################################################
###########################################################################
###########################################################################

def import_data (path, pp, devices):

    """
    Imports participant-specific data from different devices and consolidates them into a dictionary.

    This function processes data files associated with different devices. For the "vu" device, it 
    specifically reads from a text file exported from VU-DAMS. Other devices' data are expected in CSV format,
    often recorded using HRV Logger. It's noteworthy that, for HRV Logger, if there's an unnecessary third 
    column in the data, it will be dropped.

    Parameters:
    -----------
    path : str
        The directory path where the data files corresponding to the participant are located.
    pp : str
        The unique ID of the participant whose data is to be imported.
    devices : list of str
        Names of devices from which the data has been collected. Data from each device is expected to be 
        in a file named in the format: <participant_id>_<device_name>.<appropriate_extension>.

    Returns:
    --------
    data : dict
        A dictionary wherein each key is a device name and the associated value is a DataFrame containing 
        the data from that device for the specified participant.
    """

    data = {device: {} for device in devices} #creating an empty dictionary to store data from all devices
    #2. reading data from devices:

    for device in devices:
        path_vu = path + pp + '_' + "vu" + '.txt'
        path_devices = path + pp + '_' + device + '.csv'

        if device == "vu": #this is the text file exported from VU-DAMS
            data[device] = pd.read_csv(path_vu, sep="\t", skiprows=1)
            data[device] = data[device][["R-peak time", "ibi"]] # Select only the "R-peak time" and "ibi" columns
            data[device] = data[device].rename(columns={"R-peak time": "timestamp", "ibi": "rr"}) # Rename the columns

        else: #these are the csv files recorded using HRV Logger
            data[device] = pd.read_csv(path_devices, header=0)
            try: #this is because in HRV, there is a third column that needs to be dropped; but if someone makes a similar dataset, there is no need for this line of code
                data[device] = data[device].drop(data[device].columns[2], axis=1) #removing the third column which is useless
                data[device].columns = data[device].columns.str.strip() # Strip leading/trailing whitespace from column labels
            except:
                data[device].columns = data[device].columns.str.strip() # Strip leading/trailing whitespace from column labels

     #3. changing dataset timestamps:
    for device in devices:

        if device == "vu":
            data[device]['timestamp'] = data[device]['timestamp'].apply(lambda x: x.split('/')[-1])
            data[device]['timestamp'] = pd.to_datetime(data[device]['timestamp'], format='%H:%M:%S.%f')

        else:

            for i in range(np.size(data[device]['timestamp'])):
                timestamp_float = float(data[device].loc[i, 'timestamp'])
                data[device].loc[i, 'timestamp'] = datetime.datetime.fromtimestamp(timestamp_float / 1000)
            data[device]['timestamp'] = pd.to_datetime(data[device]['timestamp'])

        # Format timestamp column as string with format hh:mm:ss.mmm
        data[device]['timestamp'] = data[device]['timestamp'].apply(lambda x: x.strftime('%H:%M:%S.%f')[:-3])   
    
    print ("Datasets imported succesfully")
    return data

###########################################################################
###########################################################################
###########################################################################

def chop_data (data, conditions, events, devices):

    """
    This function chops the data from different devices into separate segments based on the events.

    Parameters:
    -----------
    data : dict
        A dictionary containing the raw data for all devices and conditions.
    conditions : list
        A list of strings that represent the different conditions in the task.
    events : pandas DataFrame
        A DataFrame containing the event data for the participant.
    devices : list
        A list of strings that represent the different devices used to collect the data.

    Returns:
    --------
    data_chopped : dict
        A dictionary containing the chopped data for all devices and conditions.
    """
    
    #it contains the begening and end of each condition
    eventchopped = {}  # it contains the beginning and end of each condition

    for condition in conditions:
        start = events.loc[(events['conditions'] == condition) & (events['datapoint'] == 'start')]['timestamp'].iloc[0]
        end = events.loc[(events['conditions'] == condition) & (events['datapoint'] == 'end')]['timestamp'].iloc[0]
        start = datetime.datetime.strptime(start, '%H:%M:%S')
        end = datetime.datetime.strptime(end, '%H:%M:%S')
        start = start.strftime('%H:%M:%S.%f')[:-3]
        end = end.strftime('%H:%M:%S.%f')[:-3]
        eventchopped[condition] = (start, end)

    # a new dictionary that contains chopped (based on events) rr intervals for each device and condition
    data_chopped = {device: {condition: {} for condition in conditions} for device in devices}

    for device in devices:
        for condition in conditions:
            filtered_rows = data[device][(data[device]['timestamp'] >= eventchopped[condition][0]) & (data[device]['timestamp'] < eventchopped[condition][1])].copy()
            filtered_rows['timestamp'] = pd.to_datetime(filtered_rows['timestamp']) # convert to datetime format
            data_chopped[device][condition] = filtered_rows #this one now contains both rr intervals and timestamps

    print ("Data are chopped based on the events succesfully")
    return data_chopped

###########################################################################
###########################################################################
###########################################################################
def calculate_ibi (data_chopped, devices, conditions):

    """
    This function calculates the number of Inter-Beat Intervals (IBI) for each condition and device.

    Parameters:
    -----------
    data_chopped : dict
        A dictionary containing the chopped RR interval data for all devices and conditions.
    conditions : list
        A list of strings that represent the different conditions in the task.
    devices : list
        A list of strings that represent the different devices used to collect the data.

    Returns:
    --------
    nibis : dict
        A dictionary containing the number of IBIs for each condition and device.
    """

    nibis = {device: {condition: {} for condition in conditions} for device in devices}

    for device in devices:
        for condition in conditions:
            nibis[device][condition] = np.shape(data_chopped[device][condition])[0]

    print ("The number of calculated IBI per condition per deivice has been succesfully calculated")
    return nibis

###########################################################################
###########################################################################
###########################################################################

def visual_inspection (data_chopped, devices, conditions, criterion):

    """
    This function allows for visual inspection and manual modification of the RR interval data.

    Parameters:
    -----------
    data_chopped : dict
        A dictionary containing the chopped RR interval data for all devices and conditions.
    devices : list
        A list of strings that represent the different devices used to collect the data.
    conditions : list
        A list of strings that represent the different conditions in the task.
    criterion : str
        A string that represents the device used as the criterion device.

    Returns:
    --------
    None
    """

    # Define the function that creates the plot
    def plot_rr_intervals(device, condition, lag, device_start, device_end, criterion_start, criterion_end):
        # Trimming
        trim_device = slice(device_start, device_end)
        trim_criterion = slice(criterion_start, criterion_end)

        # Get the RR intervals and timestamps for the selected condition and device
        ppg_rr = data_chopped[device][condition]['rr'][trim_device]
        ppg_timestamp = data_chopped[device][condition]['timestamp'][trim_device]

        # Get the RR intervals and timestamps for the criterion (vu)
        criterion_rr = data_chopped[criterion][condition]['rr'][trim_criterion]
        criterion_timestamp = data_chopped[criterion][condition]['timestamp'][trim_criterion]

        # Adjust lag based on precision (seconds or milliseconds)
        if precision_dropdown.value == 'Milliseconds':
            lag = lag / 1000  # Convert milliseconds to seconds

        # Shift the timestamps of the PPG device by the lag amount
        ppg_timestamp = ppg_timestamp + pd.Timedelta(seconds=lag)

        # Create a figure with a larger size
        plt.figure(figsize=(17, 5))

        # Plot the RR intervals for the selected device and the criterion
        plt.plot(ppg_timestamp, ppg_rr, '-o', color='red', label=device, markersize=7)
        plt.plot(criterion_timestamp, criterion_rr, '-o', color='black', label=criterion, markersize=7)

        # Add grid lines to the plot
        plt.grid()

        # Set the title and axis labels
        plt.title("Beat-to-beat intervals for {} condition".format(condition))
        plt.xlabel("Timestamp", fontsize=20)
        plt.ylabel("RR Intervals", fontsize=20)

        # Format the x-axis as dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))

        # Rotate the x-axis labels
        plt.xticks(rotation=90)

        # Add a legend to the plot
        plt.legend()

        # Show the plot
        plt.show()

    # Define the function that saves the lag
    def save_lag(lag):
        # Adjust lag based on precision (seconds or milliseconds)
        if precision_dropdown.value == 'Milliseconds':
            lag = lag / 1000  # Convert milliseconds to seconds

        if correction_mode_dropdown.value == "Full Lag Correction":
            for condition in conditions:
                ppg_timestamp = data_chopped[device_dropdown.value][condition]['timestamp']
                data_chopped[device_dropdown.value][condition]['timestamp'] = ppg_timestamp + pd.Timedelta(seconds=lag)
        else:
            ppg_timestamp = data_chopped[device_dropdown.value][condition_dropdown.value]['timestamp']
            data_chopped[device_dropdown.value][condition_dropdown.value]['timestamp'] = ppg_timestamp + pd.Timedelta(seconds=lag)

        # Reset the lag to 0
        lag_slider.value = 0

        # Display a message to inform the user that the data has been modified
        print("Data has been modified with a lag of {} seconds.".format(lag))

    def save_crop(device_start, device_end, criterion_start, criterion_end):
        data_chopped[device_dropdown.value][condition_dropdown.value]['rr'] = data_chopped[device_dropdown.value][condition_dropdown.value]['rr'][device_start:device_end]
        data_chopped[device_dropdown.value][condition_dropdown.value]['timestamp'] = data_chopped[device_dropdown.value][condition_dropdown.value]['timestamp'][device_start:device_end]

        data_chopped[criterion][condition_dropdown.value]['rr'] = data_chopped[criterion][condition_dropdown.value]['rr'][criterion_start:criterion_end]
        data_chopped[criterion][condition_dropdown.value]['timestamp'] = data_chopped[criterion][condition_dropdown.value]['timestamp'][criterion_start:criterion_end]

        # Drop any rows with NaN or NaT values in RR intervals and timestamp columns for both device and criterion
        data_chopped[device_dropdown.value][condition_dropdown.value].dropna(subset=['rr', 'timestamp'], inplace=True)
        data_chopped[criterion][condition_dropdown.value].dropna(subset=['rr', 'timestamp'], inplace=True)

        update_device_condition()
        print("Cropped data has been saved.")

    def update_device_condition(*args):
        lag_slider.value = 0

        device_start_slider.value = 0
        device_start_slider.max = len(data_chopped[device_dropdown.value][condition_dropdown.value]['rr']) - 1
        device_end_slider.max = len(data_chopped[device_dropdown.value][condition_dropdown.value]['rr'])
        device_end_slider.value = len(data_chopped[device_dropdown.value][condition_dropdown.value]['rr'])

        criterion_start_slider.value = 0
        criterion_start_slider.max = len(data_chopped[criterion][condition_dropdown.value]['rr']) - 1
        criterion_end_slider.max = len(data_chopped[criterion][condition_dropdown.value]['rr'])
        criterion_end_slider.value = len(data_chopped[criterion][condition_dropdown.value]['rr'])

        with out:
            clear_output(True)
            plot_rr_intervals(device_dropdown.value, condition_dropdown.value, lag_slider.value, device_start_slider.value, device_end_slider.value, criterion_start_slider.value, criterion_end_slider.value)

    def update_plot(change):
        with out:
            clear_output(True)
            plot_rr_intervals(device_dropdown.value, condition_dropdown.value, lag_slider.value, device_start_slider.value, device_end_slider.value, criterion_start_slider.value, criterion_end_slider.value)

    # Create two sets of start and end slider widgets for device and criterion
    device_start_slider = widgets.IntSlider(min=0, max=len(data_chopped[devices[0]][conditions[0]]['rr'])-1, value=0, description='Device Start:', continuous_update=False)
    device_end_slider = widgets.IntSlider(min=1, max=len(data_chopped[devices[0]][conditions[0]]['rr']), value=len(data_chopped[devices[0]][conditions[0]]['rr']), description='Device End:', continuous_update=False)

    criterion_start_slider = widgets.IntSlider(min=0, max=len(data_chopped[criterion][conditions[0]]['rr'])-1, value=0, description='Criterion Start:', continuous_update=False)
    criterion_end_slider = widgets.IntSlider(min=1, max=len(data_chopped[criterion][conditions[0]]['rr']), value=len(data_chopped[criterion][conditions[0]]['rr']), description='Criterion End:', continuous_update=False)

    # Define the widget for lag correction mode
    correction_mode_dropdown = widgets.Dropdown(
        options=['Individual Lag Correction', 'Full Lag Correction'],
        value='Individual Lag Correction',
        description='Correction Mode:',
        disabled=False,
    )

    # Define the precision dropdown widget
    precision_dropdown = widgets.Dropdown(
        options=['Seconds', 'Milliseconds'],
        value='Seconds',
        description='Precision:',
        disabled=False,
    )

    # Function to update lag_slider parameters based on precision
    def update_lag_slider_precision(*args):
        if precision_dropdown.value == 'Seconds':
            lag_slider.min = -20
            lag_slider.max = 20
            lag_slider.value = 0
            lag_slider.description = 'Lag (s):'
            lag_slider.readout_format = 'd'
        else:
            lag_slider.min = -20000
            lag_slider.max = 20000
            lag_slider.value = 0
            lag_slider.description = 'Lag (ms):'
            lag_slider.readout_format = 'd'

    # Observe changes in precision dropdown and update lag slider accordingly
    precision_dropdown.observe(update_lag_slider_precision, names='value')

    # Create the device dropdown widget
    device_dropdown = widgets.Dropdown(
        options=devices,
        value=devices[0],
        description='Device:',
        disabled=False,
    )

    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description='Condition:',
        disabled=False,
    )

    lag_slider = widgets.IntSlider(
        value=0,
        min=-20,
        max=20,
        step=1,
        description='Lag (s):',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    lag_slider.layout = widgets.Layout(width='80%')

    # Define the buttons
    plot_button = widgets.Button(
        description='Plot RR Intervals',
        disabled=False,
        button_style='',
        tooltip='Click me',
        icon='check'
    )
    save_button = widgets.Button(
        description='Save Lag',
        disabled=False,
        button_style='',
        tooltip='Click me',
        icon='save'
    )
    save_crop_button = widgets.Button(
        description='Save Crop',
        disabled=False,
        button_style='',
        tooltip='Click me',
        icon='crop'
    )

    # Create the output widget
    out = widgets.Output()

    # Define the trimming box
    trimming_box = widgets.VBox(children=[
        widgets.HBox(children=[device_start_slider, device_end_slider]),
        widgets.HBox(children=[criterion_start_slider, criterion_end_slider]),
    ])

    # Register event listeners
    lag_slider.observe(update_plot, names='value')
    device_start_slider.observe(update_plot, names='value')
    device_end_slider.observe(update_plot, names='value')
    criterion_start_slider.observe(update_plot, names='value')
    criterion_end_slider.observe(update_plot, names='value')
    device_dropdown.observe(update_device_condition, names='value')
    condition_dropdown.observe(update_device_condition, names='value')
    save_button.on_click(lambda b: save_lag(lag_slider.value))
    save_crop_button.on_click(lambda b: save_crop(device_start_slider.value, device_end_slider.value, criterion_start_slider.value, criterion_end_slider.value))

    # Create the GUI layout

    widgets_box = widgets.VBox(children=[
        widgets.HBox(children=[device_dropdown, condition_dropdown]),
        widgets.HBox(children=[correction_mode_dropdown, precision_dropdown]),
        widgets.HBox(children=[lag_slider, widgets.VBox(children=[save_button, save_crop_button])]),
        trimming_box
    ])

    output_box = widgets.VBox(children=[out])

    gui_box_layout = widgets.Layout(display='flex',
                                    flex_flow='column',
                                    align_items='stretch',
                                    width='80%')
    gui_box = widgets.Box(children=[widgets_box, output_box],
                        layout=gui_box_layout)

    # Call the function to render the initial plot inside the output widget
    update_device_condition()

    # Display the GUI
    display(gui_box)

###########################################################################
###########################################################################
###########################################################################

def save_backup (pp, path, data_chopped):

    """
    This function saves the processed and chopped data into a pickle file.

    Parameters:
    -----------
    pp : str
        The name of the preprocessing applied to the data.
    path : str
        The path where the pickle file will be saved.
    data_chopped : dict
        A dictionary containing the chopped data that has been processed.

    Returns:
    --------
    None

    This function will print a message indicating if the data was saved successfully. It does not have a return value.
    """

    # Save data_chopped
    filename = os.path.join(path, f"{pp}_data_chopped.pkl")
    with open(filename, "wb") as file:
        pickle.dump(data_chopped, file)

    print("Data saved successfully!")

###########################################################################
###########################################################################
###########################################################################

def import_backup (pp, path):

    """
    This function loads the processed and chopped data from a pickle file.

    Parameters:
    -----------
    pp : str
        The name of the preprocessing applied to the data.
    path : str
        The path where the pickle file is located.

    Returns:
    --------
    data_chopped : dict
        A dictionary containing the chopped data that has been processed.

    The function will print a message indicating if the data was loaded successfully.
    """

    # Read the file
    filename = os.path.join(path, f"{pp}_data_chopped.pkl")
    with open(filename, "rb") as file:
        data_chopped = pickle.load(file)

    print("Data loaded successfully!")
    return data_chopped

###########################################################################
###########################################################################
###########################################################################

def pre_processing (data_chopped, devices, conditions, method="karlsson", custom_removing_rule = 0.25, low_rri=300, high_rri=2000):

    """
    This function preprocesses the RR intervals data using the HRV analysis package, by removing outliers,
    interpolating missing values, and removing ectopic beats, and stores the preprocessed data in a dictionary.

    Parameters:
    -----------
    data_chopped : dict
        A dictionary containing the chopped RR interval data for all devices and conditions.
    devices : list
        A list of strings that represent the different devices used to collect the data.
    conditions : list
        A list of strings that represent the different conditions in the task.
    method : str, optional
        A string that represents the method to use for removing ectopic beats. Default is "karlsson".
    custom_removing_rule : float, optional
        A float that represents the custom removing rule for the ectopic beats removal method. Default is 0.25.
    low_rri : int, optional
        An integer that represents the lower threshold for outlier detection. Default is 300.
    high_rri : int, optional
        An integer that represents the higher threshold for outlier detection. Default is 2000.

    Returns:
    --------
    dict
        A dictionary containing the preprocessed RR intervals data for each condition for each device.
    """

    # Turning the dataset into RR intervals only: now that we have visualized the data and learned about its structure, we can simplify the next steps by discarding the x-axis (time axis).
    data_chopped = {device: {condition: list(data_chopped[device][condition]['rr']) for condition in conditions} for device in devices}

    #empty dic to store the pre-processed  RR intervals for each condition for each device
    data_pp = {device: {condition: {} for condition in conditions} for device in devices}

    def preprocess_rr_intervals(rr_intervals):
        """" The four following lines come from the HRV analysis package;
        here I stack them in a function. The input is a given RR interval """

        rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals,low_rri=low_rri, high_rri=high_rri)
        interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, interpolation_method="linear")
        nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method=method, custom_removing_rule = custom_removing_rule)
        interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)
        return interpolated_nn_intervals

    # storing the pre-processed rr values for each condition in a dictionary
    for device in devices:
        for condition in conditions:
            try: #this is because for whatever reason there might not be possible to do the analysis
                data_pp[device][condition] = preprocess_rr_intervals(data_chopped[device][condition])
            except:
                print ("Error: it was not possible to preprocess the data in {} condition for {} device".format (condition, device))
                continue
    print ("Pre-processed RR intervals succesfully stored in dictionaries for each condition")
    return data_pp, data_chopped

###########################################################################
###########################################################################
###########################################################################

def calculate_artefact (data_chopped, data_pp, devices, conditions):

    """
    This function calculates the number of artifacts for each device and condition.

    Parameters:
    -----------
    data_chopped : dict
        A dictionary containing the chopped RR interval data for all devices and conditions.
    data_pp : dict
        A dictionary containing the pre-processed RR interval data for all devices and conditions.
    devices : list
        A list of strings that represent the different devices used to collect the data.
    conditions : list
        A list of strings that represent the different conditions in the task.

    Returns:
    --------
    artefact : dict
        A dictionary containing the number of artifacts for each device and condition.

    """

    artefact = {device: {condition: {} for condition in conditions} for device in devices}

    for device in devices:
        for condition in conditions:
            artefact[device][condition] = sum([1 for x, y in zip(data_chopped[device][condition], data_pp[device][condition]) if x != y])


    print ("The number of detected artefact per condition per deivice has been succesfully calculated")
    return artefact

###########################################################################
###########################################################################
###########################################################################

def ibi_comparison_plot (data_chopped, data_pp, devices, conditions, criterion,  width = 20, height = 10):

    """
    This function plots a comparison of the original and pre-processed RR intervals for each condition and device,
    as well as the reference device, which is used as a criterion. The plots show the difference between the two
    types of RR intervals. The function takes in the chopped and pre-processed RR interval data for all devices
    and conditions, the devices, the conditions, the criterion, as well as optional parameters to customize the
    plot such as the start and end index, width, and height. It then plots the data using matplotlib and returns
    nothing.

    Parameters:
    -----------
    data_chopped : dict
        A dictionary containing the chopped RR interval data for all devices and conditions.
    data_pp : dict
        A dictionary containing the pre-processed RR interval data for all devices and conditions.
    devices : list
        A list of strings that represent the different devices used to collect the data.
    conditions : list
        A list of strings that represent the different conditions in the task.
    criterion : str
        A string that represents the device used as the criterion device.
    width : int, optional
        An integer that represents the width of the plot. Default is 25.
    height : int, optional
        An integer that represents the height of the plot. Default is 30.

    Returns:
    --------
    None

    """

    # Define the function that updates the plot
    def update_plot(*args):
        condition = condition_dropdown.value
        device = device_dropdown.value

        with out:
            clear_output(wait=True)
            fig, axs = plt.subplots(2, 1, figsize=(width, height))

            axs[0].plot(data_chopped[device][condition], '-o', color='red', label='Original')
            axs[0].plot(data_pp[device][condition], '-o', color='black', label='Pre-processed')
            axs[0].grid()
            axs[0].set_title("Beat-to-beat intervals for {} condition in {} device".format(condition.upper(), device.upper()))
            axs[0].set_xlabel("Beats")
            axs[0].set_ylabel("RR Intervals")
            axs[0].legend()

            axs[1].plot(data_chopped[criterion][condition], '-o', color='red', label='Original')
            axs[1].plot(data_pp[criterion][condition], '-o', color='black', label='Pre-processed')
            axs[1].grid()
            axs[1].set_title("Beat-to-beat intervals for {} condition in {} device".format(condition.upper(), criterion.upper()))
            axs[1].set_xlabel("Beats")
            axs[1].set_ylabel("RR Intervals")
            axs[1].legend()

            plt.subplots_adjust(hspace=0.5)
            plt.show()

    # Create the device dropdown widget
    device_dropdown = widgets.Dropdown(
        options=devices,
        value=devices[0],
        description='Device:',
        disabled=False,
    )

    # Create the condition dropdown widget
    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description='Condition:',
        disabled=False,
    )

    device_dropdown.observe(update_plot, names='value')
    condition_dropdown.observe(update_plot, names='value')

    widgets_box = widgets.HBox([device_dropdown, condition_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_plot()

###########################################################################
###########################################################################
###########################################################################

def data_analysis (data_pp, devices, conditions):

    """
    This function calculates all the time domain and frequency domain features by the hrvanalysis package for the pre-processed RR intervals
    data.

    Parameters:
    -----------
    data_pp : dict
        A dictionary containing the pre-processed RR interval data for all devices and conditions.
    devices : list
        A list of strings that represent the different devices used to collect the data.
    conditions : list
        A list of strings that represent the different conditions in the task.

    Returns:
    --------
    time_domain_features : dict
        A dictionary containing the time domain features for each device and condition.
    frequency_domain_features : dict
        A dictionary containing the frequency domain features for each device and condition.
    """

    #calculating all the time domain and frequency domain features by the hrvanalysis package
    time_domain_features = {device: {condition: {} for condition in conditions} for device in devices}
    frequency_domain_features = {device: {condition: {} for condition in conditions} for device in devices}

    for device in devices:
        for condition in conditions:
            try: #this is because for whatever reason there might not be possible to do the analysis
                time_domain_features[device][condition] = get_time_domain_features (data_pp[device][condition])
                frequency_domain_features[device][condition] = get_frequency_domain_features (data_pp[device][condition])
            except:
                print ("Error: it was not possible to analyse the data in {} condition for {} device".format (condition, device))
                continue

    print ("All time domain and frequency domain measures succesfully calculated and stored")
    return time_domain_features, frequency_domain_features

###########################################################################
###########################################################################
###########################################################################

def result_comparison_plot (data_chopped, time_domain_features, frequency_domain_features, devices, conditions, bar_width = 0.20,  width = 20, height = 25):

    """
    This function creates comparison bar charts for time and frequency domain measures of HRV for each device, both for the original data and after pre-processing.

    Parameters:
    -----------
    data_chopped : dict
        A nested dictionary containing the raw HRV data for each device and condition.
    time_domain_features : dict
        A nested dictionary containing the time domain measures of HRV for each device and condition, after pre-processing.
    frequency_domain_features : dict
        A nested dictionary containing the frequency domain measures of HRV for each device and condition, after pre-processing.
    devices : list
        A list of strings representing the different devices used to collect the data.
    conditions : list
        A list of strings representing the different experimental conditions of the data.
    bar_width : float, optional, default=0.20
        The width of the bars in the bar charts.
    width : int, optional, default=20
        The width of the entire plot figure.
    height : int, optional, default=25
        The height of the entire plot figure.

    Returns:
    --------
    None

    This function displays comparison bar charts using interactive widgets. The user can select the time and frequency domain features, and the condition, to visualize the bar charts.
    """

    #calculating all the time domain and frequency domain features by the hrvanalysis package for the original dataset
    time_domain_features_original = {device: {condition: {} for condition in conditions} for device in devices}
    frequency_domain_features_original = {device: {condition: {} for condition in conditions} for device in devices}

    for device in devices:
        for condition in conditions:
            time_domain_features_original[device][condition] = get_time_domain_features (data_chopped[device][condition])
            frequency_domain_features_original[device][condition] = get_frequency_domain_features (data_chopped[device][condition])

    print ("All time domain and frequency domain measures succesfully calculated the original data, before pre-processing for comparison")

    time_features = list (time_domain_features[devices[0]][conditions[0]].keys())
    frequency_features = list (frequency_domain_features[devices[0]][conditions[0]].keys())

    def plot_bar_charts(time_feature, frequency_feature, condition):
        fig, axs = plt.subplots(2, 1, figsize=(width, height))

        x = np.arange(len(devices))

        time_original = [[time_domain_features_original[device][condition][time_feature] for device in devices]]
        time_processed = [[time_domain_features[device][condition][time_feature] for device in devices]]
        freq_original = [[frequency_domain_features_original[device][condition][frequency_feature] for device in devices]]
        freq_processed = [[frequency_domain_features[device][condition][frequency_feature] for device in devices]]

        bar1 = axs[0].bar(x - bar_width/2, time_original[0], bar_width, color='red', label='Original')
        bar2 = axs[0].bar(x + bar_width/2, time_processed[0], bar_width, color='black', label='Pre-processed')
        axs[0].bar_label(bar1, padding=3)
        axs[0].bar_label(bar2, padding=3)

        bar3 = axs[1].bar(x - bar_width/2, freq_original[0], bar_width, color='red', label='Original')
        bar4 = axs[1].bar(x + bar_width/2, freq_processed[0], bar_width, color='black', label='Pre-processed')
        axs[1].bar_label(bar3, padding=3)
        axs[1].bar_label(bar4, padding=3)

        axs[0].set_ylabel(time_feature.upper(), fontsize=25)
        axs[0].set_xlabel('Devices', fontsize=25)
        axs[0].set_title("Time Domain - " + time_feature.upper(), fontsize=25, y=1.02)
        axs[1].set_ylabel(frequency_feature.upper(), fontsize=25)
        axs[1].set_xlabel('Devices', fontsize=25)
        axs[1].set_title("Frequency Domain - " + frequency_feature.upper(), fontsize=25, y=1.02)

        axs[0].set_xticks(x)
        axs[0].set_xticklabels(devices, fontsize=15)
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(devices, fontsize=15)

        axs[0].legend()
        axs[1].legend()

        axs[0].grid(linestyle='--', alpha=0.8)
        axs[1].grid(linestyle='--', alpha=0.8)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        plt.show()

    def update_plots(*args):
        time_feature = time_feature_dropdown.value
        frequency_feature = frequency_feature_dropdown.value
        condition = condition_dropdown.value

        with out:
            clear_output(wait=True)
            plot_bar_charts(time_feature, frequency_feature, condition)

    time_feature_dropdown = widgets.Dropdown(
        options=time_features,
        value=time_features[0],
        description='Time Feature:',
        disabled=False,
    )

    frequency_feature_dropdown = widgets.Dropdown(
        options=frequency_features,
        value=frequency_features[0],
        description='Frequency Feature:',
        disabled=False,
    )

    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description='Condition:',
        disabled=False,
    )

    time_feature_dropdown.observe(update_plots, names='value')
    frequency_feature_dropdown.observe(update_plots, names='value')
    condition_dropdown.observe(update_plots, names='value')

    widgets_box = widgets.HBox([time_feature_dropdown, frequency_feature_dropdown, condition_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_plots()

###########################################################################
###########################################################################
###########################################################################

def unfolding_plot (data_pp, devices, conditions, width = 8, height = 10):

    """
    This function creates a plot of RMSSD and heart rate values over time for a given device and condition.

    Parameters:
    -----------
    data_pp : dict
        A dictionary containing preprocessed data for each device and condition.
    devices : list
        A list of strings representing the different devices.
    conditions : list
        A list of strings representing the different conditions.
    width : int, optional
        An integer that represents the width of the plot. Default is 8.
    height : int, optional
        An integer that represents the height of the plot. Default is 10.

    Returns:
    --------
    None
    """

    sec_text = IntText(
        value=30,
        description='Sec:',
    )

    condition_dropdown = Dropdown(
        options=conditions,
        value=conditions[0],
        description='Condition:',
    )

    device_dropdown = Dropdown(
        options=devices,
        value=devices[0],
        description='Device:',
    )

    def update_plots(*args):
        sec = sec_text.value
        condition = condition_dropdown.value
        device = device_dropdown.value

        with out:
            clear_output(wait=True)
            def calculate_rmssd(rr_intervals):
                differences = np.diff(rr_intervals)  # Calculate the successive differences
                squared_diff = np.square(differences)  # Square the differences
                mean_squared_diff = np.mean(squared_diff)  # Calculate the mean of squared differences
                rmssd_value = np.sqrt(mean_squared_diff)  # Take the square root to get RMSSD
                return rmssd_value

            def calculate_mean_hr(rr_intervals):
                heart_rate_list = np.divide(60000, rr_intervals)  # Calculate heart rate list as 60000 divided by each RR interval
                mean_hr = np.mean(heart_rate_list)  # Calculate mean heart rate from the heart rate list
                return mean_hr

            rr_intervals_chunk = []
            rmssd_values = []
            heart_rates = []

            for rr_interval in data_pp[device][condition]:
                rr_intervals_chunk.append(rr_interval)
                chunk_sum = np.sum(rr_intervals_chunk)
                if (chunk_sum / 1000) >= sec:
                    rmssd_value = calculate_rmssd(rr_intervals_chunk)
                    heart_rate_value = calculate_mean_hr(rr_intervals_chunk)
                    rmssd_values.append(rmssd_value)
                    heart_rates.append(heart_rate_value)
                    rr_intervals_chunk = []

            # Calculate for the last chunk in rr_intervals_chunk
            if rr_intervals_chunk:
                rmssd_value = calculate_rmssd(rr_intervals_chunk)
                heart_rate_value = calculate_mean_hr(rr_intervals_chunk)
                rmssd_values.append(rmssd_value)
                heart_rates.append(heart_rate_value)

            # Create a figure and a set of subplots
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(width, height))

            # Plot rmssd
            axs[0].plot(np.arange(sec, (len(rmssd_values)+1)*sec, sec), rmssd_values, "-o", color='blue')
            axs[0].set_xlabel("Time (sec)")
            axs[0].set_ylabel('RMSSD')
            axs[0].set_title('RMSSD values in {} condition every {} seconds'.format(condition, sec))
            axs[0].grid(True)  # Add grid to the plot

            # Plot HR
            axs[1].plot(np.arange(sec, (len(rmssd_values)+1)*sec, sec), heart_rates, "-o", color='red')
            axs[1].set_xlabel("Time (sec)")
            axs[1].set_ylabel('HR')
            axs[1].set_title('HR values in {} condition every {} seconds'.format(condition, sec))
            axs[1].grid(True)  # Add grid to the plot

            plt.subplots_adjust(hspace=0.3)  # Increase the value to increase the distance
            plt.show()  # Displays the plot

    sec_text.observe(update_plots, names='value')
    condition_dropdown.observe(update_plots, names='value')
    device_dropdown.observe(update_plots, names='value')

    widgets_box = HBox([sec_text, condition_dropdown, device_dropdown])

    out = Output()
    display(widgets_box, out)
    update_plots()

###########################################################################
###########################################################################
###########################################################################

def bar_plot (time_domain_features, frequency_domain_features, devices, conditions, width=20, height=25, bar_width = 0.20):

    """
    This function creates a bar plot of the time domain and frequency domain features for each device and condition.

    Parameters:
    -----------
    time_domain_features : dict
        A dictionary containing the time domain features for each device and condition.
    frequency_domain_features : dict
        A dictionary containing the frequency domain features for each device and condition.
    devices : list
        A list of strings that represent the different devices used to collect the data.
    conditions : list
        A list of strings that represent the different conditions in the task.
    width : int, optional
        An integer that represents the width of the plot. Default is 20.
    height : int, optional
        An integer that represents the height of the plot. Default is 25.
    bar_width : float, optional
        A float that represents the width of each bar. Default is 0.20.
    time_feature : str, optional
        A string that represents the time domain feature to plot. Default is 'rmssd'.
    frequency_feature : str, optional
        A string that represents the frequency domain feature to plot. Default is 'hf'.

    Returns:
    --------
    None
    """

    time_features = list (time_domain_features[devices[0]][conditions[0]].keys())
    frequency_features = list (frequency_domain_features[devices[0]][conditions[0]].keys())

    def plot_bar_charts(time_feature, frequency_feature):
        fig, axs = plt.subplots(2, 1, figsize=(width, height))

        x = np.arange(len(conditions))

        time = [[time_domain_features[device][condition][time_feature] for condition in conditions] for device in devices]
        freq = [[frequency_domain_features[device][condition][frequency_feature] for condition in conditions] for device in devices]

        for i, device_time in enumerate(time):
            axs[0].bar(x + i * bar_width, device_time, bar_width, label=devices[i])

        for i, device_freq in enumerate(freq):
            axs[1].bar(x + i * bar_width, device_freq, bar_width, label=devices[i])

        axs[0].set_ylabel(time_feature.upper(), fontsize=25)
        axs[0].set_xlabel('Conditions', fontsize=25)
        axs[0].set_title("Time Domain - " + time_feature.upper(), fontsize=25, y=1)
        axs[1].set_ylabel(frequency_feature.upper(), fontsize=25)
        axs[1].set_xlabel('Conditions', fontsize=25)
        axs[1].set_title("Frequency Domain - " + frequency_feature.upper(), fontsize=25, y=1)

        axs[0].set_xticks(x)
        axs[0].set_xticklabels(conditions, fontsize=15)
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(conditions, fontsize=15)

        axs[0].legend()
        axs[1].legend()

        axs[0].grid(linestyle='--', alpha=0.8)
        axs[1].grid(linestyle='--', alpha=0.8)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        plt.show()

    def update_plots(*args):
        time_feature = time_feature_dropdown.value
        frequency_feature = frequency_feature_dropdown.value

        with out:
            clear_output(wait=True)
            plot_bar_charts(time_feature, frequency_feature)

    time_feature_dropdown = widgets.Dropdown(
        options=time_features,
        value=time_features[0],
        description='Time Feature:',
        disabled=False,
    )

    frequency_feature_dropdown = widgets.Dropdown(
        options=frequency_features,
        value=frequency_features[0],
        description='Frequency Feature:',
        disabled=False,
    )

    time_feature_dropdown.observe(update_plots, names='value')
    frequency_feature_dropdown.observe(update_plots, names='value')

    widgets_box = widgets.HBox([time_feature_dropdown, frequency_feature_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_plots()

###########################################################################
###########################################################################
###########################################################################

def line_plot (time_domain_features, frequency_domain_features, devices, conditions, width=20, height=25):

    """
    Plots a line graph for time domain and frequency domain features for each device and condition.

    Parameters:
    time_domain_features (dict): A dictionary containing time domain features for each device and condition.
    frequency_domain_features (dict): A dictionary containing frequency domain features for each device and condition.
    devices (list): A list of the devices used in the study.
    conditions (list): A list of the conditions used in the study.
    width (int): Width of the plot in inches.
    height (int): Height of the plot in inches.
    time_feature (str): Time domain feature to plot (default: 'rmssd').
    frequency_feature (str): Frequency domain feature to plot (default: 'hf').

    Returns:
    None
    """

    time_features = list (time_domain_features[devices[0]][conditions[0]].keys())
    frequency_features = list (frequency_domain_features[devices[0]][conditions[0]].keys())

    def plot_line_charts(time_feature, frequency_feature):
        fig, axs = plt.subplots(2, 1, figsize=(width, height))

        x = np.arange(len(conditions))

        time = [[time_domain_features[device][condition][time_feature] for condition in conditions] for device in devices]
        freq = [[frequency_domain_features[device][condition][frequency_feature] for condition in conditions] for device in devices]

        for i, device_time in enumerate(time):
            axs[0].plot(x, device_time, marker='o', linestyle='-', linewidth=2, label=devices[i])

        for i, device_freq in enumerate(freq):
            axs[1].plot(x, device_freq, marker='o', linestyle='-', linewidth=2, label=devices[i])

        axs[0].set_ylabel(time_feature.upper(), fontsize=25)
        axs[0].set_xlabel('Conditions', fontsize=25)
        axs[0].set_title("Time Domain - " + time_feature.upper(), fontsize=25, y=1)
        axs[1].set_ylabel(frequency_feature.upper(), fontsize=25)
        axs[1].set_xlabel('Conditions', fontsize=25)
        axs[1].set_title("Frequency Domain - " + frequency_feature.upper(), fontsize=25, y=1)

        axs[0].set_xticks(x)
        axs[0].set_xticklabels(conditions, fontsize=15)
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(conditions, fontsize=15)

        axs[0].legend()
        axs[1].legend()

        axs[0].grid(linestyle='--', alpha=0.8)
        axs[1].grid(linestyle='--', alpha=0.8)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.2)
        plt.show()

    def update_plots(*args):
        time_feature = time_feature_dropdown.value
        frequency_feature = frequency_feature_dropdown.value

        with out:
            clear_output(wait=True)
            plot_line_charts(time_feature, frequency_feature)

    time_feature_dropdown = widgets.Dropdown(
        options=time_features,
        value=time_features[0],
        description='Time Feature:',
        disabled=False,
    )

    frequency_feature_dropdown = widgets.Dropdown(
        options=frequency_features,
        value=frequency_features[0],
        description='Frequency Feature:',
        disabled=False,
    )

    time_feature_dropdown.observe(update_plots, names='value')
    frequency_feature_dropdown.observe(update_plots, names='value')

    widgets_box = widgets.HBox([time_feature_dropdown, frequency_feature_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_plots()

###########################################################################
###########################################################################
###########################################################################

def radar_plot (time_domain_features, criterion, devices, conditions):

    """
    This function creates a radar plot of the time domain features of two devices (criterion and another device) for a given
    condition. The function takes in the time domain features dictionary, criterion device name, another device name, and
    the condition name as input and outputs a radar plot using the plotly library.

    Parameters:
    -----------
    time_domain_features (dict): A dictionary containing the time domain features for each device and each condition
    criterion (str): The name of the criterion device for comparison
    device (str): The name of the device to be compared with the criterion device
    condition (str): The name of the condition for which the comparison is to be made

    Returns:
    -----------
    None
    """

    def plot_spider_chart(device, condition):
        rmssd_device = time_domain_features[device][condition]['rmssd']
        pnni_50_device = time_domain_features[device][condition]['pnni_50']
        mean_hr_device = time_domain_features[device][condition]['mean_hr']
        sdnn_device = time_domain_features[device][condition]['sdnn']

        rmssd_criterion = time_domain_features[criterion][condition]['rmssd']
        pnni_50_criterion = time_domain_features[criterion][condition]['pnni_50']
        mean_hr_criterion = time_domain_features[criterion][condition]['mean_hr']
        sdnn_criterion = time_domain_features[criterion][condition]['sdnn']

        features = ['rmssd', 'pnni_50', 'mean_hr', 'sdnn']
        data_criterion = [rmssd_criterion, pnni_50_criterion, mean_hr_criterion, sdnn_criterion]
        data_device = [rmssd_device, pnni_50_device, mean_hr_device, sdnn_device]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=data_criterion + [data_criterion[0]],
            theta=features + [features[0]],
            fill='toself',
            name=criterion.upper(),
            line_color='blue',
            opacity=0.5
        ))

        fig.add_trace(go.Scatterpolar(
            r=data_device + [data_device[0]],
            theta=features + [features[0]],
            fill='toself',
            name=device.upper(),
            line_color='orange',
            opacity=0.5
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[min(data_criterion + data_device) * 0.9, max(data_criterion + data_device) * 1.1]
                )),
            showlegend=True,
            title='Condition: {} - Devices: {} and {}'.format(condition.capitalize(), device.capitalize(), criterion.capitalize())
        )

        fig.show()

    def update_plot(*args):
        device = device_dropdown.value
        condition = condition_dropdown.value

        with out:
            clear_output(wait=True)
            plot_spider_chart(device, condition)

    device_dropdown = widgets.Dropdown(
        options=devices,
        value=devices[0],
        description='Device:',
        disabled=False,
    )

    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description='Condition:',
        disabled=False,
    )

    device_dropdown.observe(update_plot, names='value')
    condition_dropdown.observe(update_plot, names='value')

    widgets_box = widgets.HBox([device_dropdown, condition_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_plot()

###########################################################################
###########################################################################
###########################################################################

def display_changes (time_domain_features, frequency_domain_features, devices, conditions):

    """
    Display changes in time and frequency domain features for given devices and conditions.

    Parameters:
    -----------
    - time_domain_features (dict): dictionary containing time domain features for different devices and conditions
    - frequency_domain_features (dict): dictionary containing frequency domain features for different devices and conditions
    - devices (list): list of devices to be analyzed
    - conditions (list): list of conditions to be analyzed
    - time_feature (str): time domain feature to be analyzed (default 'rmssd')
    - frequency_feature (str): frequency domain feature to be analyzed (default 'hf')

    Returns:
    -----------
    - None

    Displays formatted DataFrames showing time and frequency domain features and their changes for given devices and conditions.
    """
    time_features = list (time_domain_features[devices[0]][conditions[0]].keys())
    frequency_features = list (frequency_domain_features[devices[0]][conditions[0]].keys())

    def display_formatted_dataframe(df, title):
        formatted_df = df.applymap(lambda x: f"{x:.2f}")
        try:
            from IPython.display import display, Markdown
            display(Markdown(f"**{title}**"))
            display(formatted_df)
            display(Markdown("---"))
        except:
            print(title)
            print(formatted_df)
            print("\n")

    def display_dataframes(time_feature, frequency_feature):
        time = [[time_domain_features[device][condition][time_feature] for condition in conditions] for device in devices]
        freq = [[frequency_domain_features[device][condition][frequency_feature] for condition in conditions] for device in devices]

        time_df = pd.DataFrame(time, columns=conditions, index=devices)
        time_df_changes = time_df.diff(axis=1)

        freq_df = pd.DataFrame(freq, columns=conditions, index=devices)
        freq_df_changes = freq_df.diff(axis=1)

        display_formatted_dataframe(time_df, "Time Domain Features - " + time_feature.upper())
        display_formatted_dataframe(time_df_changes, "Time Domain Feature Changes - " + time_feature.upper())
        display_formatted_dataframe(freq_df, "Frequency Domain Features - " + frequency_feature.upper())
        display_formatted_dataframe(freq_df_changes, "Frequency Domain Feature Changes - " + frequency_feature.upper())

    def update_dataframes(*args):
        time_feature = time_feature_dropdown.value
        frequency_feature = frequency_feature_dropdown.value

        with out:
            clear_output(wait=True)
            display_dataframes(time_feature, frequency_feature)

    time_feature_dropdown = widgets.Dropdown(
        options=time_features,
        value=time_features[0],
        description='Time Feature:',
        disabled=False,
    )

    frequency_feature_dropdown = widgets.Dropdown(
        options=frequency_features,
        value=frequency_features[0],
        description='Frequency Feature:',
        disabled=False,
    )

    time_feature_dropdown.observe(update_dataframes, names='value')
    frequency_feature_dropdown.observe(update_dataframes, names='value')

    widgets_box = widgets.HBox([time_feature_dropdown, frequency_feature_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_dataframes()

###########################################################################
###########################################################################
###########################################################################

def save_data (pp, path, time_domain_features, frequency_domain_features, data_pp, devices, conditions, events, artefact=None, nibi_before_cropping=None, nibi_after_cropping=None, save_as_csv=False):

    """
    Saves the processed data into a csv file.

    Parameters:
    -----------
    pp (str): Participant ID.
    path (str): Path to save the file.
    time_domain_features (dict): Time domain features.
    frequency_domain_features (dict): Frequency domain features.
    data_pp (dict): Raw data.
    devices (list): List of devices used.
    conditions (list): List of conditions in the study.
    events (dict): Dictionary containing the events and their corresponding timestamps.
    nbbi (dict, optional): Dictionary containing the number of detected and removed artefacts. Defaults to None.
    artefact (dict, optional): Dictionary containing the detected beat-to-beat intervals. Defaults to None.
    timebefore (dict, optional): Dictionary containing the time before an event. Defaults to None.
    timeafter (dict, optional): Dictionary containing the time after an event. Defaults to None.
    save_as_csv (bool, optional): Boolean value indicating whether to save the data as a CSV file. Defaults to False.

    Returns:
    -----------
    df_all (pandas.DataFrame): Dataframe containing all the processed data.
    """

    def create_df(device, time_features, freq_features, artefact, nibi_before_cropping, nibi_after_cropping, conditions=conditions):

        df1 = pd.DataFrame(time_features).transpose().reset_index(drop=True)
        df2 = pd.DataFrame(freq_features).transpose().reset_index(drop=True)
        df = df1.assign(**df2)
        df['artefact'] = artefact if artefact is not None else 'N/D'
        df['nibi_before_cropping'] = nibi_before_cropping if nibi_before_cropping is not None else 'N/D'
        df['nibi_after_cropping'] = nibi_after_cropping if nibi_after_cropping is not None else 'N/D'
        df['conditions'] = conditions
        df['device'] = device
        return df

    # extracting the values of number of detected and removed artefact in each device for each condition, and detected beat-to-beat intervals
    if artefact is not None:
        artefact_values = {device: list(artefact[device].values()) for device in devices}
    else:
        artefact_values = {device: 'N/D' for device in devices}

    if nibi_before_cropping is not None:
        nibi_before_cropping_values = {device: list(nibi_before_cropping[device].values()) for device in devices}
    else:
        nibi_before_cropping_values = {device: 'N/D' for device in devices}

    if nibi_after_cropping is not None:
        nibi_after_cropping_values = {device: list(nibi_after_cropping[device].values()) for device in devices}
    else:
        nibi_after_cropping_values = {device: 'N/D' for device in devices}

    # creating a df
    df = {device: {} for device in devices}
    for device in devices:
        df[device] = create_df(device, time_domain_features[device], frequency_domain_features[device],artefact_values[device],nibi_before_cropping_values[device],nibi_after_cropping_values[device])

    # putting all together
    df_all = pd.concat([df[device] for device in devices])

    df_all['pp'] = pp
    df_all['time'] = events['timestamp'][0]  # attaching the starting time point

    if save_as_csv:
        path_save = path + pp + ".csv"  # creating the path
        df_all.to_csv(path_save, index=False)
        print("Data Saved Successfully! We are done! :) ")

    return df_all

###########################################################################
###########################################################################
###########################################################################
