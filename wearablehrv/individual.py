# INDIVIDUAL pipeline for Wearablehrv package

import datetime
import os
import copy
import pickle
import pandas as pd
import tkinter as tk
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import ipywidgets as widgets
import matplotlib.dates as mdates
from ipywidgets import IntText, Dropdown, Output, HBox
from IPython.display import display, clear_output
from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis import get_time_domain_features
from hrvanalysis import get_frequency_domain_features
from avro.datafile import DataFileReader
from avro.io import DatumReader
from pathlib import Path

###########################################################################
###########################################################################
###########################################################################


def labfront_conversion(path, pp, file_name, device_name, date):
    """
    Converts Labfront data into a standardized CSV format, filtering by a specific date.

    This function reads CSV data from Labfront, focusing on the 'isoDate' and 'bbi' columns. It filters the data based on the provided date and renames relevant columns for standardization. The processed data is saved into a new CSV file named with the participant ID and device name.

    Parameters
    ----------
    path : str or Path
        The directory path pointing to the location of the Labfront data. Can be a string or a Path object.
    pp : str
        The unique ID of the participant for which the data is being processed.
    file_name : str
        The name of the Labfront file (including its extension) to be processed.
    device_name : str
        The name of the device used to collect the data, used in the resulting CSV's filename.
    date : str
        The specific date for which data should be extracted, in a format that can be parsed by pandas' to_datetime function (e.g., 'YYYY-MM-DD').

    Returns
    -------
    None
        The function saves the output directly to a CSV file in the specified path. The final CSV contains a timestamp column and a renamed 'bbi' column to 'rr', indicating respiratory rate.

    Notes
    -----
    The function prints a message upon successful completion of the conversion and saving process.
    """

    path = Path(path)
    labfront_file_path = path / file_name
    output_file_path = path / f"{pp}_{device_name}.csv"
    labfront = pd.read_csv(labfront_file_path + file_name, skiprows=5)
    # convert isoDate to datetime column
    labfront["isoDate"] = pd.to_datetime(labfront["isoDate"])
    # filter rows for the selected date
    labfront = labfront[labfront["isoDate"].dt.date == pd.to_datetime(date).date()]
    labfront = labfront[["unixTimestampInMs", "bbi"]]
    labfront = labfront.rename(columns={"unixTimestampInMs": "timestamp", "bbi": "rr"})
    labfront.to_csv(output_file_path, index=False)

    print("Dataset Successfully Converted and Saved in Your Path!")


###########################################################################
###########################################################################
###########################################################################


def empatica_conversion(path, pp):
    """
    Converts Empatica data from Avro format to a CSV file, focusing on the 'systolicPeaks' field.

    This function processes Empatica device data files stored in Avro format for a given participant.
    It extracts the 'systolicPeaks' field from these files, which includes peak times in nanoseconds.
    These times are converted to milliseconds, and the interbeat intervals (IBIs) are calculated.
    The resulting data, comprising timestamps (in milliseconds) and IBIs, is saved to a CSV file.

    Parameters
    ----------
    path : str or Path
        The directory path pointing to the location of the participant's Empatica data.
        This can be provided as a string or a Path object from the pathlib module.
    pp : str
        The unique ID of the participant whose Empatica data is to be converted.

    Returns
    -------
    None
        The function saves the output directly to a CSV file in the specified path. The CSV file is named
        using the participant's ID with '_empatica.csv' as the suffix and includes columns for 'timestamp'
        (milliseconds) and 'rr' (calculated IBIs).

    Notes
    -----
    - The expected directory structure for the Avro files is `<path>/<participant_id>_empatica/raw_data/v6`.
    - The function assumes Avro files contain the 'systolicPeaks' field with peak times in nanoseconds.
    - The final CSV file excludes the last timestamp since there's no corresponding IBI.
    """
    path = Path(path)
    avrofiles_path = path / f"{pp}_empatica/raw_data/v6"

    # Function to read systolicPeaks data from a single Avro file
    def read_systolic_peaks_from_file(avro_file):
        reader = DataFileReader(open(avro_file, "rb"), DatumReader())
        data = []
        for datum in reader:
            data = datum
        reader.close()

        systolic_peaks = data["rawData"]["systolicPeaks"]
        return systolic_peaks

    # Iterate through all Avro files in the given directory and combine the 'systolicPeaks' data
    combined_systolic_peaks_ns = []
    for avro_file in avrofiles_path.glob(
        "*.avro"
    ):  # Using Path.glob to find .avro files
        systolic_peaks = read_systolic_peaks_from_file(avro_file)
        combined_systolic_peaks_ns.extend(systolic_peaks["peaksTimeNanos"])

    # convert each nanosecond value to millisecond
    combined_systolic_peaks_ms = [x // 1000000 for x in combined_systolic_peaks_ns]
    # calculating the interbeat intervals
    ibis = np.diff(combined_systolic_peaks_ms)
    # create a DataFrame
    data = {
        "timestamp": combined_systolic_peaks_ms[:-1],
        "rr": ibis,
    }  # exclude the last timestamp since there's no corresponding rr_interval
    # turn the dataframe into a pandas dataframe
    df = pd.DataFrame(data)
    # saving the file
    output_path = path / f"{pp}_empatica.csv"  # Construct the output path using Path
    df.to_csv(output_path, index=False)

    print("Data saved succesfully in your path")


###########################################################################
###########################################################################
###########################################################################


def define_events(path, pp, conditions, already_saved=True, save_as_csv=False):
    """
    Defines and optionally saves events that occurred during a task for a specific participant.

    This function either reads previously saved events from a CSV file or allows the user to define events interactively through a GUI, depending on the 'already_saved' parameter. The events are associated with different conditions of a task and are stored in a pandas DataFrame. If 'save_as_csv' is True, the events DataFrame is saved as a CSV file in the specified path.

    Parameters
    ----------
    path : str or Path
        The path to the directory where the events file should be saved or has been saved.
        Can be a string or a Path object from the pathlib module.
    pp : str
        The ID of the participant for whom the events are being defined.
    conditions : list of str
        A list of strings representing the different conditions in the task.
    already_saved : bool, optional
        Indicates if the events file has already been saved. If True, reads the events from the specified path. If False, opens a GUI for interactive event definition. Default is True.
    save_as_csv : bool, optional
        Indicates if the events DataFrame should be saved as a CSV file. Default is False.

    Returns
    -------
    events : pandas.DataFrame
        A DataFrame containing the events data for the participant, with columns for timestamps, conditions, and whether the event marks the start or end of a condition.

    Notes
    -----
    - The CSV file structure, when saved, includes columns for timestamps, conditions, and event types (start/end of a condition).
    - If 'already_saved' is False, the function launches a GUI for the user to input events data interactively. The GUI requires the user to input start and end times for each condition.
    - The 'conditions' parameter should match the conditions expected to be found or entered for the events.
    """
    # Define the path to the events file
    path = Path(path)
    path_events = path / f"{pp}_events.csv"

    if (
        already_saved == True
    ):  # If there is already a pp_events.csv file with the spesified format saved

        # Read the events file into a DataFrame
        events = pd.read_csv(
            path_events,
            names=["timestamp", "conditions", "datapoint", "remove"],
            skiprows=1,
        )
        events = events.drop(
            events.columns[3], axis=1
        )  # removing the fourth column which is useless
        events = events.sort_values(
            by="timestamp", ascending=True
        )  # sorting the dataframe based on the times
        events = events.reset_index(drop=True)  # restting the indexing

    elif already_saved == False:  # Opening the GUI to enter the conditions

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
                start_time_var.set("")
                end_time_var.set("")
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
            events_df.columns = ["timestamp", "conditions", "datapoint"]
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
            condition_label.grid(row=i + 5, column=0)

            start_status_label = tk.Label(window, text="NOT FILLED", fg="red")
            start_status_label.grid(row=i + 5, column=1)
            start_status_labels.append(start_status_label)

            end_status_label = tk.Label(window, text="NOT FILLED", fg="red")
            end_status_label.grid(row=i + 5, column=2)
            end_status_labels.append(end_status_label)

        error_label = tk.Label(window, fg="red")
        error_label.grid(row=len(conditions) + 6, column=0, columnspan=2, sticky="W")

        # Create the "Save All" button
        save_all_button = tk.Button(window, text="Save All", command=save_all)
        save_all_button.grid(row=len(conditions) + 6, column=0, columnspan=2, pady=10)

        # Create the DataFrame to hold the events
        events_df = pd.DataFrame(columns=["timestamp", "conditions", "datapoint"])
        events = events_df

        # Run the GUI loop
        window.mainloop()

    if save_as_csv == True:
        events.to_csv(path_events, index=False)
        print("The event.csv is saved for futute use")

    print("Events imported succesfully")
    return events


###########################################################################
###########################################################################
###########################################################################


def import_data(path, pp, devices):
    """
    Imports participant-specific data from different devices and consolidates them into a dictionary.

    This function processes data files for a given participant from multiple devices. For data from the "vu" device,
    it reads from a text file (exported from VU-DAMS), selecting and renaming specific columns. For other devices,
    presumably recorded using HRV Logger, it expects CSV files and drops an unnecessary third column if present.
    Timestamps are standardized across devices in the final dataset.

    Parameters
    ----------
    path : str
        The directory path where the data files corresponding to the participant are located.
        Can be a string or a Path object from the pathlib module.
    pp : str
        The unique ID of the participant whose data is to be imported.
    devices : list of str
        Names of devices from which the data has been collected. Each device's data should be in a file named
        <participant_id>_<device_name>.<appropriate_extension>, where the extension is `.txt` for the "vu" device
        and `.csv` for other devices.

    Returns
    -------
    data : dict
        A dictionary where each key is a device name, and the associated value is a DataFrame containing the data
        from that device for the specified participant. The DataFrames have columns for timestamps (formatted as
        "HH:MM:SS.mmm") and rr intervals, with any irrelevant columns removed.

    Notes
    -----
    - The function handles the "vu" device data differently by reading from a text file and specifically focusing
    on "R-peak time" and "ibi" columns, which are then renamed to "timestamp" and "rr" respectively.
    - For other devices, it reads CSV files and drops the third column if it exists, standardizing the column names
    by stripping leading and trailing whitespace.
    - Timestamps for all devices are converted to pandas datetime objects and then reformatted to strings that
    represent the time in "HH:MM:SS.mmm".
    - It's assumed that timestamps from the "vu" device are in a different initial format than those from other
    devices, necessitating specific preprocessing steps for each.
    """

    path = Path(path)

    data = {
        device: {} for device in devices
    }  # creating an empty dictionary to store data from all devices
    # 2. reading data from devices:

    for device in devices:
        path_vu = path / f"{pp}_vu.txt"
        path_devices = path / f"{pp}_{device}.csv"

        if device == "vu":  # this is the text file exported from VU-DAMS
            data[device] = pd.read_csv(path_vu, sep="\t", skiprows=1)
            data[device] = data[device][
                ["R-peak time", "ibi"]
            ]  # Select only the "R-peak time" and "ibi" columns
            data[device] = data[device].rename(
                columns={"R-peak time": "timestamp", "ibi": "rr"}
            )  # Rename the columns

        else:  # these are the csv files recorded using HRV Logger
            data[device] = pd.read_csv(path_devices, header=0)
            try:  # this is because in HRV, there is a third column that needs to be dropped; but if someone makes a similar dataset, there is no need for this line of code
                data[device] = data[device].drop(
                    data[device].columns[2], axis=1
                )  # removing the third column which is useless
                data[device].columns = data[
                    device
                ].columns.str.strip()  # Strip leading/trailing whitespace from column labels
            except:
                data[device].columns = data[
                    device
                ].columns.str.strip()  # Strip leading/trailing whitespace from column labels

    # 3. changing dataset timestamps:
    for device in devices:

        if device == "vu":
            data[device]["timestamp"] = data[device]["timestamp"].apply(
                lambda x: x.split("/")[-1]
            )
            data[device]["timestamp"] = pd.to_datetime(
                data[device]["timestamp"], format="%H:%M:%S.%f"
            )

        else:

            for i in range(np.size(data[device]["timestamp"])):
                timestamp_float = float(data[device].loc[i, "timestamp"])
                data[device].loc[i, "timestamp"] = datetime.datetime.fromtimestamp(
                    timestamp_float / 1000
                )
            data[device]["timestamp"] = pd.to_datetime(data[device]["timestamp"])

        # Format timestamp column as string with format hh:mm:ss.mmm
        data[device]["timestamp"] = data[device]["timestamp"].apply(
            lambda x: x.strftime("%H:%M:%S.%f")[:-3]
        )

    print("Datasets imported succesfully")
    return data


###########################################################################
###########################################################################
###########################################################################


def lag_correction(data, devices, criterion):
    """
    Adjusts and visualizes the lag in timestamped data for different devices using an interactive GUI.

    This function employs IPython widgets to create a user interface that facilitates the adjustment of time lags
    between data recorded by different devices. Users can select a device, define a start and end time for analysis,
    and adjust the lag (in milliseconds) using a slider. The adjusted and original data are visualized in a plot for
    comparison. Changes can be saved to adjust the timestamps in the original dataset.

    Parameters
    ----------
    data : dict
        A dictionary where keys are device names and values are DataFrames containing the data for each device.
        Each DataFrame must have a 'timestamp' column and at least one other column for data values (e.g., 'rr').
    devices : list of str
        A list of strings representing the devices available for lag correction.
    criterion : str
        The criterion used for selecting the relevant data from 'data'. This is typically the name of a device or a
        specific dataset within `data` used as a reference for alignment.

    Notes
    -----
    - The GUI allows for dynamic selection of devices from the provided list and adjustment of the time range
    and lag with immediate visualization feedback.
    - The 'Start Time' and 'End Time' inputs determine the subset of data to be visualized and adjusted.
    - The lag slider supports a range of -20,000ms to +20,000ms and updates the plot in real-time as adjustments are made.
    - A 'Save Lag' button applies the lag adjustment to the data for the selected device and resets the slider, allowing
    for subsequent adjustments if necessary.
    - This function is designed to be used in a Jupyter Notebook environment where IPython widgets are supported.
    - Adjustments made to the data using this function are temporary and affect only the session's data unless explicitly
    saved or processed further.

    Examples
    --------
    To use `lag_correction` in a Jupyter Notebook:

    ```python
    data = {'device1': device1_df, 'device2': device2_df}
    devices = ['device1', 'device2']
    criterion = 'device1'  # Use device1 as the reference for alignment
    lag_correction(data, devices, criterion)
    ```
    """
    # Create the device dropdown widget
    device_dropdown = widgets.Dropdown(
        options=devices,
        description="Device:",
        disabled=False,
    )

    # Time input widgets for selecting start and end time
    start_time_picker = widgets.Text(
        value=data[criterion]["timestamp"].iloc[200],
        description="Start Time:",
        continuous_update=False,
    )
    end_time_picker = widgets.Text(
        value=data[criterion]["timestamp"].iloc[300],
        description="End Time:",
        continuous_update=False,
    )

    # Lag adjustment slider
    lag_slider = widgets.IntSlider(
        value=0,
        min=-20000,
        max=20000,
        step=1,
        description="Lag (ms):",
        disabled=False,
        continuous_update=True,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )
    lag_slider.layout = widgets.Layout(width="80%")

    # Output widget for the plot
    out = widgets.Output()

    # Update plot based on the current settings
    def update_plot(*args):
        with out:
            clear_output(True)
            selected_device = select_data(device_dropdown.value)
            selected_criterion = select_data(criterion)
            plot_data(selected_device, selected_criterion, lag_slider.value)

    # Select data based on the time range
    def select_data(device):
        selected_time_start = pd.to_datetime(
            start_time_picker.value, format="%H:%M:%S.%f"
        )
        selected_time_end = pd.to_datetime(end_time_picker.value, format="%H:%M:%S.%f")
        data_converted = copy.deepcopy(data)
        data_converted[device]["timestamp"] = pd.to_datetime(
            data_converted[device]["timestamp"], format="%H:%M:%S.%f"
        )
        return data_converted[device][
            (data_converted[device]["timestamp"] > selected_time_start)
            & (data_converted[device]["timestamp"] < selected_time_end)
        ]

    # Plot the data
    def plot_data(selected_device, selected_criterion, lag):
        lag = lag / 1000  # Convert milliseconds to seconds
        plt.figure(figsize=(17, 5))
        plt.plot(
            selected_device["timestamp"] + pd.Timedelta(seconds=lag),
            selected_device["rr"],
            "-o",
            label=device_dropdown.value,
        )
        plt.plot(
            selected_criterion["timestamp"],
            selected_criterion["rr"],
            "-o",
            label=criterion,
        )
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.gca().xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=3, maxticks=7)
        )
        plt.xlabel("Timestamp", fontsize=12)
        plt.ylabel("RR Intervals", fontsize=12)
        plt.grid()
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()

    # Save lag
    def save_lag(b):
        lag = lag_slider.value / 1000
        data[device_dropdown.value]["timestamp"] = data[device_dropdown.value][
            "timestamp"
        ] + pd.Timedelta(seconds=lag)
        baseline = pd.Timestamp("1900-01-01")
        data[device_dropdown.value]["timestamp"] = (
            baseline + data[device_dropdown.value]["timestamp"]
        )
        data[device_dropdown.value]["timestamp"] = data[device_dropdown.value][
            "timestamp"
        ].apply(lambda x: x.strftime("%H:%M:%S.%f")[:-3])
        lag_slider.value = 0  # Reset the lag slider to zero
        print(f"Lag of {lag} seconds applied to {device_dropdown.value}")

    save_button = widgets.Button(description="Save Lag")
    save_button.on_click(save_lag)

    # Observe changes and update plot
    device_dropdown.observe(update_plot, names="value")
    start_time_picker.observe(update_plot, names="value")
    end_time_picker.observe(update_plot, names="value")
    lag_slider.observe(update_plot, names="value")

    # Layout the widgets
    input_widgets = widgets.VBox(
        [device_dropdown, start_time_picker, end_time_picker, lag_slider, save_button]
    )
    gui = widgets.VBox([input_widgets, out])

    # Update plot
    update_plot()

    # Display the GUI
    display(gui)


###########################################################################
###########################################################################
###########################################################################


def chop_data(data, conditions, events, devices):
    """
    Chops the data from different devices into separate segments based on specified events.

    This function segments the raw data for each device into smaller parts corresponding to different conditions
    of a task, as defined by start and end times in an events DataFrame. The segmented data is organized in a new
    dictionary, preserving the device and condition structure, and includes both timestamps and corresponding data points.

    Parameters
    ----------
    data : dict
        A dictionary containing the raw data for all devices. Each key is a device name, and the value is a DataFrame
        with a 'timestamp' column and one or more data columns (e.g., 'rr' for respiratory rate).
    conditions : list of str
        A list of strings representing the different conditions in the task. Each condition is expected to be present
        in the 'conditions' column of the events DataFrame.
    events : pandas.DataFrame
        A DataFrame containing the event data for the participant, with at least 'timestamp', 'conditions', and
        'datapoint' columns, where 'datapoint' indicates the start or end of a condition.
    devices : list of str
        A list of strings representing the different devices used to collect the data. Each device name should correspond
        to a key in the `data` dictionary.

    Returns
    -------
    data_chopped : dict
        A dictionary containing the chopped data for all devices and conditions. Each entry under a device key is a
        nested dictionary where each key is a condition name and the value is a DataFrame of the data for that condition,
        including timestamps within the start and end times defined in the events DataFrame.

    Notes
    -----
    - The function expects the timestamps in the events DataFrame and the data DataFrames to be compatible and formatted
    similarly to allow for accurate comparison and filtering.
    - Start and end times for each condition are extracted from the events DataFrame and used to filter the data
    for each device into segments corresponding to those conditions.
    - The output dictionary structure allows for easy access to data by device and condition, facilitating further analysis.

    """

    # it contains the begening and end of each condition
    eventchopped = {}  # it contains the beginning and end of each condition

    for condition in conditions:
        start = events.loc[
            (events["conditions"] == condition) & (events["datapoint"] == "start")
        ]["timestamp"].iloc[0]
        end = events.loc[
            (events["conditions"] == condition) & (events["datapoint"] == "end")
        ]["timestamp"].iloc[0]
        start = datetime.datetime.strptime(start, "%H:%M:%S")
        end = datetime.datetime.strptime(end, "%H:%M:%S")
        start = start.strftime("%H:%M:%S.%f")[:-3]
        end = end.strftime("%H:%M:%S.%f")[:-3]
        eventchopped[condition] = (start, end)

    # a new dictionary that contains chopped (based on events) rr intervals for each device and condition
    data_chopped = {
        device: {condition: {} for condition in conditions} for device in devices
    }

    for device in devices:
        for condition in conditions:
            filtered_rows = data[device][
                (data[device]["timestamp"] >= eventchopped[condition][0])
                & (data[device]["timestamp"] < eventchopped[condition][1])
            ].copy()
            filtered_rows["timestamp"] = pd.to_datetime(
                filtered_rows["timestamp"]
            )  # convert to datetime format
            data_chopped[device][
                condition
            ] = filtered_rows  # this one now contains both rr intervals and timestamps

    print("Data are chopped based on the events succesfully")
    return data_chopped


###########################################################################
###########################################################################
###########################################################################


def calculate_ibi(data_chopped, devices, conditions):
    """
    Calculates the number of Inter-Beat Intervals (IBI) for each condition and device from segmented data.

    This function iterates over the provided segmented data for each device and condition, calculating the number
    of IBIs. The IBIs are expected to be represented as the number of rows in each condition's data segment for a device.

    Parameters
    ----------
    data_chopped : dict
        A dictionary containing the chopped data for all devices and conditions. Each entry under a device key should be a
        nested dictionary where each key is a condition name, and the value is a DataFrame of the segmented data,
        including timestamps and corresponding data points (e.g., 'rr' for respiratory rate).
    conditions : list of str
        A list of strings representing the different conditions in the task. These should match the keys in the
        nested dictionaries under each device key in `data_chopped`.
    devices : list of str
        A list of strings representing the different devices used to collect the data. Each device name should correspond
        to a key in the `data_chopped` dictionary.

    Returns
    -------
    nibis : dict
        A dictionary where each entry is keyed by device name, and the value is another dictionary with conditions as keys.
        The values in this nested dictionary are integers representing the count of data points (or IBIs) for each condition
        within the data segment of the specified device.

    Notes
    -----
    - This function assumes that the input data in `data_chopped` is properly segmented according to the specified conditions
    and that each segment's data points represent IBIs.
    - The output dictionary can be used to compare the number of IBIs across different conditions and devices, providing a
    basis for further statistical analysis or comparison.
    """
    nibis = {device: {condition: {} for condition in conditions} for device in devices}

    for device in devices:
        for condition in conditions:
            nibis[device][condition] = np.shape(data_chopped[device][condition])[0]

    print(
        "The number of calculated IBI per condition per deivice has been succesfully calculated"
    )
    return nibis


###########################################################################
###########################################################################
###########################################################################


def visual_inspection(data_chopped, devices, conditions, criterion):
    """
    Allows for visual inspection and manual modification of the RR interval data through an interactive GUI.

    This function provides a graphical user interface for visually inspecting and manually adjusting RR interval data.
    Users can select a device and condition, visualize the RR intervals alongside those of a criterion device, and apply
    manual corrections for lag and data trimming. Adjustments include lag correction in seconds or milliseconds and
    trimming data by specifying start and end points. The interface supports both individual and full lag corrections
    across all conditions, with modifications directly applied to the input `data_chopped` dictionary.

    Parameters
    ----------
    data_chopped : dict
        A dictionary containing the chopped RR interval data for all devices and conditions. Each device's data
        is stored under its name as a key, with nested dictionaries for each condition containing DataFrames of
        timestamps and RR intervals.
    devices : list of str
        A list of strings representing the different devices used to collect the data. Each string must correspond
        to a key in `data_chopped`.
    conditions : list of str
        A list of strings representing the different conditions in the task. Each condition must be a key within
        the nested dictionaries under each device key in `data_chopped`.
    criterion : str
        A string representing the device used as the criterion for comparison. This should be one of the devices
        listed in `devices` and used to reference the expected correct timing of events for comparison and correction.

    Notes
    -----
    - The GUI includes dropdown menus for selecting devices and conditions, sliders for adjusting lag and specifying
    start and end points for trimming, and buttons for applying changes.
    - Lag adjustments can be made in seconds or milliseconds, and users can choose between individual and full lag
    correction modes. Individual mode applies adjustments only to the selected condition, whereas full mode applies
    the same adjustment across all conditions.
    - Data trimming allows users to specify start and end points within the RR interval data to exclude irrelevant or
    erroneous data segments.
    - Changes made through the GUI are applied directly to the `data_chopped` dictionary and can be saved permanently by
    the user if desired.
    - This function is designed to facilitate data preprocessing by enabling detailed examination and correction of RR
    interval data prior to analysis.
    """

    # Define the function that creates the plot
    def plot_rr_intervals(
        device, condition, lag, device_start, device_end, criterion_start, criterion_end
    ):
        # Trimming
        trim_device = slice(device_start, device_end)
        trim_criterion = slice(criterion_start, criterion_end)

        # Get the RR intervals and timestamps for the selected condition and device
        ppg_rr = data_chopped[device][condition]["rr"][trim_device]
        ppg_timestamp = data_chopped[device][condition]["timestamp"][trim_device]

        # Get the RR intervals and timestamps for the criterion (vu)
        criterion_rr = data_chopped[criterion][condition]["rr"][trim_criterion]
        criterion_timestamp = data_chopped[criterion][condition]["timestamp"][
            trim_criterion
        ]

        # Adjust lag based on precision (seconds or milliseconds)
        if precision_dropdown.value == "Milliseconds":
            lag = lag / 1000  # Convert milliseconds to seconds

        # Shift the timestamps of the PPG device by the lag amount
        ppg_timestamp = ppg_timestamp + pd.Timedelta(seconds=lag)

        # Create a figure with a larger size
        plt.figure(figsize=(17, 5))

        # Plot the RR intervals for the selected device and the criterion
        plt.plot(ppg_timestamp, ppg_rr, "-o", color="red", label=device, markersize=7)
        plt.plot(
            criterion_timestamp,
            criterion_rr,
            "-o",
            color="black",
            label=criterion,
            markersize=7,
        )

        # Add grid lines to the plot
        plt.grid()

        # Set the title and axis labels
        plt.title("Beat-to-beat intervals for {} condition".format(condition))
        plt.xlabel("Timestamp", fontsize=20)
        plt.ylabel("RR Intervals", fontsize=20)

        # Format the x-axis as dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.gca().xaxis.set_major_locator(
            mdates.AutoDateLocator(minticks=3, maxticks=7)
        )

        # Rotate the x-axis labels
        plt.xticks(rotation=90)

        # Add a legend to the plot
        plt.legend()

        # Show the plot
        plt.show()

    # Define the function that saves the lag
    def save_lag(lag):
        # Adjust lag based on precision (seconds or milliseconds)
        if precision_dropdown.value == "Milliseconds":
            lag = lag / 1000  # Convert milliseconds to seconds

        if correction_mode_dropdown.value == "Full Lag Correction":
            for condition in conditions:
                ppg_timestamp = data_chopped[device_dropdown.value][condition][
                    "timestamp"
                ]
                data_chopped[device_dropdown.value][condition]["timestamp"] = (
                    ppg_timestamp + pd.Timedelta(seconds=lag)
                )
        else:
            ppg_timestamp = data_chopped[device_dropdown.value][
                condition_dropdown.value
            ]["timestamp"]
            data_chopped[device_dropdown.value][condition_dropdown.value][
                "timestamp"
            ] = ppg_timestamp + pd.Timedelta(seconds=lag)

        # Reset the lag to 0
        lag_slider.value = 0

        # Display a message to inform the user that the data has been modified
        print("Data has been modified with a lag of {} seconds.".format(lag))

    def save_crop(device_start, device_end, criterion_start, criterion_end):
        data_chopped[device_dropdown.value][condition_dropdown.value]["rr"] = (
            data_chopped[device_dropdown.value][condition_dropdown.value]["rr"][
                device_start:device_end
            ]
        )
        data_chopped[device_dropdown.value][condition_dropdown.value]["timestamp"] = (
            data_chopped[device_dropdown.value][condition_dropdown.value]["timestamp"][
                device_start:device_end
            ]
        )

        data_chopped[criterion][condition_dropdown.value]["rr"] = data_chopped[
            criterion
        ][condition_dropdown.value]["rr"][criterion_start:criterion_end]
        data_chopped[criterion][condition_dropdown.value]["timestamp"] = data_chopped[
            criterion
        ][condition_dropdown.value]["timestamp"][criterion_start:criterion_end]

        # Drop any rows with NaN or NaT values in RR intervals and timestamp columns for both device and criterion
        data_chopped[device_dropdown.value][condition_dropdown.value].dropna(
            subset=["rr", "timestamp"], inplace=True
        )
        data_chopped[criterion][condition_dropdown.value].dropna(
            subset=["rr", "timestamp"], inplace=True
        )

        update_device_condition()
        print("Cropped data has been saved.")

    def update_device_condition(*args):
        lag_slider.value = 0

        device_start_slider.value = 0
        device_start_slider.max = (
            len(data_chopped[device_dropdown.value][condition_dropdown.value]["rr"]) - 1
        )
        device_end_slider.max = len(
            data_chopped[device_dropdown.value][condition_dropdown.value]["rr"]
        )
        device_end_slider.value = len(
            data_chopped[device_dropdown.value][condition_dropdown.value]["rr"]
        )

        criterion_start_slider.value = 0
        criterion_start_slider.max = (
            len(data_chopped[criterion][condition_dropdown.value]["rr"]) - 1
        )
        criterion_end_slider.max = len(
            data_chopped[criterion][condition_dropdown.value]["rr"]
        )
        criterion_end_slider.value = len(
            data_chopped[criterion][condition_dropdown.value]["rr"]
        )

        with out:
            clear_output(True)
            plot_rr_intervals(
                device_dropdown.value,
                condition_dropdown.value,
                lag_slider.value,
                device_start_slider.value,
                device_end_slider.value,
                criterion_start_slider.value,
                criterion_end_slider.value,
            )

    def update_plot(change):
        with out:
            clear_output(True)
            plot_rr_intervals(
                device_dropdown.value,
                condition_dropdown.value,
                lag_slider.value,
                device_start_slider.value,
                device_end_slider.value,
                criterion_start_slider.value,
                criterion_end_slider.value,
            )

    # Create two sets of start and end slider widgets for device and criterion
    device_start_slider = widgets.IntSlider(
        min=0,
        max=len(data_chopped[devices[0]][conditions[0]]["rr"]) - 1,
        value=0,
        description="Device Start:",
        continuous_update=False,
    )
    device_end_slider = widgets.IntSlider(
        min=1,
        max=len(data_chopped[devices[0]][conditions[0]]["rr"]),
        value=len(data_chopped[devices[0]][conditions[0]]["rr"]),
        description="Device End:",
        continuous_update=False,
    )

    criterion_start_slider = widgets.IntSlider(
        min=0,
        max=len(data_chopped[criterion][conditions[0]]["rr"]) - 1,
        value=0,
        description="Criterion Start:",
        continuous_update=False,
    )
    criterion_end_slider = widgets.IntSlider(
        min=1,
        max=len(data_chopped[criterion][conditions[0]]["rr"]),
        value=len(data_chopped[criterion][conditions[0]]["rr"]),
        description="Criterion End:",
        continuous_update=False,
    )

    # Define the widget for lag correction mode
    correction_mode_dropdown = widgets.Dropdown(
        options=["Individual Lag Correction", "Full Lag Correction"],
        value="Individual Lag Correction",
        description="Correction Mode:",
        disabled=False,
    )

    # Define the precision dropdown widget
    precision_dropdown = widgets.Dropdown(
        options=["Seconds", "Milliseconds"],
        value="Seconds",
        description="Precision:",
        disabled=False,
    )

    # Function to update lag_slider parameters based on precision
    def update_lag_slider_precision(*args):
        if precision_dropdown.value == "Seconds":
            lag_slider.min = -20
            lag_slider.max = 20
            lag_slider.value = 0
            lag_slider.description = "Lag (s):"
            lag_slider.readout_format = "d"
        else:
            lag_slider.min = -20000
            lag_slider.max = 20000
            lag_slider.value = 0
            lag_slider.description = "Lag (ms):"
            lag_slider.readout_format = "d"

    # Observe changes in precision dropdown and update lag slider accordingly
    precision_dropdown.observe(update_lag_slider_precision, names="value")

    # Create the device dropdown widget
    device_dropdown = widgets.Dropdown(
        options=devices,
        value=devices[0],
        description="Device:",
        disabled=False,
    )

    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description="Condition:",
        disabled=False,
    )

    lag_slider = widgets.IntSlider(
        value=0,
        min=-20,
        max=20,
        step=1,
        description="Lag (s):",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )
    lag_slider.layout = widgets.Layout(width="80%")

    # Define the buttons
    save_button = widgets.Button(
        description="Save Lag",
        disabled=False,
        button_style="",
        tooltip="Click me",
        icon="save",
    )
    save_crop_button = widgets.Button(
        description="Save Crop",
        disabled=False,
        button_style="",
        tooltip="Click me",
        icon="crop",
    )

    # Create the output widget
    out = widgets.Output()

    # Define the trimming box
    trimming_box = widgets.VBox(
        children=[
            widgets.HBox(children=[device_start_slider, device_end_slider]),
            widgets.HBox(children=[criterion_start_slider, criterion_end_slider]),
        ]
    )

    # Register event listeners
    lag_slider.observe(update_plot, names="value")
    device_start_slider.observe(update_plot, names="value")
    device_end_slider.observe(update_plot, names="value")
    criterion_start_slider.observe(update_plot, names="value")
    criterion_end_slider.observe(update_plot, names="value")
    device_dropdown.observe(update_device_condition, names="value")
    condition_dropdown.observe(update_device_condition, names="value")
    save_button.on_click(lambda b: save_lag(lag_slider.value))
    save_crop_button.on_click(
        lambda b: save_crop(
            device_start_slider.value,
            device_end_slider.value,
            criterion_start_slider.value,
            criterion_end_slider.value,
        )
    )

    # Create the GUI layout

    widgets_box = widgets.VBox(
        children=[
            widgets.HBox(children=[device_dropdown, condition_dropdown]),
            widgets.HBox(children=[correction_mode_dropdown, precision_dropdown]),
            widgets.HBox(
                children=[
                    lag_slider,
                    widgets.VBox(children=[save_button, save_crop_button]),
                ]
            ),
            trimming_box,
        ]
    )

    output_box = widgets.VBox(children=[out])

    gui_box_layout = widgets.Layout(
        display="flex", flex_flow="column", align_items="stretch", width="80%"
    )
    gui_box = widgets.Box(children=[widgets_box, output_box], layout=gui_box_layout)

    # Call the function to render the initial plot inside the output widget
    update_device_condition()

    # Display the GUI
    display(gui_box)


###########################################################################
###########################################################################
###########################################################################


def save_backup(pp, path, data_chopped):
    """
    This function saves the processed and chopped data into a pickle file.

    Parameters:
    -----------
    pp : str
        The name of the preprocessing applied to the data.
    path : str or Path
        The path where the pickle file will be saved. Can be a string or a Path object from the pathlib module.
    data_chopped : dict
        A dictionary containing the chopped data that has been processed.

    Returns:
    --------
    None

    This function will print a message indicating if the data was saved successfully. It does not have a return value.
    """

    path = Path(path)
    filename = path / f"{pp}_data_chopped.pkl"
    with open(filename, "wb") as file:
        pickle.dump(data_chopped, file)

    print("Data saved successfully!")


###########################################################################
###########################################################################
###########################################################################


def import_backup(pp, path):
    """
    Loads the processed and chopped data from a pickle file.

    This function deserializes the data from a pickle file containing processed and segmented data,
    allowing for retrieval and further analysis. The file is identified with the preprocessing identifier.

    Parameters
    ----------
    pp : str
        The identifier or name of the preprocessing applied to the data, used to identify the output file.
    path : str or Path
        The directory path where the pickle file will be saved. Can be provided as a string or a Path object from the pathlib module.

    Returns
    -------
    data_chopped : dict
        A dictionary containing the chopped data that has undergone preprocessing. This data is
        deserialized and loaded from the file.

    Notes
    -----
    - The input filename is constructed using the preprocessing identifier and a '_data_chopped.pkl' suffix.
    - A message is printed upon successful loading of the data, indicating the completion of the operation.
    """
    path = Path(path)
    filename = path / f"{pp}_data_chopped.pkl"
    with open(filename, "rb") as file:
        data_chopped = pickle.load(file)

    print("Data loaded successfully!")
    return data_chopped


###########################################################################
###########################################################################
###########################################################################


def pre_processing(
    data_chopped,
    devices,
    conditions,
    method="karlsson",
    custom_removing_rule=0.25,
    low_rri=300,
    high_rri=2000,
):
    """
    Preprocesses the RR intervals data using specified methodologies for outlier removal, interpolation of missing values,
    and ectopic beat removal, and stores the preprocessed data in a new dictionary.

    This function processes the RR interval data from various devices and conditions. It applies outlier detection
    based on specified lower and upper threshold values, interpolates missing RR interval values, removes ectopic beats
    using a specified method, and interpolates again to ensure a continuous dataset. The preprocessing steps are
    executed using functions from the HRV Analysis package (https://aura-healthcare.github.io/hrv-analysis/).

    Parameters
    ----------
    data_chopped : dict
        A dictionary containing the chopped RR interval data for all devices and conditions. Each device's data is
        stored under its name as a key, with nested dictionaries for each condition containing lists or DataFrames of
        RR intervals.
    devices : list of str
        A list of strings representing the different devices used to collect the data.
    conditions : list of str
        A list of strings representing the different conditions in the task.
    method : str, optional
        The method used for removing ectopic beats. Defaults to "karlsson".
    custom_removing_rule : float, optional
        A custom rule parameter for the ectopic beat removal method, specified as a float. Defaults to 0.25.
    low_rri : int, optional
        The lower threshold for RR interval outlier detection, in milliseconds. Defaults to 300.
    high_rri : int, optional
        The upper threshold for RR interval outlier detection, in milliseconds. Defaults to 2000.

    Returns
    -------
    dict
        A dictionary containing the preprocessed RR intervals data for each device and condition, structured similarly
        to `data_chopped`.

    Notes
    -----
    - The function leverages external functions from an HRV analysis package for data cleaning and preprocessing.
    These functions include `remove_outliers`, `interpolate_nan_values`, and `remove_ectopic_beats`.
    - If preprocessing fails for a specific condition on a device, an error message is printed, and the function
    continues processing the remaining data.
    - The original `data_chopped` dictionary is modified to contain only the 'rr' values, discarding the timestamps.
    This simplification is made prior to applying the preprocessing steps.
    - The function returns the preprocessed data along with the modified `data_chopped` for reference or further use.
    """

    # Turning the dataset into RR intervals only: now that we have visualized the data and learned about its structure, we can simplify the next steps by discarding the x-axis (time axis).
    data_chopped = {
        device: {
            condition: list(data_chopped[device][condition]["rr"])
            for condition in conditions
        }
        for device in devices
    }

    # empty dic to store the pre-processed  RR intervals for each condition for each device
    data_pp = {
        device: {condition: {} for condition in conditions} for device in devices
    }

    def preprocess_rr_intervals(rr_intervals):
        """ " The four following lines come from the HRV analysis package;
        here I stack them in a function. The input is a given RR interval"""

        rr_intervals_without_outliers = remove_outliers(
            rr_intervals=rr_intervals, low_rri=low_rri, high_rri=high_rri
        )
        interpolated_rr_intervals = interpolate_nan_values(
            rr_intervals=rr_intervals_without_outliers, interpolation_method="linear"
        )
        nn_intervals_list = remove_ectopic_beats(
            rr_intervals=interpolated_rr_intervals,
            method=method,
            custom_removing_rule=custom_removing_rule,
        )
        interpolated_nn_intervals = interpolate_nan_values(
            rr_intervals=nn_intervals_list
        )
        return interpolated_nn_intervals

    # storing the pre-processed rr values for each condition in a dictionary
    for device in devices:
        for condition in conditions:
            try:  # this is because for whatever reason there might not be possible to do the analysis
                data_pp[device][condition] = preprocess_rr_intervals(
                    data_chopped[device][condition]
                )
            except:
                print(
                    "Error: it was not possible to preprocess the data in {} condition for {} device".format(
                        condition, device
                    )
                )
                continue
    print(
        "Pre-processed RR intervals succesfully stored in dictionaries for each condition"
    )
    return data_pp, data_chopped


###########################################################################
###########################################################################
###########################################################################


def calculate_artefact(data_chopped, data_pp, devices, conditions):
    """
    Calculates the number of artifacts in RR interval data for each device and condition by comparing original (chopped)
    data with preprocessed data.

    Artifacts are identified as discrepancies between the original and preprocessed RR intervals for each device and
    condition. This function counts the instances where RR intervals from the original data do not match those in the
    preprocessed data, assuming that preprocessing corrects or removes artifacts, thus indicating their presence in the
    original dataset.

    Parameters
    ----------
    data_chopped : dict
        A dictionary containing the chopped RR interval data for all devices and conditions. Each entry under a device
        key is a nested dictionary where each key is a condition name, and the value is a list or DataFrame of RR intervals.
    data_pp : dict
        A dictionary containing the preprocessed RR interval data for all devices and conditions, structured similarly
        to `data_chopped`. Preprocessing may include outlier removal, interpolation, and ectopic beat correction.
    devices : list of str
        A list of strings representing the different devices used to collect the data.
    conditions : list of str
        A list of strings representing the different conditions in the task.

    Returns
    -------
    artefact : dict
        A dictionary where each entry is keyed by device name, with nested dictionaries for each condition. The value for
        each condition is an integer representing the count of detected artifacts in the RR interval data.

    Notes
    -----
    - The comparison is made element-wise between the lists of RR intervals in `data_chopped` and `data_pp` for each
    device and condition. Differences are counted as artifacts.
    - This method assumes that all discrepancies between the original and preprocessed data are due to artifacts that
    were corrected or removed during preprocessing.
    - The function prints a message upon successful calculation of artifacts, summarizing the operation's outcome.
    """

    artefact = {
        device: {condition: {} for condition in conditions} for device in devices
    }

    for device in devices:
        for condition in conditions:
            artefact[device][condition] = sum(
                [
                    1
                    for x, y in zip(
                        data_chopped[device][condition], data_pp[device][condition]
                    )
                    if x != y
                ]
            )

    print(
        "The number of detected artefact per condition per deivice has been succesfully calculated"
    )
    return artefact


###########################################################################
###########################################################################
###########################################################################


def ibi_comparison_plot(
    data_chopped, data_pp, devices, conditions, criterion, width=20, height=10
):
    """
    Plots a comparison of the original (chopped) and pre-processed RR intervals for selected devices and conditions against
    the criterion device's data. This visual representation helps in assessing the effect of preprocessing steps on the RR
    interval data. Each selected device's RR intervals are plotted alongside the criterion device's data for comparison.

    Parameters
    ----------
    data_chopped : dict
        A dictionary containing the chopped RR interval data for all devices and conditions, with each device's data
        being a nested dictionary where each key is a condition name, and the value is a list of RR intervals.
    data_pp : dict
        A dictionary containing the preprocessed RR interval data for all devices and conditions, structured similarly
        to `data_chopped`. Preprocessing may include outlier removal, interpolation, and ectopic beat correction.
    devices : list of str
        A list of strings representing the different devices used to collect the data.
    conditions : list of str
        A list of strings representing the different conditions in the task.
    criterion : str
        The device name used as the reference or criterion for comparison. This device's data is plotted as the
        benchmark for assessing preprocessing effects.
    width : int, optional
        The width of the plot in inches. Default is 20.
    height : int, optional
        The height of the plot in inches. Default is 10.

    Notes
    -----
    - The function utilizes matplotlib for plotting and IPython widgets for interactive selection of devices and
    conditions to be visualized.
    - Two plots are generated for each selected condition: one for the selected device and one for the criterion device,
    each comparing original against preprocessed RR intervals.
    - This function is designed to be used in a Jupyter notebook environment where the interactive features provided
    by IPython widgets can be fully utilized.
    """

    # Define the function that updates the plot
    def update_plot(*args):
        condition = condition_dropdown.value
        device = device_dropdown.value

        with out:
            clear_output(wait=True)
            fig, axs = plt.subplots(2, 1, figsize=(width, height))

            axs[0].plot(
                data_chopped[device][condition], "-o", color="red", label="Original"
            )
            axs[0].plot(
                data_pp[device][condition], "-o", color="black", label="Pre-processed"
            )
            axs[0].grid()
            axs[0].set_title(
                "Beat-to-beat intervals for {} condition in {} device".format(
                    condition.upper(), device.upper()
                )
            )
            axs[0].set_xlabel("Beats")
            axs[0].set_ylabel("RR Intervals")
            axs[0].legend()

            axs[1].plot(
                data_chopped[criterion][condition], "-o", color="red", label="Original"
            )
            axs[1].plot(
                data_pp[criterion][condition],
                "-o",
                color="black",
                label="Pre-processed",
            )
            axs[1].grid()
            axs[1].set_title(
                "Beat-to-beat intervals for {} condition in {} device".format(
                    condition.upper(), criterion.upper()
                )
            )
            axs[1].set_xlabel("Beats")
            axs[1].set_ylabel("RR Intervals")
            axs[1].legend()

            plt.subplots_adjust(hspace=0.5)
            plt.show()

    # Create the device dropdown widget
    device_dropdown = widgets.Dropdown(
        options=devices,
        value=devices[0],
        description="Device:",
        disabled=False,
    )

    # Create the condition dropdown widget
    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description="Condition:",
        disabled=False,
    )

    device_dropdown.observe(update_plot, names="value")
    condition_dropdown.observe(update_plot, names="value")

    widgets_box = widgets.HBox([device_dropdown, condition_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_plot()


###########################################################################
###########################################################################
###########################################################################


def data_analysis(data_pp, devices, conditions):
    """
    Calculates time domain and frequency domain HRV (Heart Rate Variability) features for pre-processed RR interval data
    using the hrvanalysis package.

    This function iterates over pre-processed RR interval data for each device and condition, applying HRV analysis to
    extract meaningful statistical measures. These measures include time domain features, which describe the variability
    in time between successive heartbeats, and frequency domain features, which provide insights into the oscillatory
    components of heart rate signals under different physiological states.

    Parameters
    ----------
    data_pp : dict
        A dictionary containing the pre-processed RR interval data for all devices and conditions. Each entry should
        be a list of RR intervals in milliseconds.
    devices : list of str
        A list of strings representing the different devices used to collect the RR interval data.
    conditions : list of str
        A list of strings representing the different conditions under which the RR interval data was collected.

    Returns
    -------
    time_domain_features : dict
        A dictionary where each key is a device name, each value is a nested dictionary with conditions as keys, and
        the corresponding values are dictionaries of time domain HRV features calculated for that device and condition.
    frequency_domain_features : dict
        A dictionary structured similarly to `time_domain_features`, but containing frequency domain HRV features instead.

    Notes
    -----
    - The HRV analysis is performed using the `get_time_domain_features` and `get_frequency_domain_features` functions
    from the hrvanalysis package.
    - In case analysis cannot be performed for a given condition on a device (e.g., due to insufficient or unsuitable
    data), an error message will be printed, and the function will proceed with the remaining data.
    - This function is designed for use with RR interval data that has already undergone preprocessing steps such as
    outlier removal, interpolation, and ectopic beat correction.
    """

    # calculating all the time domain and frequency domain features by the hrvanalysis package
    time_domain_features = {
        device: {condition: {} for condition in conditions} for device in devices
    }
    frequency_domain_features = {
        device: {condition: {} for condition in conditions} for device in devices
    }

    for device in devices:
        for condition in conditions:
            try:  # this is because for whatever reason there might not be possible to do the analysis
                time_domain_features[device][condition] = get_time_domain_features(
                    data_pp[device][condition]
                )
                frequency_domain_features[device][condition] = (
                    get_frequency_domain_features(data_pp[device][condition])
                )
            except:
                print(
                    "Error: it was not possible to analyse the data in {} condition for {} device".format(
                        condition, device
                    )
                )
                continue

    print(
        "All time domain and frequency domain measures succesfully calculated and stored"
    )
    return time_domain_features, frequency_domain_features


###########################################################################
###########################################################################
###########################################################################


def result_comparison_plot(
    data_chopped,
    time_domain_features,
    frequency_domain_features,
    devices,
    conditions,
    bar_width=0.20,
    width=20,
    height=25,
):
    """
    Creates comparison bar charts for time and frequency domain measures of HRV for each device against the criterion device, both for original and pre-processed data.

    This function visualizes the differences in HRV measures before and after preprocessing to assess the impact of data cleaning techniques. It generates bar charts comparing time and frequency domain measures across devices for selected conditions. The function leverages interactive widgets for selecting HRV features and conditions to be plotted.

    Parameters
    ----------
    data_chopped : dict
        A dictionary containing the raw HRV data (RR intervals) for each device and condition.
    time_domain_features : dict
        A dictionary containing time domain HRV measures for each device and condition, obtained after preprocessing the RR interval data.
    frequency_domain_features : dict
        A dictionary containing frequency domain HRV measures for each device and condition, obtained similarly.
    devices : list
        A list of device identifiers used to collect the data.
    conditions : list
        A list of conditions under which the data was collected.
    bar_width : float, optional
        The width of the bars in the bar charts. Default is 0.20.
    width : int, optional
        The figure width. Default is 20.
    height : int, optional
        The figure height. Default is 25.

    Notes
    -----
    - The function initially calculates HRV measures for the original (unchopped) dataset to provide a baseline for comparison.
    - Interactive widgets allow users to select which HRV features (from both time and frequency domains) and which condition to display.
    - The comparison is visualized in two separate bar charts: one for time domain features and another for frequency domain features, allowing for an easy assessment of the preprocessing effects.
    - The function is designed to be used in interactive Python environments, such as Jupyter notebooks, where widget functionality can be fully utilized.
    """

    # calculating all the time domain and frequency domain features by the hrvanalysis package for the original dataset
    time_domain_features_original = {
        device: {condition: {} for condition in conditions} for device in devices
    }
    frequency_domain_features_original = {
        device: {condition: {} for condition in conditions} for device in devices
    }

    for device in devices:
        for condition in conditions:
            time_domain_features_original[device][condition] = get_time_domain_features(
                data_chopped[device][condition]
            )
            frequency_domain_features_original[device][condition] = (
                get_frequency_domain_features(data_chopped[device][condition])
            )

    print(
        "All time domain and frequency domain measures succesfully calculated the original data, before pre-processing for comparison"
    )

    time_features = list(time_domain_features[devices[0]][conditions[0]].keys())
    frequency_features = list(
        frequency_domain_features[devices[0]][conditions[0]].keys()
    )

    def plot_bar_charts(time_feature, frequency_feature, condition):
        fig, axs = plt.subplots(2, 1, figsize=(width, height))

        x = np.arange(len(devices))

        time_original = [
            [
                time_domain_features_original[device][condition][time_feature]
                for device in devices
            ]
        ]
        time_processed = [
            [
                time_domain_features[device][condition][time_feature]
                for device in devices
            ]
        ]
        freq_original = [
            [
                frequency_domain_features_original[device][condition][frequency_feature]
                for device in devices
            ]
        ]
        freq_processed = [
            [
                frequency_domain_features[device][condition][frequency_feature]
                for device in devices
            ]
        ]

        bar1 = axs[0].bar(
            x - bar_width / 2,
            time_original[0],
            bar_width,
            color="red",
            label="Original",
        )
        bar2 = axs[0].bar(
            x + bar_width / 2,
            time_processed[0],
            bar_width,
            color="black",
            label="Pre-processed",
        )
        axs[0].bar_label(bar1, padding=3)
        axs[0].bar_label(bar2, padding=3)

        bar3 = axs[1].bar(
            x - bar_width / 2,
            freq_original[0],
            bar_width,
            color="red",
            label="Original",
        )
        bar4 = axs[1].bar(
            x + bar_width / 2,
            freq_processed[0],
            bar_width,
            color="black",
            label="Pre-processed",
        )
        axs[1].bar_label(bar3, padding=3)
        axs[1].bar_label(bar4, padding=3)

        axs[0].set_ylabel(time_feature.upper(), fontsize=25)
        axs[0].set_xlabel("Devices", fontsize=25)
        axs[0].set_title("Time Domain - " + time_feature.upper(), fontsize=25, y=1.02)
        axs[1].set_ylabel(frequency_feature.upper(), fontsize=25)
        axs[1].set_xlabel("Devices", fontsize=25)
        axs[1].set_title(
            "Frequency Domain - " + frequency_feature.upper(), fontsize=25, y=1.02
        )

        axs[0].set_xticks(x)
        axs[0].set_xticklabels(devices, fontsize=15)
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(devices, fontsize=15)

        axs[0].legend()
        axs[1].legend()

        axs[0].grid(linestyle="--", alpha=0.8)
        axs[1].grid(linestyle="--", alpha=0.8)

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
        description="Time Feature:",
        disabled=False,
    )

    frequency_feature_dropdown = widgets.Dropdown(
        options=frequency_features,
        value=frequency_features[0],
        description="Frequency Feature:",
        disabled=False,
    )

    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description="Condition:",
        disabled=False,
    )

    time_feature_dropdown.observe(update_plots, names="value")
    frequency_feature_dropdown.observe(update_plots, names="value")
    condition_dropdown.observe(update_plots, names="value")

    widgets_box = widgets.HBox(
        [time_feature_dropdown, frequency_feature_dropdown, condition_dropdown]
    )

    out = widgets.Output()
    display(widgets_box, out)
    update_plots()


###########################################################################
###########################################################################
###########################################################################


def unfolding_plot(data_pp, devices, conditions, width=8, height=10):
    """
    Creates sequential plots of RMSSD (Root Mean Square of Successive Differences) and heart rate (HR) values over specified intervals for preprocessed data from a selected device and condition.

    This function generates two plots: one for RMSSD and another for HR, plotted over time. The plots display how these values change over specified time intervals, allowing for an examination of variability and trends within a given condition. The user can interactively select the device, condition, and the time window for analysis through widgets.

    Parameters
    ----------
    data_pp : dict
        A dictionary containing preprocessed data for each device and condition. Each entry is expected to be a list of RR intervals in milliseconds.
    devices : list
        A list of strings representing the different devices from which data was collected.
    conditions : list
        A list of strings representing the different conditions under which the data was collected.
    width : int, optional
        The width of the plot in inches. Defaults to 8.
    height : int, optional
        The height of the plot in inches. Defaults to 10.

    Notes
    -----
    - The RMSSD values are calculated as the square root of the mean of the squares of successive differences between adjacent RR intervals.
    - The HR is computed as the mean of 60,000 divided by RR intervals, reflecting the number of beats per minute.
    - The function uses interactive IPython widgets to select the device, condition, and interval for which the data will be plotted. This interactivity requires execution in a Jupyter notebook or similar environment.
    - Time (in seconds) specifies the interval over which RMSSD and HR are averaged and plotted.
    """

    sec_text = IntText(
        value=30,
        description="Sec:",
    )

    condition_dropdown = Dropdown(
        options=conditions,
        value=conditions[0],
        description="Condition:",
    )

    device_dropdown = Dropdown(
        options=devices,
        value=devices[0],
        description="Device:",
    )

    def update_plots(*args):
        sec = sec_text.value
        condition = condition_dropdown.value
        device = device_dropdown.value

        with out:
            clear_output(wait=True)

            def calculate_rmssd(rr_intervals):
                differences = np.diff(
                    rr_intervals
                )  # Calculate the successive differences
                squared_diff = np.square(differences)  # Square the differences
                mean_squared_diff = np.mean(
                    squared_diff
                )  # Calculate the mean of squared differences
                rmssd_value = np.sqrt(
                    mean_squared_diff
                )  # Take the square root to get RMSSD
                return rmssd_value

            def calculate_mean_hr(rr_intervals):
                heart_rate_list = np.divide(
                    60000, rr_intervals
                )  # Calculate heart rate list as 60000 divided by each RR interval
                mean_hr = np.mean(
                    heart_rate_list
                )  # Calculate mean heart rate from the heart rate list
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
            axs[0].plot(
                np.arange(sec, (len(rmssd_values) + 1) * sec, sec),
                rmssd_values,
                "-o",
                color="blue",
            )
            axs[0].set_xlabel("Time (sec)")
            axs[0].set_ylabel("RMSSD")
            axs[0].set_title(
                "RMSSD values in {} condition every {} seconds".format(condition, sec)
            )
            axs[0].grid(True)  # Add grid to the plot

            # Plot HR
            axs[1].plot(
                np.arange(sec, (len(rmssd_values) + 1) * sec, sec),
                heart_rates,
                "-o",
                color="red",
            )
            axs[1].set_xlabel("Time (sec)")
            axs[1].set_ylabel("HR")
            axs[1].set_title(
                "HR values in {} condition every {} seconds".format(condition, sec)
            )
            axs[1].grid(True)  # Add grid to the plot

            plt.subplots_adjust(
                hspace=0.3
            )  # Increase the value to increase the distance
            plt.show()  # Displays the plot

    sec_text.observe(update_plots, names="value")
    condition_dropdown.observe(update_plots, names="value")
    device_dropdown.observe(update_plots, names="value")

    widgets_box = HBox([sec_text, condition_dropdown, device_dropdown])

    out = Output()
    display(widgets_box, out)
    update_plots()


###########################################################################
###########################################################################
###########################################################################


def bar_plot(
    time_domain_features,
    frequency_domain_features,
    devices,
    conditions,
    width=20,
    height=25,
    bar_width=0.20,
):
    """
    Creates bar plots for selected time domain and frequency domain Heart Rate Variability (HRV) features across different devices and conditions.

    This function leverages interactive widgets to allow users to select specific HRV features for comparison. It generates two bar plots: one for a selected time domain feature and another for a frequency domain feature, comparing these across all specified conditions for each device. The plots provide a visual representation of the variability and differences in HRV metrics resulting from different experimental conditions and devices.

    Parameters
    ----------
    time_domain_features : dict
        A nested dictionary containing time domain HRV features for each device and condition. The structure is {device: {condition: {feature: value}}}.
    frequency_domain_features : dict
        A nested dictionary similar to `time_domain_features`, but containing frequency domain HRV features.
    devices : list
        A list of device identifiers used for data collection.
    conditions : list
        A list of experimental conditions under which the data was collected.
    width : int, optional
        The figure width in inches. Defaults to 20.
    height : int, optional
        The figure height in inches. Defaults to 25.
    bar_width : float, optional
        The width of the bars in the plot. Defaults to 0.20.

    Notes
    -----
    - The function initializes with a predefined set of HRV features available for selection, based on the first device and condition in the provided data.
    - Users can select which time and frequency domain features to plot using dropdown menus, enabling dynamic comparison across different conditions and devices.
    - This function is designed for interactive use in Jupyter notebooks or similar environments where IPython widgets are supported.

    Returns
    -------
    None
        The function does not return any value. It displays the plots inline.
    """

    time_features = list(time_domain_features[devices[0]][conditions[0]].keys())
    frequency_features = list(
        frequency_domain_features[devices[0]][conditions[0]].keys()
    )

    def plot_bar_charts(time_feature, frequency_feature):
        fig, axs = plt.subplots(2, 1, figsize=(width, height))

        x = np.arange(len(conditions))

        time = [
            [
                time_domain_features[device][condition][time_feature]
                for condition in conditions
            ]
            for device in devices
        ]
        freq = [
            [
                frequency_domain_features[device][condition][frequency_feature]
                for condition in conditions
            ]
            for device in devices
        ]

        for i, device_time in enumerate(time):
            axs[0].bar(x + i * bar_width, device_time, bar_width, label=devices[i])

        for i, device_freq in enumerate(freq):
            axs[1].bar(x + i * bar_width, device_freq, bar_width, label=devices[i])

        axs[0].set_ylabel(time_feature.upper(), fontsize=25)
        axs[0].set_xlabel("Conditions", fontsize=25)
        axs[0].set_title("Time Domain - " + time_feature.upper(), fontsize=25, y=1)
        axs[1].set_ylabel(frequency_feature.upper(), fontsize=25)
        axs[1].set_xlabel("Conditions", fontsize=25)
        axs[1].set_title(
            "Frequency Domain - " + frequency_feature.upper(), fontsize=25, y=1
        )

        axs[0].set_xticks(x)
        axs[0].set_xticklabels(conditions, fontsize=15)
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(conditions, fontsize=15)

        axs[0].legend()
        axs[1].legend()

        axs[0].grid(linestyle="--", alpha=0.8)
        axs[1].grid(linestyle="--", alpha=0.8)

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
        description="Time Feature:",
        disabled=False,
    )

    frequency_feature_dropdown = widgets.Dropdown(
        options=frequency_features,
        value=frequency_features[0],
        description="Frequency Feature:",
        disabled=False,
    )

    time_feature_dropdown.observe(update_plots, names="value")
    frequency_feature_dropdown.observe(update_plots, names="value")

    widgets_box = widgets.HBox([time_feature_dropdown, frequency_feature_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_plots()


###########################################################################
###########################################################################
###########################################################################


def line_plot(
    time_domain_features,
    frequency_domain_features,
    devices,
    conditions,
    width=20,
    height=25,
):
    """
    Plots line graphs for selected time domain and frequency domain HRV features across different devices and conditions.

    This function generates line plots for HRV features, allowing for the comparison of selected time domain and frequency domain metrics across various conditions and devices. It utilizes interactive widgets for the selection of HRV features to plot, facilitating a dynamic exploration of the data.

    Parameters
    ----------
    time_domain_features : dict
        A dictionary containing the time domain HRV features for each device and condition, structured as {device: {condition: {feature: value}}}.
    frequency_domain_features : dict
        A dictionary similar to `time_domain_features` but containing frequency domain HRV features.
    devices : list
        A list of device identifiers that were used to collect the data.
    conditions : list
        A list of conditions under which the data was collected.
    width : int, optional
        The width of the plot figure in inches. Default is 20.
    height : int, optional
        The height of the plot figure in inches. Default is 25.

    Notes
    -----
    - The function initializes with a set of available HRV features for selection based on the provided data for the first device and condition.
    - Users can dynamically select which HRV feature to plot for both the time domain and frequency domain using dropdown menus.
    - The function is designed to be used in interactive Python environments, such as Jupyter notebooks, where IPython widget functionality can be fully utilized.

    Returns
    -------
    None
        The function does not return any value; it renders the plots inline.
    """

    time_features = list(time_domain_features[devices[0]][conditions[0]].keys())
    frequency_features = list(
        frequency_domain_features[devices[0]][conditions[0]].keys()
    )

    def plot_line_charts(time_feature, frequency_feature):
        fig, axs = plt.subplots(2, 1, figsize=(width, height))

        x = np.arange(len(conditions))

        time = [
            [
                time_domain_features[device][condition][time_feature]
                for condition in conditions
            ]
            for device in devices
        ]
        freq = [
            [
                frequency_domain_features[device][condition][frequency_feature]
                for condition in conditions
            ]
            for device in devices
        ]

        for i, device_time in enumerate(time):
            axs[0].plot(
                x, device_time, marker="o", linestyle="-", linewidth=2, label=devices[i]
            )

        for i, device_freq in enumerate(freq):
            axs[1].plot(
                x, device_freq, marker="o", linestyle="-", linewidth=2, label=devices[i]
            )

        axs[0].set_ylabel(time_feature.upper(), fontsize=25)
        axs[0].set_xlabel("Conditions", fontsize=25)
        axs[0].set_title("Time Domain - " + time_feature.upper(), fontsize=25, y=1)
        axs[1].set_ylabel(frequency_feature.upper(), fontsize=25)
        axs[1].set_xlabel("Conditions", fontsize=25)
        axs[1].set_title(
            "Frequency Domain - " + frequency_feature.upper(), fontsize=25, y=1
        )

        axs[0].set_xticks(x)
        axs[0].set_xticklabels(conditions, fontsize=15)
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(conditions, fontsize=15)

        axs[0].legend()
        axs[1].legend()

        axs[0].grid(linestyle="--", alpha=0.8)
        axs[1].grid(linestyle="--", alpha=0.8)

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
        description="Time Feature:",
        disabled=False,
    )

    frequency_feature_dropdown = widgets.Dropdown(
        options=frequency_features,
        value=frequency_features[0],
        description="Frequency Feature:",
        disabled=False,
    )

    time_feature_dropdown.observe(update_plots, names="value")
    frequency_feature_dropdown.observe(update_plots, names="value")

    widgets_box = widgets.HBox([time_feature_dropdown, frequency_feature_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_plots()


###########################################################################
###########################################################################
###########################################################################


def radar_plot(time_domain_features, criterion, devices, conditions):
    """
    Creates a radar plot comparing time domain HRV features between a criterion device and another selected device for a specific condition using Plotly.

    This function visualizes differences in time domain HRV features (such as RMSSD, PNNI 50, mean HR, and SDNN) between two devices for a given condition. The radar plot facilitates easy comparison to identify similarities or disparities in HRV metrics between the devices. It leverages interactive widgets for selecting the comparison device and condition.

    Parameters
    ----------
    time_domain_features : dict
        A dictionary containing the time domain HRV features for each device and each condition. The structure is {device: {condition: {feature: value}}}.
    criterion : str
        The name of the criterion device for comparison.
    devices : list
        A list including the criterion device and other devices available for comparison.
    conditions : list
        A list of conditions under which the HRV data was collected.

    Notes
    -----
    - The function uses Plotly to create radar (or spider) charts, providing an interactive and visually appealing way to compare HRV features.
    - Users can select the device to compare with the criterion device and the condition for the comparison using dropdown menus.
    - The radar plot includes four key HRV metrics: RMSSD, PNNI 50, mean HR, and SDNN, allowing for a comprehensive comparison of time domain features.

    Returns
    -------
    None
        The function renders the radar plot inline and does not return any value.
    """

    def plot_spider_chart(device, condition):
        rmssd_device = time_domain_features[device][condition]["rmssd"]
        pnni_50_device = time_domain_features[device][condition]["pnni_50"]
        mean_hr_device = time_domain_features[device][condition]["mean_hr"]
        sdnn_device = time_domain_features[device][condition]["sdnn"]

        rmssd_criterion = time_domain_features[criterion][condition]["rmssd"]
        pnni_50_criterion = time_domain_features[criterion][condition]["pnni_50"]
        mean_hr_criterion = time_domain_features[criterion][condition]["mean_hr"]
        sdnn_criterion = time_domain_features[criterion][condition]["sdnn"]

        features = ["rmssd", "pnni_50", "mean_hr", "sdnn"]
        data_criterion = [
            rmssd_criterion,
            pnni_50_criterion,
            mean_hr_criterion,
            sdnn_criterion,
        ]
        data_device = [rmssd_device, pnni_50_device, mean_hr_device, sdnn_device]

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=data_criterion + [data_criterion[0]],
                theta=features + [features[0]],
                fill="toself",
                name=criterion.upper(),
                line_color="blue",
                opacity=0.5,
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=data_device + [data_device[0]],
                theta=features + [features[0]],
                fill="toself",
                name=device.upper(),
                line_color="orange",
                opacity=0.5,
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[
                        min(data_criterion + data_device) * 0.9,
                        max(data_criterion + data_device) * 1.1,
                    ],
                )
            ),
            showlegend=True,
            title="Condition: {} - Devices: {} and {}".format(
                condition.capitalize(), device.capitalize(), criterion.capitalize()
            ),
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
        description="Device:",
        disabled=False,
    )

    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description="Condition:",
        disabled=False,
    )

    device_dropdown.observe(update_plot, names="value")
    condition_dropdown.observe(update_plot, names="value")

    widgets_box = widgets.HBox([device_dropdown, condition_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_plot()


###########################################################################
###########################################################################
###########################################################################


def display_changes(
    time_domain_features, frequency_domain_features, devices, conditions
):
    """
    Displays changes in time and frequency domain features for selected devices and conditions through interactive DataFrames.

    This function provides an interactive analysis of HRV (Heart Rate Variability) features, allowing users to select specific time domain and frequency domain features to examine. It displays formatted DataFrames showing the selected features for given devices and conditions, along with their changes between conditions. The function is designed to facilitate a detailed comparison of HRV metrics across various experimental setups.

    Parameters
    ----------
    time_domain_features : dict
        A dictionary containing the time domain HRV features for each device and condition. The structure is {device: {condition: {feature: value}}}.
    frequency_domain_features : dict
        A dictionary similar to `time_domain_features` but for frequency domain HRV features.
    devices : list
        A list of device identifiers used to collect the HRV data.
    conditions : list
        A list of conditions under which the HRV data was collected.

    Notes
    -----
    - The function uses interactive IPython widgets to allow users to select the HRV feature they wish to analyze. This interactivity requires execution in a Jupyter notebook or similar environment.
    - DataFrames are displayed showing the selected features across all devices and conditions, as well as the changes in these features between conditions. Changes are calculated as the difference between consecutive conditions for each device.
    - This approach provides a visual and numerical representation of how HRV features vary with experimental conditions, aiding in the interpretation and analysis of HRV data.

    Returns
    -------
    None
        The function renders the DataFrames inline and does not return any value.
    """

    time_features = list(time_domain_features[devices[0]][conditions[0]].keys())
    frequency_features = list(
        frequency_domain_features[devices[0]][conditions[0]].keys()
    )

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
        time = [
            [
                time_domain_features[device][condition][time_feature]
                for condition in conditions
            ]
            for device in devices
        ]
        freq = [
            [
                frequency_domain_features[device][condition][frequency_feature]
                for condition in conditions
            ]
            for device in devices
        ]

        time_df = pd.DataFrame(time, columns=conditions, index=devices)
        time_df_changes = time_df.diff(axis=1)

        freq_df = pd.DataFrame(freq, columns=conditions, index=devices)
        freq_df_changes = freq_df.diff(axis=1)

        display_formatted_dataframe(
            time_df, "Time Domain Features - " + time_feature.upper()
        )
        display_formatted_dataframe(
            time_df_changes, "Time Domain Feature Changes - " + time_feature.upper()
        )
        display_formatted_dataframe(
            freq_df, "Frequency Domain Features - " + frequency_feature.upper()
        )
        display_formatted_dataframe(
            freq_df_changes,
            "Frequency Domain Feature Changes - " + frequency_feature.upper(),
        )

    def update_dataframes(*args):
        time_feature = time_feature_dropdown.value
        frequency_feature = frequency_feature_dropdown.value

        with out:
            clear_output(wait=True)
            display_dataframes(time_feature, frequency_feature)

    time_feature_dropdown = widgets.Dropdown(
        options=time_features,
        value=time_features[0],
        description="Time Feature:",
        disabled=False,
    )

    frequency_feature_dropdown = widgets.Dropdown(
        options=frequency_features,
        value=frequency_features[0],
        description="Frequency Feature:",
        disabled=False,
    )

    time_feature_dropdown.observe(update_dataframes, names="value")
    frequency_feature_dropdown.observe(update_dataframes, names="value")

    widgets_box = widgets.HBox([time_feature_dropdown, frequency_feature_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_dataframes()


###########################################################################
###########################################################################
###########################################################################


def save_data(
    pp,
    path,
    time_domain_features,
    frequency_domain_features,
    data_pp,
    devices,
    conditions,
    events,
    artefact=None,
    nibi_before_cropping=None,
    nibi_after_cropping=None,
    save_as_csv=False,
):
    """
    Saves the comprehensive dataset, including time domain features, frequency domain features, and other relevant data, into a CSV file if specified.

    This function compiles data from various sources into a single DataFrame and optionally saves it to a CSV file. The dataset includes time domain and frequency domain HRV features, artefact information, the number of beats before and after cropping, among other details, for each device and condition in the study.

    Parameters
    ----------
    pp : str
        Participant ID used to identify the dataset. Can be a string or a Path object from the pathlib module.
    path : str or Path
        Directory path where the file will be saved.
    time_domain_features : dict
        Dictionary containing time domain HRV features for each device and condition.
    frequency_domain_features : dict
        Dictionary containing frequency domain HRV features for each device and condition.
    data_pp : dict
        Dictionary containing preprocessed HRV data for each device and condition.
    devices : list
        List of devices used to collect HRV data.
    conditions : list
        List of conditions under which HRV data was collected.
    events : dict
        Dictionary containing the events and their corresponding timestamps.
    artefact : dict, optional
        Dictionary containing information about detected artefacts for each device and condition. Defaults to None.
    nibi_before_cropping : dict, optional
        Dictionary containing the number of inter-beat intervals (IBIs) before data cropping for each device and condition. Defaults to None.
    nibi_after_cropping : dict, optional
        Dictionary containing the number of IBIs after data cropping for each device and condition. Defaults to None.
    save_as_csv : bool, optional
        Flag indicating whether to save the compiled data as a CSV file. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing all the processed data, including HRV features, artefact information, and IBIs, for each device and condition.

    Notes
    -----
    - The function combines HRV features, artefact counts, and IBIs into a single DataFrame, facilitating comprehensive analysis.
    - If `save_as_csv` is True, the DataFrame is saved to a CSV file named after the participant ID, allowing for easy data retrieval and further analysis.
    """

    def create_df(
        device,
        time_features,
        freq_features,
        artefact,
        nibi_before_cropping,
        nibi_after_cropping,
        conditions=conditions,
    ):

        df1 = pd.DataFrame(time_features).transpose().reset_index(drop=True)
        df2 = pd.DataFrame(freq_features).transpose().reset_index(drop=True)
        df = df1.assign(**df2)
        df["artefact"] = artefact if artefact is not None else "N/D"
        df["nibi_before_cropping"] = (
            nibi_before_cropping if nibi_before_cropping is not None else "N/D"
        )
        df["nibi_after_cropping"] = (
            nibi_after_cropping if nibi_after_cropping is not None else "N/D"
        )
        df["conditions"] = conditions
        df["device"] = device
        return df

    # extracting the values of number of detected and removed artefact in each device for each condition, and detected beat-to-beat intervals
    if artefact is not None:
        artefact_values = {
            device: list(artefact[device].values()) for device in devices
        }
    else:
        artefact_values = {device: "N/D" for device in devices}

    if nibi_before_cropping is not None:
        nibi_before_cropping_values = {
            device: list(nibi_before_cropping[device].values()) for device in devices
        }
    else:
        nibi_before_cropping_values = {device: "N/D" for device in devices}

    if nibi_after_cropping is not None:
        nibi_after_cropping_values = {
            device: list(nibi_after_cropping[device].values()) for device in devices
        }
    else:
        nibi_after_cropping_values = {device: "N/D" for device in devices}

    # creating a df
    df = {device: {} for device in devices}
    for device in devices:
        df[device] = create_df(
            device,
            time_domain_features[device],
            frequency_domain_features[device],
            artefact_values[device],
            nibi_before_cropping_values[device],
            nibi_after_cropping_values[device],
        )

    # putting all together
    df_all = pd.concat([df[device] for device in devices])

    df_all["pp"] = pp
    df_all["time"] = events["timestamp"][0]  # attaching the starting time point

    if save_as_csv:
        path = Path(path)
        path_save = path / f"{pp}.csv"
        df_all.to_csv(path_save, index=False)
        print("Data Saved Successfully! We are done! :) ")

    return df_all


###########################################################################
###########################################################################
###########################################################################
