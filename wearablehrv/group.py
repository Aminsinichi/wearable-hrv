# Group pipeline for Wearablehrv package

import os
import pandas as pd
import pingouin as pg
import numpy as np
from scipy import stats
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import plotly.express as px
import ipywidgets as widgets
from scipy.stats import linregress
from copy import deepcopy
from IPython.display import display, clear_output
from pathlib import Path

###########################################################################
###########################################################################
###########################################################################


def import_data(path, conditions, devices, features):
    """
    Imports data from multiple CSV files located in a specific directory and structures the data into a nested dictionary based on devices, features, and conditions.

    This function scans a directory for CSV files, reads the data from these files, and organizes it into a nested dictionary. This dictionary is structured to allow easy access to data based on device, feature, and condition. Each file is assumed to contain data for all conditions, devices, and features specified by the parameters.

    Parameters
    ----------
    path : str
        The path to the directory where the CSV files are located. The function expects to find files named in a specific pattern that includes the participant ID.
    conditions : list of str
        A list of strings representing the different experimental conditions under which the data was collected.
    devices : list of str
        A list of strings representing the different devices used to collect the data.
    features : list of str
        A list of strings representing the different features recorded in the data.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - data: A nested dictionary with the structure {device: {feature: {condition: {participant: [values]}}}}, representing the imported data.
        - file_names: A list of the file names that were imported, providing a record of which files were processed.

    Notes
    -----
    - The function automatically identifies CSV files in the provided path and imports data from them. The expected file naming convention and data structure within the files should match the specified devices, conditions, and features.
    """
    path = Path(path)
    # list all files in the directory
    file_names = [f.name for f in path.glob("*.csv") if f.is_file()]
    # sort the resulting list of file names
    file_names.sort()

    participants = (
        {}
    )  # this variable is a dictionary. The keys are each participant's file name, e.g., P01.csv. the values are a big pandas containing the csv file for each participant

    for pp in file_names:
        participants[pp] = pd.read_csv(path / pp)

    data = {
        device: {
            feature: {
                condition: {pp: [] for pp in file_names} for condition in conditions
            }
            for feature in features
        }
        for device in devices
    }

    for pp in file_names:
        for feature in features:
            for condition in conditions:
                for device in devices:
                    value = participants[pp][
                        (participants[pp]["device"] == device)
                        & (participants[pp]["conditions"] == condition)
                    ][feature]
                    data[device][feature][condition][pp].extend(value)

    print("These are the detected .csv files: {}".format(file_names))
    print("Group dataset imported succesfully")

    return data, file_names


###########################################################################
###########################################################################
###########################################################################


def nan_handling(data, devices, features, conditions):
    """
    This function takes a nested dictionary containing the data and replaces [nan] values with an empty list [].

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data, with keys for devices, features, and conditions.
    devices : list
        A list of strings representing the different devices used to collect the data.
    features : list
        A list of strings representing the different features of the data.
    conditions : list
        A list of strings representing the different experimental conditions of the data.

    Returns:
    --------
    new_data : dict
        A nested dictionary with [nan] values replaced with [].
    """
    new_data = deepcopy(data)
    for device in devices:
        for feature in features:
            for condition in conditions:
                values = new_data[device][feature][condition]
                for key, value in values.items():
                    if (
                        isinstance(value, list)
                        and len(value) == 1
                        and pd.isna(value[0])
                    ):
                        new_data[device][feature][condition][key] = []

    print("NaN values are removed from the data")
    return new_data


###########################################################################
###########################################################################
###########################################################################


def save_data(data, path, conditions, devices, features, file_names):
    """
    Saves aggregated data from multiple sources to a CSV file based on specified conditions, devices, and features.

    This function compiles data into a single DataFrame and saves it to a CSV file. The data is organized by combining specified conditions, devices, and features into unique column names.

    Parameters
    ----------
    data : dict
        A nested dictionary with structure {device: {feature: {condition: [values]}}}, representing the data to be saved.
    path : str or Path
        The directory path where the CSV file will be saved. Can be provided as a string or a Path object from the pathlib module.
    conditions : list of str
        Experimental conditions under which the data was collected.
    devices : list of str
        Devices used to collect the data.
    features : list of str
        Features present in the data.
    file_names : list of str
        File names corresponding to the participants' data, used to identify the source files.

    Returns
    -------
    None
        The function does not return any value. It saves the aggregated data into a CSV file named 'group_data.csv' in the specified directory.
    """

    group_data = pd.DataFrame()

    for condition in conditions:
        for feature in features:
            for device in devices:
                column_name = device + "_" + feature + "_" + condition
                column = [
                    value[0] if value else None
                    for value in data[device][feature][condition].values()
                ]
                group_data[column_name] = column

    # Save the datafile:
    path = Path(path)
    path_save = path / "group_data.csv"

    group_data.to_csv(path_save, index=False)
    print("Data Saved Succesfully!")


###########################################################################
###########################################################################
###########################################################################


def signal_quality(
    data,
    path,
    conditions,
    devices,
    features,
    criterion,
    file_names,
    exclude=False,
    save_as_csv=True,
    ibi_threshold=0.20,
    artefact_threshold=0.20,
    manual_missing=False,
    missing_threshold=10,
):
    """
    Analyze the signal quality of the data and generate a quality report. The function processes data from multiple
    devices and conditions, and uses a thresholding method to exclude participants with poor signal quality or manually
    flag data as missing based on the number of detected beats.

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data to be analyzed. Keys represent each device, feature, and condition.

    path : str or Path
        The directory path where the quality report CSV files should be saved. Can be provided as a string or a Path object from the pathlib module.

    conditions : list
        A list of strings indicating the different experimental conditions in the data.

    devices : list
        A list of strings specifying the different devices used for data collection.

    features : list
        A list of strings describing the different data features.

    criterion : str
        The device used as a criterion reference for assessing signal quality.

    file_names : list
        A list of strings containing the participant file names.

    exclude : bool, optional
        A flag indicating whether to exclude participants based on signal quality analysis. Defaults to False.

    save_as_csv : bool, optional
        A flag indicating whether to save the quality report as CSV files. Defaults to True.

    ibi_threshold : float, optional
        A threshold for comparing a device's normalized interbeat interval (nibi) with the criterion nibi. Defaults to 0.20.

    artefact_threshold : float, optional
        A threshold for comparing a device's artefact value with the device's nibi value. Defaults to 0.20.

    manual_missing : bool, optional
        A flag indicating whether to manually flag data as missing based on the number of detected beats. Defaults to False.

    missing_threshold : int, optional
        The threshold for the number of detected beats below which data is flagged as missing when manual_missing is True.
        Defaults to 10.

    Returns:
    --------
    tuple : (dict, list)
        The updated data dictionary after potential participant exclusion based on signal quality analysis, and
        a list of the remaining data features after the removal of 'artefact' and 'nibi_after_cropping'.
    """

    # Create an empty DataFrame to store the results
    quality_df = pd.DataFrame(
        columns=[
            "Device",
            "Condition",
            "Participant",
            "Detected Beats",
            "Criterion Beats",
            "Detected Artefacts",
            "Decision",
        ]
    )

    for (
        device
    ) in devices:  # if you want to excludes the criterion device,  use devices[:-1]
        for condition in conditions:
            for pp in file_names:

                device_nibi = data[device]["nibi_after_cropping"][condition][pp]
                criterion_nibi = data[criterion]["nibi_after_cropping"][condition][pp]
                device_artefact = data[device]["artefact"][condition][pp]

                decision = "Acceptable"

                # check manually for missing data
                if (
                    manual_missing == True
                    and device_nibi
                    and device_nibi[0] < missing_threshold
                ):
                    decision = "Missing"
                elif not device_nibi:
                    decision = "Missing"
                else:
                    device_nibi_val = device_nibi[0]
                    criterion_nibi_val = criterion_nibi[0]
                    if (
                        abs(device_nibi_val - criterion_nibi_val) / criterion_nibi_val
                        > ibi_threshold
                    ):
                        decision = "Poor"

                    if decision != "Poor":
                        device_artefact_val = device_artefact[0]
                        if device_artefact_val / device_nibi_val > artefact_threshold:
                            decision = "Poor"

                # If the decision is to exclude and exclude==True, replace all feature values for that participant and condition with empty lists
                if decision == "Poor" and exclude == True:
                    print("Outliers are excluded from the data!")
                    for feature in features:
                        data[device][feature][condition][pp] = []

                row_df = pd.DataFrame(
                    {
                        "Device": [device],
                        "Condition": [condition],
                        "Participant": [pp],
                        "Detected Beats": [device_nibi_val if device_nibi else None],
                        "Criterion Beats": [
                            criterion_nibi_val if criterion_nibi else None
                        ],
                        "Detected Artefacts": [
                            device_artefact_val if device_artefact else None
                        ],
                        "Decision": [decision],
                    }
                )

                quality_df = pd.concat([quality_df, row_df], ignore_index=True)

    summary_df = (
        quality_df.groupby(["Device", "Condition", "Decision"])
        .size()
        .unstack(fill_value=0)
    )
    summary_df["Total"] = summary_df["Acceptable"] + summary_df["Poor"]

    # Remove 'artefact' and 'nibi_after_cropping' from the data variable
    for device in devices:
        del data[device]["artefact"]
        del data[device]["nibi_after_cropping"]

    # Remove 'artefact' and 'nibi_after_cropping' from the features
    features.remove("artefact")
    features.remove("nibi_after_cropping")

    print("Signal quality is assessed successfully!")

    # Save the datafiles:
    if save_as_csv == True:
        path = Path(path)
        path_save_quality = path / "quality_report1.csv"
        path_save_summary = path / "quality_report2.csv"

        quality_df.to_csv(path_save_quality, index=False)
        summary_df.to_csv(path_save_summary)
        print("Data Saved Succesfully!")

    return data, features, summary_df, quality_df


###########################################################################
###########################################################################
###########################################################################


def signal_quality_plot1(
    summary_df,
    condition_mapping,
    criterion,
    device_selection=False,
    device=None,
    criterion_exclusion=True,
    x_label="'Condition Categories",
):
    """
    Generates a stacked bar plot to visualize data quality by condition category. The function can either
    display data for all devices or focus on a single specified device, and it has the option to exclude data
    from the criterion device. The data quality metrics ('Acceptable', 'Missing', 'Poor') are aggregated from
    the summary DataFrame for each condition category and displayed as percentages.

    Parameters:
    -----------
    summary_df : DataFrame
        A pandas DataFrame containing the summary of data quality metrics. It should have a MultiIndex with
        levels 'Device' and 'Condition', and columns 'Acceptable', 'Missing', and 'Poor'.

    condition_mapping : dict
        A dictionary mapping each condition to its respective category. Keys should correspond to conditions
        in summary_df's index, and values should be the desired category names.

    criterion : str
        The name of the criterion device. Used to exclude data from this device if criterion_exclusion is True.

    device_selection : bool, optional
        If True, the plot will only display data for the device specified in the 'device' parameter. Defaults to False.

    device : str, optional
        The name of the device to focus on if device_selection is True. Ignored if device_selection is False.

    criterion_exclusion : bool, optional
        If True, data from the criterion device will be excluded from the plot. Defaults to True.

    x_label : str, optional
        The label for the x-axis of the plot. Defaults to "'Condition Categories'".

    Returns:
    --------
    None
        This function does not return anything. It displays the generated plot directly.

    Notes:
    ------
    The function first checks for the criterion device in the 'Device' level of the summary_df's MultiIndex. If
    criterion_exclusion is True, data from this device is excluded. If device_selection is True, the function
    further filters the summary_df to only include data from the specified device. It then maps each condition
    to its respective category using the provided condition_mapping and groups the data by these categories. The
    quality metrics are then converted to percentages and plotted as a stacked bar chart. The function also adds
    text annotations to the plot to display the percentage values and customizes the plot with labels, title, and
    legend.
    """

    if criterion_exclusion:
        summary_df = summary_df[
            summary_df.index.get_level_values("Device") != criterion
        ].copy()

    if device_selection:
        summary_df = summary_df[
            summary_df.index.get_level_values("Device") == device
        ].copy()

    # Apply the mapping to create a new column for condition categories
    summary_df["Condition Category"] = summary_df.index.get_level_values(
        "Condition"
    ).map(condition_mapping)

    # Grouping by new condition category
    grouped_by_condition = summary_df.groupby("Condition Category")[
        ["Acceptable", "Missing", "Poor"]
    ].sum()

    # Convert counts to percentages
    grouped_by_condition_percent = (
        grouped_by_condition.div(grouped_by_condition.sum(axis=1), axis=0) * 100
    )

    # Plotting
    ax = grouped_by_condition_percent.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        color={"Poor": "red", "Acceptable": "green", "Missing": "yellow"},
    )
    if device_selection:
        plt.title("Data Quality for {} Device (Percentage)".format(device))
    else:
        plt.title("Data Quality for All Devices (Percentage)")
    plt.xlabel(x_label)
    plt.ylabel("Percentage")
    plt.legend(title="Data Quality")

    # Adding text annotations
    for i, row in enumerate(grouped_by_condition_percent.itertuples()):
        cumulative_height = 0
        for value in row[1:]:
            if value != 0:
                ax.text(
                    x=i,
                    y=cumulative_height + value / 2,
                    s=f"{value:.2f}%",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
                cumulative_height += value

    # Moving the legend
    plt.legend(title="Data Quality", loc="center left", bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.show()


###########################################################################
###########################################################################
###########################################################################


def signal_quality_plot2(summary_df, condition_selection=False, condition=None):
    """
    Generates a stacked bar plot to visualize data quality by device. This function allows the option to either
    display data for all conditions or focus on a single specified condition. It aggregates data quality metrics
    ('Acceptable', 'Missing', 'Poor') for each device and displays them as percentages.

    Parameters:
    -----------
    summary_df : DataFrame
        A pandas DataFrame containing the summary of data quality metrics. It should have a MultiIndex with
        levels 'Device' and 'Condition', and columns 'Acceptable', 'Missing', and 'Poor'.

    conditions : list
        A list of strings representing all the conditions included in the summary_df.

    devices : list
        A list of strings representing all the devices included in the summary_df.

    condition_selection : bool, optional
        If True, the plot will only display data for the condition specified in the 'condition' parameter.
        Defaults to False.

    condition : str, optional
        The name of the condition to focus on if condition_selection is True. Ignored if condition_selection
        is False.

    Returns:
    --------
    None
        This function does not return anything. It displays the generated plot directly.

    Notes:
    ------
    The function checks if condition_selection is True. If so, it filters the summary_df to only include
    data for the specified condition. It then groups the data by 'Device' and calculates the sum of
    'Acceptable', 'Missing', and 'Poor' metrics for each device. These counts are converted to percentages
    and plotted as a stacked bar chart. The function also adds text annotations to the plot for clarity and
    customizes the plot with a title, labels, and a legend. The legend is moved outside of the plot to avoid
    obscuring any data.
    """

    if condition_selection:
        summary_df = summary_df[
            summary_df.index.get_level_values("Condition") == condition
        ].copy()

    grouped_data = summary_df.groupby("Device")[["Acceptable", "Missing", "Poor"]].sum()

    # Convert counts to percentages
    grouped_data_percent = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100

    # Plotting the stacked bar chart with custom colors
    ax = grouped_data_percent.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
        color={"Poor": "red", "Acceptable": "green", "Missing": "yellow"},
    )

    if condition_selection:
        plt.title(
            "Data Quality by Device for {} Condition (Percentage)".format(condition)
        )
    else:
        plt.title("Data Quality by Device for All Conditions (Percentage)")

    plt.xlabel("Device")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    plt.legend(title="Data Quality")

    # Adding text annotations with corrected x-coordinate
    for i, row in enumerate(grouped_data_percent.itertuples()):
        cumulative_height = 0
        for value in row[1:]:
            if value != 0:  # This check avoids placing a label for a zero value
                ax.text(
                    x=i,
                    y=cumulative_height + value / 2,
                    s=f"{value:.2f}%",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
                cumulative_height += value

    # Moving the legend outside of the plot to the right
    plt.legend(title="Data Quality", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


###########################################################################
###########################################################################
###########################################################################


def violin_plot(data, conditions, features, devices):
    """
    This function creates an interactive violin plot comparing different devices for the given data, conditions, features, and devices. It uses Plotly and Jupyter widgets for visualization and user interaction.

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data, with keys for devices, features, and conditions.
    conditions : list
        A list of strings representing the different experimental conditions of the data.
    features : list
        A list of strings representing the different features of the data.
    devices : list
        A list of strings representing the different devices used to collect the data.

    Returns:
    --------
    None
        The function directly creates and displays the interactive violin plot.
    """

    def create_violin_plot(feature, condition):
        traces = []
        for d, device in enumerate(devices):
            y_values = [
                value[0] for value in data[device][feature][condition].values() if value
            ]

            trace = go.Violin(
                y=y_values,
                name=device,
                box_visible=True,
                line_color=px.colors.qualitative.Plotly[
                    d % len(px.colors.qualitative.Plotly)
                ],
            )
            traces.append(trace)

        layout = go.Layout(
            title=feature.upper() + " - Comparison",
            yaxis=dict(title=feature.upper()),
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    def update_violin_plot(*args):
        feature = feature_dropdown.value
        condition = condition_dropdown.value

        with out:
            clear_output(wait=True)
            create_violin_plot(feature, condition)

    feature_dropdown = widgets.Dropdown(
        options=features,
        value=features[0],
        description="Feature:",
        disabled=False,
    )

    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description="Condition:",
        disabled=False,
    )

    feature_dropdown.observe(update_violin_plot, names="value")
    condition_dropdown.observe(update_violin_plot, names="value")

    widgets_box = widgets.HBox([feature_dropdown, condition_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_violin_plot()


###########################################################################
###########################################################################
###########################################################################


def box_plot(data, conditions, features, devices):
    """
    This function creates an interactive box plot comparing different devices for the given data, conditions, features, and devices. It uses Plotly and Jupyter widgets for visualization and user interaction.

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data, with keys for devices, features, and conditions.
    conditions : list
        A list of strings representing the different experimental conditions of the data.
    features : list
        A list of strings representing the different features of the data.
    devices : list
        A list of strings representing the different devices used to collect the data.

    Returns:
    --------
    None
        The function directly creates and displays the interactive box plot.
    """

    def create_box_plot(feature, condition):
        # Create traces
        traces = []
        for d, device in enumerate(devices):
            # Extract non-empty values for the current device, feature, and condition
            y_values = [
                value[0] for value in data[device][feature][condition].values() if value
            ]

            trace = go.Box(y=y_values, name=device)
            traces.append(trace)

        # Create layout
        layout = go.Layout(
            title=feature.upper() + " - Comparison",
            yaxis=dict(title=feature.upper()),
            boxmode="group",
        )

        # Create figure
        fig = go.Figure(data=traces, layout=layout)

        # Display figure
        fig.show()

    def update_box_plot(*args):
        feature = feature_dropdown.value
        condition = condition_dropdown.value

        with out:
            clear_output(wait=True)
            create_box_plot(feature, condition)

    feature_dropdown = widgets.Dropdown(
        options=features,
        value=features[0],
        description="Feature:",
        disabled=False,
    )

    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description="Condition:",
        disabled=False,
    )

    feature_dropdown.observe(update_box_plot, names="value")
    condition_dropdown.observe(update_box_plot, names="value")

    widgets_box = widgets.HBox([feature_dropdown, condition_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_box_plot()


###########################################################################
###########################################################################
###########################################################################


def radar_plot(data, criterion, conditions, devices):
    """
    This function creates an interactive radar plot comparing a selected device to a criterion device for the given data, conditions, features, and devices. It uses Plotly and Jupyter widgets for visualization and user interaction.

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data, with keys for devices, features, and conditions.
    criterion : str
        The criterion device to be compared with the selected device.
    conditions : list
        A list of strings representing the different experimental conditions of the data.
    devices : list
        A list of strings representing the different devices used to collect the data.

    Returns:
    --------
    None
        The function directly creates and displays the interactive radar plot.
    """

    def create_radar_chart(device, condition):
        rmssd_device = np.nanmean(
            [value[0] for value in data[device]["rmssd"][condition].values() if value]
        )
        pnni_50_device = np.nanmean(
            [value[0] for value in data[device]["pnni_50"][condition].values() if value]
        )
        mean_hr_device = np.nanmean(
            [value[0] for value in data[device]["mean_hr"][condition].values() if value]
        )
        sdnn_device = np.nanmean(
            [value[0] for value in data[device]["sdnn"][condition].values() if value]
        )

        rmssd_criterion = np.nanmean(
            [
                value[0]
                for value in data[criterion]["rmssd"][condition].values()
                if value
            ]
        )
        pnni_50_criterion = np.nanmean(
            [
                value[0]
                for value in data[criterion]["pnni_50"][condition].values()
                if value
            ]
        )
        mean_hr_criterion = np.nanmean(
            [
                value[0]
                for value in data[criterion]["mean_hr"][condition].values()
                if value
            ]
        )
        sdnn_criterion = np.nanmean(
            [value[0] for value in data[criterion]["sdnn"][condition].values() if value]
        )

        features_for_plotting = ["rmssd", "pnni_50", "mean_hr", "sdnn"]
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
                theta=features_for_plotting + [features_for_plotting[0]],
                fill="toself",
                name=criterion.upper(),
                line_color="blue",
                opacity=0.5,
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=data_device + [data_device[0]],
                theta=features_for_plotting + [features_for_plotting[0]],
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

    def update_radar_chart(*args):
        device = device_dropdown.value
        condition = condition_dropdown.value

        with out:
            clear_output(wait=True)
            create_radar_chart(device, condition)

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

    device_dropdown.observe(update_radar_chart, names="value")
    condition_dropdown.observe(update_radar_chart, names="value")

    widgets_box = widgets.HBox([device_dropdown, condition_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_radar_chart()


###########################################################################
###########################################################################
###########################################################################


def hist_plot(data, conditions, features, devices):
    """
    This function creates an interactive histogram plot comparing different devices for the given data, conditions, features, and devices. It uses Plotly and Jupyter widgets for visualization and user interaction.

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data, with keys for devices, features, and conditions.
    conditions : list
        A list of strings representing the different experimental conditions of the data.
    features : list
        A list of strings representing the different features of the data.
    devices : list
        A list of strings representing the different devices used to collect the data.

    Returns:
    --------
    None
        The function directly creates and displays the interactive histogram plot.
    """

    def create_histogram(feature, condition):
        traces = []
        for d, device in enumerate(devices):
            y_values = [
                value[0] for value in data[device][feature][condition].values() if value
            ]

            trace = go.Histogram(x=y_values, name=device, opacity=0.75)
            traces.append(trace)

        layout = go.Layout(
            title=feature.upper() + " - Comparison",
            xaxis=dict(title=feature.upper()),
            yaxis=dict(title="Count"),
            barmode="overlay",
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    def update_histogram(*args):
        feature = feature_dropdown.value
        condition = condition_dropdown.value

        with out:
            clear_output(wait=True)
            create_histogram(feature, condition)

    feature_dropdown = widgets.Dropdown(
        options=features,
        value=features[0],
        description="Feature:",
        disabled=False,
    )

    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description="Condition:",
        disabled=False,
    )

    feature_dropdown.observe(update_histogram, names="value")
    condition_dropdown.observe(update_histogram, names="value")

    widgets_box = widgets.HBox([feature_dropdown, condition_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_histogram()


###########################################################################
###########################################################################
###########################################################################


def matrix_plot(data, file_names, conditions, features, devices):
    """
    This function creates an interactive heatmap plot comparing different devices for the given data, conditions, features, and devices. It uses Seaborn, Matplotlib, and Jupyter widgets for visualization and user interaction.

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data, with keys for devices, features, and conditions.
    file_names : list
        A list of strings containing the file names of the CSV files to be imported.
    conditions : list
        A list of strings representing the different experimental conditions of the data.
    features : list
        A list of strings representing the different features of the data.
    devices : list
        A list of strings representing the different devices used to collect the data.

    Returns:
    --------
    None
        The function directly creates and displays the interactive heatmap plot.
    """

    def create_heatmap(feature, condition):
        data_to_plot = {device: [] for device in devices}

        for device in devices:
            for pp in file_names:
                value = data[device][feature][condition][pp]
                if value:  # If value is not empty
                    data_to_plot[device].append(value[0])
                else:
                    data_to_plot[device].append(None)

        # Heatmap for comparing all devices
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            pd.DataFrame(data_to_plot),
            cmap="YlGnBu",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": feature.upper()},
        )
        plt.xlabel("Device")
        plt.ylabel("Participant")
        plt.title("{} in {} condition".format(feature.upper(), condition))

        # Change y-axis tick labels to start from 1
        y_tick_labels = list(range(1, len(file_names) + 1))
        ax.set_yticklabels(y_tick_labels)

        plt.show()

    def update_heatmap(*args):
        feature = feature_dropdown.value
        condition = condition_dropdown.value

        with out:
            clear_output(wait=True)
            create_heatmap(feature, condition)

    feature_dropdown = widgets.Dropdown(
        options=features,
        value=features[0],
        description="Feature:",
        disabled=False,
    )

    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description="Condition:",
        disabled=False,
    )

    feature_dropdown.observe(update_heatmap, names="value")
    condition_dropdown.observe(update_heatmap, names="value")

    widgets_box = widgets.HBox([feature_dropdown, condition_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_heatmap()


###########################################################################
###########################################################################
###########################################################################


def density_plot(data, file_names, conditions, features, devices):
    """
    This function creates an interactive density plot comparing different devices for the given data, conditions, features, and devices. It uses Seaborn, Matplotlib, and Jupyter widgets for visualization and user interaction.

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data, with keys for devices, features, and conditions.
    file_names : list
        A list of strings containing the file names of the CSV files to be imported.
    conditions : list
        A list of strings representing the different experimental conditions of the data.
    features : list
        A list of strings representing the different features of the data.
    devices : list
        A list of strings representing the different devices used to collect the data.

    Returns:
    --------
    None
        The function directly creates and displays the interactive density plot.
    """

    def create_density_plot(feature, condition):
        # Density plot for comparing devices
        plt.figure(figsize=(8, 6))
        for d, device in enumerate(devices):
            device_data = [
                value
                for pp in file_names
                for value in data[device][feature][condition][pp]
                if value
            ]
            sns.kdeplot(device_data, label=device, shade=True)

        plt.xlabel(feature)
        plt.title("Density Plot: {} in {} condition".format(feature.upper(), condition))
        plt.legend()
        plt.show()

    def update_density_plot(*args):
        feature = feature_dropdown.value
        condition = condition_dropdown.value

        with out:
            clear_output(wait=True)
            create_density_plot(feature, condition)

    feature_dropdown = widgets.Dropdown(
        options=features,
        value=features[0],
        description="Feature:",
        disabled=False,
    )

    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description="Condition:",
        disabled=False,
    )

    feature_dropdown.observe(update_density_plot, names="value")
    condition_dropdown.observe(update_density_plot, names="value")

    widgets_box = widgets.HBox([feature_dropdown, condition_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_density_plot()


###########################################################################
###########################################################################
###########################################################################


def bar_plot(data, conditions, features, devices):
    """
    This function creates an interactive bar chart comparing the mean values of different devices for the given data, conditions, features, and devices. It uses Plotly and Jupyter widgets for visualization and user interaction.

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data, with keys for devices, features, and conditions.
    conditions : list
        A list of strings representing the different experimental conditions of the data.
    features : list
        A list of strings representing the different features of the data.
    devices : list
        A list of strings representing the different devices used to collect the data.

    Returns:
    --------
    None
        The function directly creates and displays the interactive bar chart.
    """

    def create_bar_chart(feature, condition):
        mean_values = []
        sem_values = []
        for d, device in enumerate(devices):
            y_values = [
                value[0] for value in data[device][feature][condition].values() if value
            ]
            mean_values.append(np.mean(y_values))
            sem_values.append(np.std(y_values, ddof=1) / np.sqrt(len(y_values)))

        traces = []
        for d, device in enumerate(devices):
            trace = go.Bar(
                x=[device],
                y=[mean_values[d]],
                error_y=dict(type="data", array=[sem_values[d]]),
                name=device,
            )
            traces.append(trace)

        layout = go.Layout(
            title=feature.upper() + " - Comparison",
            yaxis=dict(title=feature.upper()),
            xaxis=dict(title="Devices"),
            barmode="group",
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    def update_bar_chart(*args):
        feature = feature_dropdown.value
        condition = condition_dropdown.value

        with out:
            clear_output(wait=True)
            create_bar_chart(feature, condition)

    feature_dropdown = widgets.Dropdown(
        options=features,
        value=features[0],
        description="Feature:",
        disabled=False,
    )

    condition_dropdown = widgets.Dropdown(
        options=conditions,
        value=conditions[0],
        description="Condition:",
        disabled=False,
    )

    feature_dropdown.observe(update_bar_chart, names="value")
    condition_dropdown.observe(update_bar_chart, names="value")

    widgets_box = widgets.HBox([feature_dropdown, condition_dropdown])

    out = widgets.Output()
    display(widgets_box, out)
    update_bar_chart()


###########################################################################
###########################################################################
###########################################################################


def mape_analysis(
    data,
    criterion,
    devices,
    conditions,
    features,
    path=None,
    alpha=0.95,
    save_as_csv=False,
):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) and its confidence intervals for each device compared to the criterion.
    Optionally saves the results to a CSV file.

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data for each device, feature, and condition.
    criterion : str
        A string representing the name of the criterion device.
    devices : list
        A list of strings representing the different devices.
    conditions : list
        A list of strings representing the different conditions.
    features : list
        A list of strings representing the different features.
    path : str or Path
        The path where the CSV file will be saved. Can be provided as a string or a Path object from the pathlib module. If not specified, the results are not saved to a file. Default is None.
    alpha : float
        Confidence level for the intervals.
    save_as_csv : bool
        Flag to save the results to a CSV file.

    Returns:
    --------
    mape_data : dict
        A nested dictionary containing the MAPE and its confidence intervals.
    """
    mape_data = {
        device: {
            feature: {condition: {} for condition in conditions} for feature in features
        }
        for device in devices[:-1]
    }

    for device in devices[:-1]:
        for feature in features:
            for condition in conditions:
                device_data = data[device][feature][condition]
                criterion_data = data[criterion][feature][condition]

                abs_percent_errors = []
                for pp, (pp_value_device, pp_value_criterion) in enumerate(
                    zip(device_data.values(), criterion_data.values())
                ):
                    if (
                        pp_value_device
                        and pp_value_criterion
                        and pp_value_criterion[0] != 0
                    ):
                        error = abs(pp_value_device[0] - pp_value_criterion[0]) / abs(
                            pp_value_criterion[0]
                        )
                        abs_percent_errors.append(error * 100)  # Convert to percentage

                if abs_percent_errors:
                    mape = np.mean(abs_percent_errors)
                    ci_lower, ci_upper = stats.t.interval(
                        alpha,
                        len(abs_percent_errors) - 1,
                        loc=mape,
                        scale=stats.sem(abs_percent_errors),
                    )

                    mape_data[device][feature][condition] = {
                        "MAPE": mape,
                        "CI": (ci_lower, ci_upper),
                    }
                else:
                    mape_data[device][feature][condition] = {
                        "MAPE": None,
                        "CI": (None, None),
                    }

    print("MAPE is calculated successfully!")

    if save_as_csv and path:
        rows = []
        for device in mape_data:
            for feature in mape_data[device]:
                for condition in mape_data[device][feature]:
                    row_data = mape_data[device][feature][condition]
                    rows.append(
                        [
                            device,
                            feature,
                            condition,
                            row_data["MAPE"],
                            row_data["CI"][0],
                            row_data["CI"][1],
                        ]
                    )

        df = pd.DataFrame(
            rows,
            columns=["Device", "Feature", "Condition", "MAPE", "CI Lower", "CI Upper"],
        )
        path = Path(path)
        df.to_csv(path / "mape_data.csv", index=False)
        print("MAPE data saved successfully to", path + "mape_data.csv")

    return mape_data


###########################################################################
###########################################################################
###########################################################################


def mape_plot(mape_data, features, conditions, devices):
    """
    Plots a grouped bar chart of Mean Absolute Percentage Error (MAPE) for different devices across all conditions for a selected feature.

    This function generates an interactive grouped bar chart that allows the user to select a feature from a dropdown menu. The chart displays the MAPE values for each device across all conditions.

    Parameters
    ----------
    mape_data : dict
        A dictionary containing the MAPE results, structured as {device: {feature: {condition: {"MAPE": value}}}}. This structure allows for indexing by device, feature, and condition to retrieve the MAPE values.
    features : list of str
        A list of strings representing the features for which MAPE values are available. The user can select from these features to generate the plot.
    conditions : list of str
        A list of strings representing the different experimental conditions. These conditions form the x-axis of the bar chart.
    devices : list of str
        A list of strings representing the different devices. MAPE values for each device are plotted as groups of bars within each condition.

    Notes
    -----
    - The bar chart includes a legend indicating the devices, and the MAPE values are expressed as percentages.
    """

    def create_mape_plot(feature):
        num_conditions = len(conditions)
        num_devices = len(devices) - 1

        bar_width = 0.15
        opacity = 0.8

        fig, ax = plt.subplots(figsize=(12, 6))
        index = np.arange(num_conditions)

        for i, device in enumerate(devices[:-1]):
            mape_values = [
                mape_data[device][feature][condition]["MAPE"]
                for condition in conditions
            ]
            bars = ax.bar(
                index + bar_width * i,
                mape_values,
                bar_width,
                alpha=opacity,
                label=device,
            )

        ax.set_xlabel("Condition")
        ax.set_ylabel("MAPE (%)")
        ax.set_title(f"MAPE for {feature.capitalize()} across Conditions")
        ax.set_xticks(index + bar_width * num_devices / 2)
        ax.set_xticklabels(conditions)
        ax.legend()

        ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.5)

        plt.tight_layout()
        plt.show()

    def update_mape_plot(*args):
        feature = feature_dropdown.value

        with out:
            clear_output(wait=True)
            create_mape_plot(feature)

    feature_dropdown = widgets.Dropdown(
        options=features,
        value=features[0],
        description="Feature:",
        disabled=False,
    )

    feature_dropdown.observe(update_mape_plot, names="value")

    out = widgets.Output()
    display(widgets.VBox([feature_dropdown, out]))
    update_mape_plot()


###########################################################################
###########################################################################
###########################################################################


def check_normality(data, conditions, devices, features, alpha=0.05):
    """
    Analyzes the normality of data using the Shapiro-Wilk test for selected features across devices and conditions.

    Parameters
    ----------
    data : dict
        Nested dictionary with structure {device: {feature: {condition: [values]}}}.
    conditions : list of str
        Experimental conditions.
    devices : list of str
        Devices used for data collection.
    features : list of str
        Features of the data to analyze.
    alpha : float, optional
        Significance level for the Shapiro-Wilk test, default is 0.05.

    Returns
    -------
    normality_results : dict
        Results of the Shapiro-Wilk test, structured as {device: {feature: {condition: {'p_value': float, 'is_normal': bool}}}}.
    suggestions : dict
        Suggestions for transformation if data is not normally distributed, with the same structure as `normality_results`.
    transformation_summary : dict
        Summary of conditions requiring transformation for each device and feature, formatted as {device: {feature: [conditions]}}.

    This function checks each combination of device, feature, and condition for normality. It suggests transformations for non-normal distributions and summarizes which conditions require transformation.
    """

    normality_results = {
        device: {
            feature: {condition: {} for condition in conditions} for feature in features
        }
        for device in devices
    }
    suggestions = {
        device: {
            feature: {condition: "" for condition in conditions} for feature in features
        }
        for device in devices
    }
    transformation_summary = {
        device: {feature: [] for feature in features} for device in devices
    }

    for device in devices:
        for feature in features:
            for condition in conditions:
                data_values = [
                    val[0] for val in data[device][feature][condition].values() if val
                ]

                if data_values:
                    stat, p = stats.shapiro(data_values)
                    normality_result = {"p_value": p, "is_normal": p > alpha}
                    normality_results[device][feature][condition] = normality_result

                    # Update suggestions
                    if p <= alpha:
                        suggestion = (
                            "Consider Box-Cox transformation."
                            if not np.all(np.array(data_values) >= 0)
                            else "Consider log or square root transformation."
                        )
                        suggestions[device][feature][condition] = suggestion

                    # Update transformation summary for features
                    if feature in features and not normality_result["is_normal"]:
                        transformation_summary[device][feature].append(condition)
                else:
                    normality_results[device][feature][condition] = {
                        "p_value": None,
                        "is_normal": None,
                    }
                    suggestions[device][feature][
                        condition
                    ] = "Data unavailable for normality test."

    print("Normality check is done successfully!")
    return normality_results, suggestions, transformation_summary


###########################################################################
###########################################################################
###########################################################################


def log_transform_data(data, devices, transform_features):
    """
    Applies a log transformation to the 'rmssd' and 'hf' features in the data for all devices and conditions.

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data, with keys for devices, features, and conditions.
    devices : list
        A list of strings representing the different devices used to collect the data.
    transform_features : list
        A list of strings representing the specific features to transform ('rmssd', 'hf').

    Returns:
    --------
    None - the function updates the 'data' dictionary in place.
    """
    for device in devices:
        for feature in transform_features:
            for condition in data[device][feature]:
                original_values = data[device][feature][condition]

                # Apply log transformation to each participant's data, ensuring values are positive
                for participant in original_values:
                    participant_data = original_values[participant]
                    transformed_data = [
                        np.log(value) for value in participant_data if value > 0
                    ]

                    # Update the data dictionary with the transformed data
                    data[device][feature][condition][participant] = transformed_data

    print("Selected features are log-transformed successfully!")


###########################################################################
###########################################################################
###########################################################################


def regression_analysis(
    data, criterion, conditions, devices, features, path, save_as_csv=False
):
    """
    Performs linear regression analysis on the given data against a criterion device for specified conditions, devices, and features. It calculates the regression parameters (slope, intercept, R-value, P-value, and standard error) for each device-feature-condition combination compared to the criterion. Optionally, saves the regression results to a CSV file.

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data, organized as {device: {feature: {condition: values}}}. The keys represent devices, features, and conditions, respectively, with values being the data points collected under those criteria.

    criterion : str
        The name of the device used as a benchmark or standard for comparison in the regression analysis.

    conditions : list of str
        The experimental conditions under which the data was collected. Each condition corresponds to a specific set of circumstances or parameters under which the data points were gathered.

    devices : list of str
        The devices from which data was collected. This list should include the criterion device and any other devices whose data is to be analyzed against the criterion.

    features : list of str
        The features or variables of interest within the data. These could represent different metrics or measurements collected from the devices.

    path : str or Path, optional
        The directory path where the output CSV file containing the regression analysis results will be saved. Can be provided as a string or a Path object from the pathlib module. If not specified, the results are not saved to a file.

    save_as_csv : bool, optional
        A flag indicating whether to save the regression results to a CSV file. If True, the results are saved to the specified path. If False, the results are not saved to a file. Default is False.

    alpha : float, optional
        The confidence level for the confidence intervals in the regression analysis. Default value is set to 0.95, representing a 95% confidence level.

    Returns:
    --------
    regression_data : dict
        A nested dictionary containing the regression analysis results for each device-feature-condition combination. The structure is {device: {feature: {condition: {'slope': value, 'intercept': value, 'r_value': value, 'p_value': value, 'std_err': value}}}}, where each parameter represents the slope, intercept, correlation coefficient, P-value, and standard error of the regression, respectively.

    Notes:
    -----
    - This function assumes that the data for each device-feature-condition combination is directly comparable and that linear regression is an appropriate analysis method.
    - The regression analysis is performed for each device against the criterion for every feature and condition specified.
    - The function outputs detailed regression results, facilitating the assessment of the linear relationship between the data from each device and the criterion across different conditions and features.
    """

    # Defining empty dictionaries to save regression data analysis
    regression_data = {
        device: {
            feature: {condition: {} for condition in conditions} for feature in features
        }
        for device in devices[:-1]
    }

    for device in devices[:-1]:
        for feature in features:
            for condition in conditions:
                # Check for matching values between the criterion and device
                device_data = data[device][feature][condition]
                criterion_data = data[criterion][feature][condition]

                filtered_data_device = []
                filtered_data_criterion = []

                for pp, (pp_value_device, pp_value_criterion) in enumerate(
                    zip(device_data.values(), criterion_data.values())
                ):
                    if pp_value_device and pp_value_criterion:
                        filtered_data_device.append(pp_value_device[0])
                        filtered_data_criterion.append(pp_value_criterion[0])

                # Calculate regression if there are matching values
                if filtered_data_device and filtered_data_criterion:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        filtered_data_device, filtered_data_criterion
                    )
                    regression = {
                        "slope": slope,
                        "intercept": intercept,
                        "r_value": r_value,
                        "p_value": p_value,
                        "std_err": std_err,
                    }
                    regression_data[device][feature][condition] = regression
                else:
                    regression_data[device][feature][condition] = None

    print("Done Successfully!")

    if save_as_csv:
        # Convert nested dictionary to a list of rows
        rows = []
        for device in regression_data:
            for feature in regression_data[device]:
                for condition in regression_data[device][feature]:
                    row_data = regression_data[device][feature][condition]
                    if row_data:
                        rows.append(
                            [
                                device,
                                feature,
                                condition,
                                row_data["slope"],
                                row_data["intercept"],
                                row_data["r_value"],
                                row_data["p_value"],
                                row_data["std_err"],
                            ]
                        )
                    else:
                        rows.append(
                            [device, feature, condition, None, None, None, None, None]
                        )

        # Convert list of rows to DataFrame
        df = pd.DataFrame(
            rows,
            columns=[
                "Device",
                "Feature",
                "Condition",
                "Slope",
                "Intercept",
                "R Value",
                "P Value",
                "Std Err",
            ],
        )

        # Save DataFrame to CSV
        path = Path(path)
        df.to_csv(path / "regression_data.csv", index=False)
        print("CSV File is saved successfully")

    return regression_data


###########################################################################
###########################################################################
###########################################################################


def bonferroni_correction_regression(regression_data, alpha=0.05):
    """
    Applies the Bonferroni correction to the p-values in the regression analysis results.

    Parameters:
    -----------
    regression_data : dict
        A nested dictionary containing the regression results for each device, feature, and condition.
    alpha : float, optional, default: 0.05
        The significance level for the tests.

    Returns:
    --------
    corrected_regression_data : dict
        The regression data with Bonferroni corrected p-values.
    """
    corrected_regression_data = regression_data.copy()
    total_tests = sum(
        len(regression_data[device][feature][condition])
        for device in regression_data
        for feature in regression_data[device]
        for condition in regression_data[device][feature]
        if regression_data[device][feature][condition]
    )

    corrected_alpha = alpha / total_tests

    for device in regression_data:
        for feature in regression_data[device]:
            for condition in regression_data[device][feature]:
                if regression_data[device][feature][condition]:
                    original_p_value = regression_data[device][feature][condition].get(
                        "p_value", 1
                    )
                    corrected_p_value = min(
                        original_p_value * total_tests, 1
                    )  # Ensures p-value doesn't exceed 1
                    corrected_regression_data[device][feature][condition][
                        "corrected_p_value"
                    ] = corrected_p_value
                    corrected_regression_data[device][feature][condition][
                        "is_significant"
                    ] = (corrected_p_value < corrected_alpha)

    print("Bonferroni correction is done for regression data successfully")
    return corrected_regression_data


###########################################################################
###########################################################################
###########################################################################


def regression_plot(
    regression_data,
    data,
    criterion,
    conditions,
    devices,
    features,
    width=15,
    height_per_condition=4,
    regression_line_style="-",
    regression_line_color="black",
    marker_color="gray",
    font_size=12,
    show_grid=True,
    background_color=None,
    bonferroni_corrected=False,
):
    """
    This function creates scatter plots for the given regression data, data, conditions, features, and devices. It displays the scatter plots along with regression lines, correlation coefficients, significance values (with an option to indicate if they are Bonferroni corrected), and the number of observations.

    Parameters:
    -----------
    regression_data : dict
        A nested dictionary containing the regression results (slope, intercept, r_value, p_value, and std_err) for each device and criterion pair, feature, and condition.
    data : dict
        A nested dictionary containing the data, with keys for devices, features, and conditions.
    criterion : str
        A string representing the device used as a criterion for comparison.
    conditions : list
        A list of strings representing the different experimental conditions of the data.
    devices : list
        A list of strings representing the different devices used to collect the data.
    features : list
        A list of strings representing the different features of the data.
    width : int, optional, default: 15
        The width of the scatter plot in inches.
    height_per_condition : int, optional, default: 4
        The height of each subplot (per condition) in inches.
    regression_line_style : str, optional, default: '-'
        The line style for the regression line.
    regression_line_color : str, optional, default: 'black'
        The color of the regression line.
    marker_color : str, optional, default: 'gray'
        The color of the data point markers.
    font_size : int, optional, default: 12
        The font size for labels, titles, etc.
    show_grid : bool, optional, default: True
        Whether to display gridlines on the plot.
    background_color : str, optional
        Background color for the plot, if specified.
    bonferroni_corrected : bool, optional, default: False
        Indicates if the data has Bonferroni corrected p-values. If True, the function will use Bonferroni corrected p-values for significance annotations.

    Returns:
    --------
    None
    """
    plt.rcParams.update({"font.size": font_size})

    def create_scatter_plots(device, feature):
        num_conditions = len(conditions)
        fig, axs = plt.subplots(
            num_conditions, 1, figsize=(width, height_per_condition * num_conditions)
        )

        # Make sure axs is always a list, even if there's only one condition
        if num_conditions == 1:
            axs = [axs]

        for c, condition in enumerate(conditions):
            device_data = data[device][feature][condition]
            criterion_data = data[criterion][feature][condition]

            filtered_data_device = []
            filtered_data_criterion = []

            for pp_value_device, pp_value_criterion in zip(
                device_data.values(), criterion_data.values()
            ):
                if pp_value_device and pp_value_criterion:
                    filtered_data_device.append(pp_value_device[0])
                    filtered_data_criterion.append(pp_value_criterion[0])

            slope = regression_data[device][feature][condition]["slope"]
            intercept = regression_data[device][feature][condition]["intercept"]
            r_value = regression_data[device][feature][condition]["r_value"]
            p_value_key = (
                "corrected_p_value" if bonferroni_corrected else "p_value"
            )  # It checks if p-values are corrected or not, then take the correct corresponding value from the dic
            p_value = regression_data[device][feature][condition][p_value_key]

            axs[c].scatter(
                filtered_data_device,
                filtered_data_criterion,
                alpha=0.8,
                color=marker_color,
            )
            axs[c].set_xlabel(device if c == num_conditions - 1 else "")
            axs[c].set_ylabel(criterion if c == 0 else "")
            axs[c].set_title(condition.capitalize(), loc="left", fontsize=12)

            # Regression line
            x = np.array(filtered_data_device)
            axs[c].plot(
                x,
                intercept + slope * x,
                regression_line_style,
                color=regression_line_color,
                label=f"y={slope:.2f}x+{intercept:.2f}",
            )

            # Correlation coefficient and significance
            correction_annotation = (
                " (Bonferroni corrected)" if bonferroni_corrected else ""
            )
            n = len(filtered_data_device)
            axs[c].annotate(
                f"r = {r_value:.2f}, p = {p_value:.2f}{correction_annotation}, n = {n}",
                (0.05, 0.9),
                xycoords="axes fraction",
            )

            # Optional Grid
            if show_grid:
                axs[c].grid(True, which="both", linestyle="--", linewidth=0.5)

            # Background color
            if background_color:
                axs[c].set_facecolor(background_color)

        fig.suptitle(
            f"Regression between {device.capitalize()} and {criterion.capitalize()} for {feature.capitalize()}",
            fontsize=16 + 2,
            y=1.02,
        )
        plt.tight_layout()  # to avoid overlaps
        plt.show()

    def update_scatter_plots(*args):
        device = device_dropdown.value
        feature = feature_dropdown.value

        with out:
            clear_output(wait=True)
            create_scatter_plots(device, feature)

    device_dropdown = widgets.Dropdown(
        options=devices[:-1],  # excluding the criterion device
        value=devices[0],
        description="Device:",
        disabled=False,
    )

    feature_dropdown = widgets.Dropdown(
        options=features,
        value=features[0],
        description="Feature:",
        disabled=False,
    )

    device_dropdown.observe(update_scatter_plots, names="value")
    feature_dropdown.observe(update_scatter_plots, names="value")

    out = widgets.Output()
    display(widgets.VBox([device_dropdown, feature_dropdown, out]))
    update_scatter_plots()


###########################################################################
###########################################################################
###########################################################################


def heatmap_plot(data, criterion, devices, conditions, features):
    """
    This function creates a correlation heatmap between each device and the criterion device for each condition and feature.

    Parameters:
    -----------
    data : dict
        A nested dictionary containing the data, with keys for devices, features, and conditions.
    criterion : str
        A string representing the device used as a criterion for comparison.
    devices : list
        A list of strings representing the different devices used to collect the data.
    conditions : list
        A list of strings representing the different experimental conditions of the data.
    features : list
        A list of strings representing the different features of the data.

    Returns:
    --------
    None
    """

    def create_correlation_heatmap(feature):
        # Create an empty dictionary to store the correlation values
        corr_dict = {}

        # Loop through each condition
        for condition in conditions:
            # Create an empty list to store the correlation values for this condition
            corr_values = []
            # Loop through each device (excluding the criterion device)
            for device in devices[:-1]:
                # Get the data for the selected feature, condition, and devices
                device_data = data[device][feature][condition]
                criterion_data = data[criterion][feature][condition]

                filtered_data_device = []
                filtered_data_criterion = []

                for pp, (pp_value_device, pp_value_criterion) in enumerate(
                    zip(device_data.values(), criterion_data.values())
                ):
                    if pp_value_device and pp_value_criterion:
                        filtered_data_device.append(pp_value_device[0])
                        filtered_data_criterion.append(pp_value_criterion[0])

                # Calculate the correlation coefficient between the two sets of filtered data
                corr = np.corrcoef(filtered_data_device, filtered_data_criterion)[0, 1]
                # Add the correlation value to the list
                corr_values.append(corr)
            # Add the list of correlation values to the dictionary for this condition
            corr_dict[condition] = corr_values

        # Convert the dictionary to a pandas dataframe
        corr_df = pd.DataFrame.from_dict(
            corr_dict, orient="index", columns=devices[:-1]
        )

        # Create a heatmap using seaborn
        sns.heatmap(
            corr_df,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": f"Correlation with {criterion} device"},
        )

        # Set the axis labels and plot title
        plt.xlabel("Devices")
        plt.ylabel("Conditions")
        plt.title(f"Correlation of {feature.upper()} with {criterion} device")

        # Show the plot
        plt.show()

    def update_correlation_heatmap(*args):
        feature = feature_dropdown.value

        with out:
            clear_output(wait=True)
            create_correlation_heatmap(feature)

    feature_dropdown = widgets.Dropdown(
        options=features,
        value=features[0],
        description="Feature:",
        disabled=False,
    )

    feature_dropdown.observe(update_correlation_heatmap, names="value")

    out = widgets.Output()
    display(feature_dropdown, out)
    update_correlation_heatmap()


###########################################################################
###########################################################################
###########################################################################


def icc_analysis(data, criterion, devices, conditions, features, path, save_as_csv):
    """
    Calculates the Intraclass Correlation Coefficient (ICC) for comparing the reliability of measurements among devices against a criterion device, across different conditions and features. The ICC assesses the consistency or reproducibility of quantitative measurements made by different observers measuring the same quantity.

    Parameters:
    -----------
    data : dict
        A nested dictionary with structure {device: {feature: {condition: [values]}}}. It contains the measurement data for each device across specified conditions and features.

    criterion : str
        The name of the device used as a reference or standard for comparison in the analysis.

    devices : list of str
        A list containing the names of the devices whose data is analyzed. This should include the criterion device.

    conditions : list of str
        The experimental conditions under which data was collected. Conditions are used to segment and compare data.

    features : list of str
        The specific data features or measurements that are analyzed for reliability.

    path : str or Path
        The path where the result CSV file (if saving is enabled) will be stored. Can be a string or a Path object from the pathlib module.

    save_as_csv : bool
        Determines whether to save the ICC analysis results to a CSV file. If True, the results are saved to the specified path.

    Returns:
    --------
    icc_data : dict
        A nested dictionary with the ICC results organized by device, feature, and condition. Each entry contains the ICC value and its confidence interval, reflecting the measurement consistency.

    Notes:
    -----
    - If `save_as_csv` is True, the function saves the detailed ICC results, including ICC type, value, 95% confidence interval, and p-value, to a CSV file at the specified path, facilitating easy access and further analysis.
    """

    # Defining empty dictionaries to save ICC data analysis
    icc_data = {
        device: {
            feature: {condition: {} for condition in conditions} for feature in features
        }
        for device in devices[:-1]
    }

    for device in devices[:-1]:
        for feature in features:
            for condition in conditions:
                # Check for matching values between the criterion and device
                device_data = data[device][feature][condition]
                criterion_data = data[criterion][feature][condition]

                filtered_data_device = []
                filtered_data_criterion = []

                for pp, (pp_value_device, pp_value_criterion) in enumerate(
                    zip(device_data.values(), criterion_data.values())
                ):
                    if pp_value_device and pp_value_criterion:
                        filtered_data_device.append(pp_value_device[0])
                        filtered_data_criterion.append(pp_value_criterion[0])

                # Calculate ICC if there are matching values
                if filtered_data_device and filtered_data_criterion:
                    icc_device = pd.DataFrame(
                        {
                            "participant": list(
                                range(1, len(filtered_data_device) + 1)
                            ),
                            "device": device,
                            "score": filtered_data_device,
                        }
                    )
                    icc_criterion = pd.DataFrame(
                        {
                            "participant": list(
                                range(1, len(filtered_data_criterion) + 1)
                            ),
                            "device": criterion,
                            "score": filtered_data_criterion,
                        }
                    )
                    df = pd.concat([icc_device, icc_criterion], ignore_index=True)
                    icc = pg.intraclass_corr(
                        data=df, targets="participant", raters="device", ratings="score"
                    )
                    icc_data[device][feature][condition] = icc
                else:
                    icc_data[device][feature][condition] = None

    print("Done Successfully!")

    if save_as_csv:
        # Convert nested dictionary to a list of rows
        rows = []
        for device in icc_data:
            for feature in icc_data[device]:
                for condition in icc_data[device][feature]:
                    row_data = icc_data[device][feature][condition]
                    if row_data is not None:
                        for index, icc_row in row_data.iterrows():
                            rows.append(
                                [
                                    device,
                                    feature,
                                    condition,
                                    icc_row["Type"],
                                    icc_row["ICC"],
                                    icc_row["CI95%"],
                                    icc_row["pval"],
                                ]
                            )
                    else:
                        rows.append(
                            [device, feature, condition, None, None, None, None]
                        )

        # Convert list of rows to DataFrame
        df = pd.DataFrame(
            rows,
            columns=[
                "Device",
                "Feature",
                "Condition",
                "ICC Type",
                "ICC Value",
                "ICC CI95%",
                "p-value",
            ],
        )

        # Save DataFrame to CSV
        path = Path(path)
        save_path = path / "icc_data.csv"
        df.to_csv(save_path, index=False)
        print("ICC data saved successfully!")

    return icc_data


###########################################################################
###########################################################################
###########################################################################


def bonferroni_correction_icc(icc_data, alpha=0.05):
    """
    Applies the Bonferroni correction to the p-values in the ICC analysis results.

    Parameters:
    -----------
    icc_data : dict
        A nested dictionary containing the ICC results for each device, feature, and condition.
    alpha : float, optional, default: 0.05
        The significance level for the tests.

    Returns:
    --------
    corrected_icc_data : dict
        The ICC data with Bonferroni corrected p-values.
    """
    corrected_icc_data = icc_data.copy()
    total_tests = sum(
        1
        for device in icc_data
        for feature in icc_data[device]
        for condition in icc_data[device][feature]
        if icc_data[device][feature][condition] is not None
    )

    corrected_alpha = alpha / total_tests

    for device in icc_data:
        for feature in icc_data[device]:
            for condition in icc_data[device][feature]:
                icc_result = icc_data[device][feature][condition]
                if icc_result is not None:
                    for index, icc_row in icc_result.iterrows():
                        original_p_value = icc_row["pval"]
                        corrected_p_value = min(
                            original_p_value * total_tests, 1
                        )  # Ensuring p-value doesn't exceed 1
                        corrected_icc_data[device][feature][condition].at[
                            index, "corrected_pval"
                        ] = corrected_p_value
                        corrected_icc_data[device][feature][condition].at[
                            index, "is_significant"
                        ] = (corrected_p_value < corrected_alpha)

    print("Bonferroni correction is done for icc data successfully!")
    return corrected_icc_data


###########################################################################
###########################################################################
###########################################################################


def icc_plot(icc_data, conditions, devices, features, font_size=12, cmap="coolwarm"):
    """
    This function creates an interactive heatmap plot for ICC values of each device compared to the criterion device,
    for each condition and feature. The heatmap provides both the ICC values and their corresponding 95% confidence intervals.

    Parameters:
    -----------
    icc_data : dict
        A nested dictionary containing the ICC results (ICC value and 95% confidence interval) for each device, feature, and condition.
    conditions : list
        A list of strings representing the different experimental conditions of the data.
    devices : list
        A list of strings representing the different devices used to collect the data. The last device is assumed to be the criterion device.
    features : list
        A list of strings representing the different features of the data.
    font_size : int, optional, default: 12
        The font size for labels, titles, and annotations.
    cmap : str, optional, default: "coolwarm"
        The color map to be used for the heatmap. This should be a valid matplotlib or seaborn colormap string.

    Returns:
    --------
    None

    Note:
    The function displays an interactive dropdown widget for feature selection and renders the heatmap based on the selected feature.
    """
    plt.rcParams.update({"font.size": font_size})

    def plot_icc_heatmap(feature):
        icc_matrix = []
        ci95_matrix = []

        for condition in conditions:
            icc_row = []
            ci95_row = []

            for device in devices[:-1]:
                icc = icc_data[device][feature][condition]
                if icc is not None:
                    icc1 = icc["ICC"][0]
                    ci95_lower, ci95_upper = icc["CI95%"][0]
                    icc_row.append(icc1)
                    ci95_row.append((ci95_lower, ci95_upper))
                else:
                    icc_row.append(None)
                    ci95_row.append((None, None))

            icc_matrix.append(icc_row)
            ci95_matrix.append(ci95_row)

        icc_df = pd.DataFrame(icc_matrix, columns=devices[:-1], index=conditions)

        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            icc_df,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            cbar_kws={"label": f"ICC1 - {feature.upper()}"},
        )

        for i, condition in enumerate(conditions):
            for j, device in enumerate(devices[:-1]):
                ci95_lower, ci95_upper = ci95_matrix[i][j]
                if ci95_lower is not None and ci95_upper is not None:
                    annotation = f"[{ci95_lower:.2f}, {ci95_upper:.2f}]"
                    ax.text(
                        j + 0.5,
                        i + 0.7,
                        annotation,
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=font_size - 2,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="black",
                            boxstyle="round,pad=0.2",
                            alpha=0.5,
                        ),
                    )

        plt.xlabel("Devices")
        plt.ylabel("Conditions")
        plt.title(f"ICC1 Heatmap for {feature.upper()}")

        plt.show()

    def update_icc_heatmap(*args):
        feature = feature_dropdown.value
        with out:
            clear_output(wait=True)
            plot_icc_heatmap(feature)

    feature_dropdown = widgets.Dropdown(
        options=features,
        value=features[0],
        description="Feature:",
        disabled=False,
    )

    feature_dropdown.observe(update_icc_heatmap, names="value")

    out = widgets.Output()
    display(feature_dropdown, out)
    update_icc_heatmap()


###########################################################################
###########################################################################
###########################################################################


def blandaltman_analysis(
    data, criterion, devices, conditions, features, path, save_as_csv=False
):
    """
    Calculates the Bland-Altman analysis to assess the agreement between measurements from various devices compared to a criterion device, across different conditions and features. This analysis includes calculating bias, limits of agreement, and their 95% Confidence Intervals.

    Parameters:
    -----------
    data : dict
        Nested dictionary containing the data, organized as {device: {feature: {condition: [values]}}}. It includes measurements for each device, feature, and condition.

    criterion : str
        The name of the device used as the standard for comparison.

    devices : list of str
        List of device names whose data will be analyzed.

    conditions : list of str
        Experimental conditions under which the data was collected.

    features : list of str
        Measured features of interest.

    path : str or Path, optional
        Directory path where the results CSV file will be saved if save_as_csv is True. Accepts both string paths and Path objects from the pathlib module.

    save_as_csv : bool, optional
        If True, saves the Bland-Altman analysis results to a CSV file specified by the path parameter. Default is False.

    Returns:
    --------
    blandaltman_data : dict
        Nested dictionary with the Bland-Altman analysis results for each device, feature, and condition, including bias, standard deviation (SD), limits of agreement (LoA), and their 95% Confidence Intervals.

    Notes:
    -----
    - The Bland-Altman analysis is a statistical method used to compare two quantitative measurement techniques.
    - In addition to bias and limits of agreement, this function calculates and includes the 95% Confidence Intervals for both bias and LoAs.
    - If save_as_csv is True and a path is provided, the function outputs the results to a CSV file, facilitating easy access to and further analysis of the data.
    """

    # Defining empty dictionaries to save Bland-Altman data analysis
    blandaltman_data = {
        device: {
            feature: {condition: {} for condition in conditions} for feature in features
        }
        for device in devices[:-1]
    }

    for device in devices[:-1]:
        for feature in features:
            for condition in conditions:
                # Check for matching values between the criterion and device
                device_data = data[device][feature][condition]
                criterion_data = data[criterion][feature][condition]

                filtered_data_device = []
                filtered_data_criterion = []

                for pp, (pp_value_device, pp_value_criterion) in enumerate(
                    zip(device_data.values(), criterion_data.values())
                ):
                    if pp_value_device and pp_value_criterion:
                        filtered_data_device.append(pp_value_device[0])
                        filtered_data_criterion.append(pp_value_criterion[0])

                # Calculate Bland-Altman if there are matching values
                if filtered_data_device and filtered_data_criterion:
                    differences = np.array(filtered_data_device) - np.array(
                        filtered_data_criterion
                    )
                    bias = np.mean(differences)
                    sd = np.std(differences, ddof=1)
                    limits_of_agreement = (bias - 1.96 * sd, bias + 1.96 * sd)

                    # Calculate 95% CI for bias
                    bias_ci = stats.t.interval(
                        0.95,
                        len(differences) - 1,
                        loc=bias,
                        scale=stats.sem(differences),
                    )

                    # Calculate 95% CI for LoAs
                    loa_sd = sd / np.sqrt(len(differences))
                    lower_loa_ci = (
                        limits_of_agreement[0] - 1.96 * loa_sd,
                        limits_of_agreement[0] + 1.96 * loa_sd,
                    )
                    upper_loa_ci = (
                        limits_of_agreement[1] - 1.96 * loa_sd,
                        limits_of_agreement[1] + 1.96 * loa_sd,
                    )

                    blandaltman = {
                        "bias": bias,
                        "sd": sd,
                        "limits_of_agreement": limits_of_agreement,
                        "bias_95_CI": bias_ci,
                        "lower_loa_95_CI": lower_loa_ci,
                        "upper_loa_95_CI": upper_loa_ci,
                    }
                    blandaltman_data[device][feature][condition] = blandaltman
                else:
                    blandaltman_data[device][feature][condition] = None

    print(" Bland-Altman Analysis Done Successfully!")

    if save_as_csv:
        # Convert nested dictionary to a list of rows
        rows = []
        for device in blandaltman_data:
            for feature in blandaltman_data[device]:
                for condition in blandaltman_data[device][feature]:
                    row_data = blandaltman_data[device][feature][condition]
                    if row_data:
                        rows.append(
                            [
                                device,
                                feature,
                                condition,
                                row_data["bias"],
                                row_data["sd"],
                                row_data["limits_of_agreement"][0],
                                row_data["limits_of_agreement"][1],
                                row_data["bias_95_CI"][0],
                                row_data["bias_95_CI"][1],
                                row_data["lower_loa_95_CI"][0],
                                row_data["lower_loa_95_CI"][1],
                                row_data["upper_loa_95_CI"][0],
                                row_data["upper_loa_95_CI"][1],
                            ]
                        )
                    else:
                        rows.append([device, feature, condition] + [None] * 10)

        df = pd.DataFrame(
            rows,
            columns=[
                "Device",
                "Feature",
                "Condition",
                "Bias",
                "SD",
                "Lower Limit of Agreement",
                "Upper Limit of Agreement",
                "Bias 95% CI Lower",
                "Bias 95% CI Upper",
                "Lower LoA 95% CI Lower",
                "Lower LoA 95% CI Upper",
                "Upper LoA 95% CI Lower",
                "Upper LoA 95% CI Upper",
            ],
        )

        path = Path(path)
        df.to_csv(path / "blandaltman_data.csv", index=False)
        print("Blandaltman Data saved successfully!")

    return blandaltman_data


###########################################################################
###########################################################################
###########################################################################


def blandaltman_plot(
    data,
    criterion,
    conditions,
    devices,
    features,
    width=10,
    height_per_plot=5,
    agreement_bound=1.96,
    confidenceInterval=95,
    percentage=False,
    mean_diff_color="#FF6347",
    boundary_color="#20B2AA",
    pointColour="#8B008B",
    shade=True,
):
    # To create this plot, some ideas come from this python package: https://github.com/jaketmp/pyCompare/blob/main/pyCompare/_plotBlandAltman.py
    """
    Generates Bland-Altman plots for data comparison between devices and a criterion.

    Parameters:
    -----------
    blandaltman_data: dict
        The Bland-Altman data.

    data: dict
        The main data containing measurement values from various devices.

    criterion: str
        The name of the criterion device against which comparisons are made.

    conditions: list
        List of conditions under which measurements were taken.

    devices: list
        List of measurement devices including the criterion device.

    features: list
        List of features or metrics being measured.

    width: int, optional, default=10
        Width of the entire plot.

    height_per_plot: int, optional, default=5
        Height of each individual subplot.

    agreement_bound: float, optional, default=1.96
        Multiplier for the standard deviation to determine limits of agreement.

    confidenceInterval: float, optional, default=95
        Confidence interval percentage for shading (if applied).

    percentage: bool, optional, default=False
        If True, plots differences as percentages. Otherwise, plots raw differences.

    mean_diff_color: str, optional, default='#FF6347'
        Color used for the mean difference line.

    boundary_color: str, optional, default='#20B2AA'
        Color used for the upper and lower limits of agreement lines.

    pointColour: str, optional, default='#8B008B'
        Color used for the individual data points.

    shade: bool, optional, default=True
        If True, shades the confidence interval around the mean and limits of agreement.

    Returns:
    --------
    None
        The function displays the Bland-Altman plots and doesn't return any value.

    Note:
    -----
    This function generates Bland-Altman plots for each combination of device (excluding the criterion)
    and feature, under each condition. The plots include the mean difference (bias) and the upper and
    lower limits of agreement, and optionally the shaded confidence intervals.
    """

    sns.set(
        style="whitegrid",
        rc={
            "grid.linewidth": 0.5,
            "grid.color": ".7",
            "ytick.major.size": 5,
            "axes.edgecolor": ".3",
        },
    )
    plt.rcParams["figure.facecolor"] = "white"

    def create_blandaltman_plots(feature, device):
        fig, axs = plt.subplots(
            len(conditions), 1, figsize=(width, height_per_plot * len(conditions))
        )

        for c, condition in enumerate(conditions):
            device_data = data[device][feature][condition]
            criterion_data = data[criterion][feature][condition]

            filtered_data_device = []
            filtered_data_criterion = []

            for pp, (pp_value_device, pp_value_criterion) in enumerate(
                zip(device_data.values(), criterion_data.values())
            ):
                if pp_value_device and pp_value_criterion:
                    filtered_data_device.append(pp_value_device[0])
                    filtered_data_criterion.append(pp_value_criterion[0])

            # Convert lists to numpy arrays and sort them
            filtered_data_device = np.array(filtered_data_device)
            filtered_data_criterion = np.array(filtered_data_criterion)

            mean_diff_pairs = sorted(
                zip(
                    np.mean([filtered_data_device, filtered_data_criterion], axis=0),
                    (
                        (
                            (filtered_data_device - filtered_data_criterion)
                            / np.mean(
                                [filtered_data_device, filtered_data_criterion], axis=0
                            )
                        )
                        * 100
                        if percentage
                        else filtered_data_device - filtered_data_criterion
                    ),
                )
            )
            mean_vals, diff_vals = zip(*mean_diff_pairs)

            md = np.mean(diff_vals)
            sd = np.std(diff_vals, axis=0, ddof=1)

            loa_upper = md + agreement_bound * sd
            loa_lower = md - agreement_bound * sd

            # Plot the data
            axs[c].scatter(mean_vals, diff_vals, alpha=0.5, c=pointColour)
            axs[c].axhline(md, color=mean_diff_color, linestyle="--")
            axs[c].axhline(loa_upper, color=boundary_color, linestyle="--")
            axs[c].axhline(loa_lower, color=boundary_color, linestyle="--")

            if shade:
                # Add shading for confidence intervals
                z_value = stats.norm.ppf((1 + (confidenceInterval / 100)) / 2.0)
                se_loa = sd * np.sqrt((1 / len(diff_vals)) + (4 / 2 / len(diff_vals)))
                loa_range = z_value * se_loa

                axs[c].fill_between(
                    mean_vals,
                    md + loa_range,
                    md - loa_range,
                    color=mean_diff_color,
                    alpha=0.2,
                )
                axs[c].fill_between(
                    mean_vals,
                    loa_upper + loa_range,
                    loa_upper - loa_range,
                    color=boundary_color,
                    alpha=0.2,
                )
                axs[c].fill_between(
                    mean_vals,
                    loa_lower + loa_range,
                    loa_lower - loa_range,
                    color=boundary_color,
                    alpha=0.2,
                )

            # Right-side annotations
            trans = transforms.blended_transform_factory(
                axs[c].transAxes, axs[c].transData
            )
            offset = (loa_upper - loa_lower) * 0.02  # 2% of range for offset

            axs[c].text(
                1.02,
                md,
                f"Mean\n{md:.2f}",
                ha="left",
                va="center",
                transform=trans,
                color=mean_diff_color,
            )
            axs[c].text(
                1.02,
                loa_upper,
                f"+{agreement_bound:.2f} SD\n{loa_upper:.2f}",
                ha="left",
                va="center",
                transform=trans,
                color=boundary_color,
            )
            axs[c].text(
                1.02,
                loa_lower,
                f"-{agreement_bound:.2f} SD\n{loa_lower:.2f}",
                ha="left",
                va="center",
                transform=trans,
                color=boundary_color,
            )

            axs[c].set_title(
                f"{condition.capitalize()}", fontsize=14, color="black", loc="left"
            )

            if c == len(conditions) - 1:
                axs[c].set_xlabel(f"Mean of {device} and {criterion}", fontsize=12)
            axs[c].set_ylabel("Difference", fontsize=12)

        fig.suptitle(
            f"Bland-Altman Plot for {feature.capitalize()} - {device.capitalize()} vs. {criterion.capitalize()}",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.03, 0.95, 0.95])
        plt.show()

    def update_blandaltman_plots(*args):
        feature = feature_dropdown.value
        device = device_dropdown.value

        with out:
            clear_output(wait=True)
            create_blandaltman_plots(feature, device)

    feature_dropdown = widgets.Dropdown(
        options=features,
        value=features[0],
        description="Feature:",
        disabled=False,
    )

    device_dropdown = widgets.Dropdown(
        options=devices[:-1],
        value=devices[0],
        description="Device:",
        disabled=False,
    )

    feature_dropdown.observe(update_blandaltman_plots, names="value")
    device_dropdown.observe(update_blandaltman_plots, names="value")

    out = widgets.Output()
    display(widgets.VBox([feature_dropdown, device_dropdown, out]))
    update_blandaltman_plots()


###########################################################################
###########################################################################
###########################################################################
