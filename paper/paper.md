---
title: 'Wearablehrv: A Python package for the validation of heart rate and heart rate variability in wearables'
tags:
  - Python
  - Psychophysiology
  - Heart rate variability
  - Heart rate
  - HRV
  - Wearables
authors:
  - name: Mohammadamin Sinichi
    orcid: 0009-0008-2491-1542
    corresponding: true
    affiliation: 1
  - name: Martin Gevonden
    ordic: 0000-0001-7867-1443
    affiliation: 2
  - name: Lydia Krabbendam
    orcid: 0000-0003-4074-5149
    affiliation: 1
affiliations:
 - name: Department of Clinical, Neuro- & Developmental Psychology, Faculty of Behavioural and Movement Sciences, Vrije Universiteit Amsterdam, Netherlands
   index: 1
 - name: Department of Biological Psychology, Faculty of Behavioural and Movement Sciences, Vrije Universiteit Amsterdam, Netherlands
   index: 2
date: 10 September 2023
bibliography: paper.bib
---

<!-- Your paper should include:

We also require that authors explain the research applications of the software. The paper should be between 250-1000 words. Authors submitting papers significantly longer than 1000 words may be asked to reduce the length of their paper.

A list of the authors of the software and their affiliations, using the correct format (see the example below).
A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.
A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work.
A list of key references, including to other software addressing related needs. Note that the references should include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a specific discipline.
Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.
Acknowledgement of any financial support. -->

# Summary
Wearable devices that enable us to monitor our physiology have become an integral part of our lives, and the market for these devices is rapidly expanding with new brands and products. Among their various features, heart rate (HR) and heart rate variability (HRV) are particularly interesting, not only for users but also for researchers. However, the question arises: how accurate are these wearables in capturing such sensitive data that require millisecond precision?

The `wearablehrv` is a Python package offers a comprehensive pipeline for validating the accuracy of HR and HRV measurements and it facilitates the process of turning raw data into advanced statistical analyses for device agreement. The package's graphical user interface allows for effortless visualization, pre-processing, and data analysis at both individual and group levels. To start, one can use a basic criterion electrocardiograph (ECG) device, such as a chest strap, to compare against the device of interest under various conditions. Researchers have the flexibility to test multiple devices against a single criterion simultaneously. The only required inputs for the pipeline are the inter-beat intervals (IBIs) and timestamps for each device; the rest is fully handled by the `wearablehrv` package.

# Statement of need
The use of wearables in psychophysiology and sports sciences has exponentially increased over the past decade. Many of these devices measure HR and HRV using the Photoplethysmography (PPG) technique that measures the changes in blood volume in peripheral tissues (e.g., earlobe, wrist, arm, fingertip) and substitutes these pulses for the actual contraction of ventricular muscles[@challoner_photoelectric_1974]. Although the PPG method is promising and versatile, it raises concerns regarding its validity. Several studies have highlighted concerns about the validity of wearables in identifying HR and HRV, especially when the user is in motion [@hill_ethnic_2015;@hill_ethnic_2015;@quintana_guidelines_2016;@nederend_impedance_2017;@nederend_impedance_2017;@schafer_how_2013;@10.3389/fspor.2021.585870;@10.3389/fspor.2021.585870].

One potential reason the validity of these wearables is often overlooked might be the absence of an integrated and user-friendly validation pipeline. Such a tool would guide users from raw data collection through advanced statistical analysis to assess device agreement. The `wearablehrv` package was developed to address this gap. Provided that a wearable device (either PPG or ECG) transfers the IBIs alongside timestamps, this Python package makes it feasible to establish the validity of a wearable in just a few steps. In summary, `wearablehrv` is a Python package tailored for data preparation, pre-processing, feature extraction, comparison, visualization, and both individual and group statistical analyses of heart rate and heart rate variability metrics from wearable devices that transmit raw IBIs and timestamps. The inclusion of graphical user interfaces (GUI) in most functions grants researchers the flexibility to easily switch between experimental conditions and devices. This offers versatility in validating an unlimited number of wearables within a single experimental setting and under various conditions.

# Main Features and Basic Usage
In this section, we offer a brief overview of the primary functions and workflow of `wearablehrv`. For detailed documentation of all the functions and examples on data recording and initiating validation, please refer to [README.md](link) and [documentation.ipynb](link). The pipeline is divided into two parts: individual and group:

```python 
from wearablehrv import individual
from wearablehrv import group
```
## Individual Pipeline
In the individual pipeline, one can establish and compare the validity of a wearable of interest against a criterion device (e.g., an ECG) for a single case. This comparison can be conducted under just one condition (e.g., sitting rest) or an unlimited number of both laboratory and ambulatory conditions.
### Recording Data and Getting Started 
The data required for the pipeline should have its first column populated with UNIX timestamps with millisecond precision, and the second column should contain IBIs. Once this continuous recording is saved as a .csv file using the `[participant ID]_[device name].csv` format, and once preliminary variables are specified (e.g., conditions, participantID, devices), the rest of the process is seamlessly managed by the pipeline.
### Defining Events and Importing the Recorded Raw Data
Either with a pre-specified .csv file (`already_saved= True`) or using the GUI (`already_saved= False`), the start and end of each experimental condition can be specified:

```python
events = individual.define_events(path, pp, conditions, already_saved=True, save_as_csv=False)
```

The continuous data from all devices can then be imported using the following function:

```python
data = individual.import_data(path, pp, devices)
```

Finally, the continuous data can be segmented into smaller chunks according to the specified experimental conditions, for all devices at once:

```python
data_chopped = individual.chop_data(data, conditions, events, devices)
```
### Visually Inspecting the Signal
One of the primary strengths of `wearablehrv` is its ability to allow researchers to visualize the IBI signals from all devices against the criterion device across all conditions simultaneously. This addresses a prevalent challenge in the field – correcting the lag between devices. With the assistance of the GUI, adjusting for this lag becomes straightforward using the following function:  

```python
individual.visual_inspection (data_chopped, devices, conditions,criterion)
```
![The black line shows the IBIs of the criterion device (ECG). The red line represents the IBIs of a given PPG device. By toggling the `Device` and `Conditions` widgets, one can easily explore other devices and conditions. The `Lag` slider allows for lag correction between the devices, and it is possible to crop a part of the signal if necessary.](visual_inspection.png)
### Pre-processing and Features Extraction
Using the `hrv-analysis` python package [@Champseix2021], we can preprocess and extract all the time-domain and frequency-domain features for all devices and conditions in one go:

```python
individual.pre_processing (data_chopped, devices, conditions, method="karlsson", custom_removing_rule = 0.25, low_rri=300, high_rri=2000)
time_domain_features, frequency_domain_features = wearablehrv.individual.data_analysis (data_pp, devices, conditions)
```
### Plotting and Data Saving
There are various plotting options available to assist in comparing the criterion device with a specific device. These include:
- `individual.result_comparison_plot()`
- `individual.bar_plot()`
- `individual.line_plot()`
- `individual.radar_plot()`
- `individual.unfolding_plot()`

Once ready, all time and frequency features for every device and condition can be exported as a .csv file using the following function, to be later used in the `Group Pipeline`:

```python
individual.save_data(pp, path, time_domain_features, frequency_domain_features, data_pp, devices, conditions, events, artefact=None, nibi_before_cropping=None, nibi_after_cropping=None, save_as_csv=False)
```
## Group Pipeline 
To establish the validity of the devices in terms of their HRV and HR features at the group level, the following function can be used. This function imports all individual cases previously analyzed using the `Individual Pipeline` for further analysis:

```python
data, file_names = group.import_data(path, conditions, devices, features)
```
### Signal Quality Check
A significant advancement in wearable validation is the ability to identify the amount of missing data in each device, assess signal quality, and generate detailed reports. These reports can show, for each participant, device, and condition, the number of detected beats and artifacts. They can also provide a decision on whether to keep or exclude the data, along with a summary report displaying the total count and percentage of decisions ("Acceptable", "Poor", "Missing") for each device and condition. By setting arbitrary thresholds, all of these can be achieved using the following function:

```python
data, features = group.signal_quality(data, path, conditions, devices, features, criterion, file_names, exclude=False, save_as_csv=False, ibi_threshold=0.40, artefact_threshold=0.20)
```
### Various Plotting Options
Subsequent to data processing, one can visualize the entirety of data for all cases, conditions, and devices using various of interactive plotting methods and GUI functionalities. The suite of available functions includes:

- `group.violin_plot()`
- `group.box_plot()`
- `group.radar_plot()`
- `group.hist_plot()`
- `group.matrix_plot()`
- `group.density_plot()`

![An example of the radar plot at the group level, illustrating a comparison between a criterion device and a specified device for `pnni_50`, `rmssd`, `sdnss` (all distinct extracted features for HRV), and mean heart rate (`mean_hr`). Utilizing the `Device` and `Condition` widgets facilitates easy transitioning between a multitude of devices and conditions.](radar_plot.png)
### Main Statistical Analysis
The core objective is to establish the validity of the wearables in comparison to a criterion device. This validation can be accomplished using the main methods designed for evaluating the agreement between two devices: regression analysis, intraclass correlation coefficient (ICC), and Bland-Altman analysis. Moreover, visualization for each method is possible:
#### Regression Analysis
```python
regression_data = group.regression_analysis(data, criterion, conditions, devices, features, path, save_as_csv=True)
group.regression_plot(regression_data, data, criterion, conditions, devices, features, width=20, height=20)
group.heatmap_plot(data, criterion, devices, conditions, features)
```
#### Intraclass Correlation Coefficient (ICC)
```python
icc_data = group.icc_analysis(data, criterion, devices, conditions, features, path, save_as_csv=False)
group.icc_plot(icc_data, conditions, devices, features)
```
#### Bland-Altman Analysis
```python
blandaltman_data = group.blandaltman_analysis(data, criterion, devices, conditions, features, path, save_as_csv=False)
group.blandaltman_plot(blandaltman_data, data, criterion, conditions, devices, features, width=20, height=20)
```
# Acknowledgements

We acknowledge contributions from Dr. Denise J. van der Mee and [to be completed]

# References 