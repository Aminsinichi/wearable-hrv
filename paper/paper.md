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
    affiliation: "1, 3"
  - name: Martin Gevonden
    ordic: 0000-0001-7867-1443
    affiliation: "2, 3"
  - name: Lydia Krabbendam
    orcid: 0000-0003-4074-5149
    affiliation: "1, 3"
affiliations:
 - name: Department of Clinical, Neuro- & Developmental Psychology, Faculty of Behavioural and Movement Sciences, Vrije Universiteit Amsterdam, The Netherlands
   index: 1
 - name: Department of Biological Psychology, Faculty of Behavioural and Movement Sciences, Vrije Universiteit Amsterdam, The Netherlands
   index: 2
 - name: Institute Brain and Behaviour (iBBA), Amsterdam, The Netherlands
   index: 3
date: 10 September 2023
bibliography: paper.bib
---
# Summary

Wearable devices that monitor physiology have become ubiquitous, and include smart watches, smart jewellery, and smart textiles. The market for these devices is rapidly expanding with new brands and products. These devices measure a variety of signals, which are translated into a large amount of different features, of which heart rate (HR) and heart rate variability (HRV) are among the most common. These features are particularly interesting, not only for consumers but also to researchers, because they are predictive of mental and physical health outcomes, and easily obtained. However, for manufacturers, there may be a trade-off between user acceptability on one hand and accuracy on the other, where profit margins typically turn out to be the decisive factor. Therefore, the following question continually comes up: is the cardiac data recorded by this new watch/ring/shirt accurate enough to use in research?

The `wearablehrv` Python package offers a comprehensive pipeline for validating the accuracy of HR and HRV measurements. It allows for advanced statistical analyses on device agreement from beat-to-beat cardiac data. The package's graphical user interface (GUI) facilitates pre-processing, visualization, and data analysis at both individual and group levels. As input, a user of `wearablehrv` should use a criterion device (i.e., a gold standard), preferably an electrocardiograph (ECG). Simultaneously acquired data from the device(s) of interest can then be compared against the criterion. The only required inputs for the pipeline are the inter-beat intervals (IBIs) and timestamps for each device; the rest is fully handled by the `wearablehrv` package.

# Statement of Need

The use of wearables in psychophysiology and sports sciences has exponentially increased over the past decade. While the golden standard is electrocardiography, for prolonged monitoring many studies use devices which measure HR and HRV using Photoplethysmography (PPG). This technique optically measures the changes in blood volume in peripheral tissues (e.g., earlobe, wrist, arm, fingertip) and substitutes detected pulses for the actual contraction of cardiac ventricular muscles[@challoner_photoelectric_1974]. Although the PPG method is promising and versatile, several studies have highlighted concerns about its validity in identifying HR and HRV, especially when the user is in motion [@hill_ethnic_2015;@pinheiro_can_2016;@quintana_guidelines_2016;@nederend_impedance_2017;@allen_photoplethysmography_2007;@schafer_how_2013;@stone_assessing_2021;@bent_investigating_2020].

The oversight regarding the validity of these wearables may be due to the lack of an integrated and user-friendly method for assessing the validity of new wearables. A validation pipeline could lead users from collecting raw data, through pre-processing and advanced statistical analysis, to obtaining the necessary parameters and visualizations required to assess device agreement. The `wearablehrv` package was developed to address this gap. Other existing packages such as `hrv-analysis` [@Champseix2021], NeuroKit2 [@Makowski2021neurokit], pyHRV [@Gomes2019], and similar toolkits offer solutions for pre-processing, analysis, and visualization once IBIs from a single device are provided. However, the added value of `wearablehrv` is rooted in a few key aspects.

First, the currently available packages are not tailored for validation purposes. This becomes especially noticeable when validating multiple wearables at once, across different experimental conditions. This manual process can become a cumbersome task with the packages currently at hand. Second, to make the validation of wearables easier, a user-friendly solution is needed, which current packages lack. This is particularly notable for essential validation steps, such as correcting linear and non-linear lags between devices or trimming signals for a specific device or condition, steps that `wearablehrv` streamlines with GUIs. Third, establishing the validity of a wearable against a gold standard involves numerous decisions. Our pipeline provides a thorough method for users to do this. The availability of such a pipeline also encourages researchers to contribute to establishing a standardized validation protocol, unifying the approach, reducing variability between methods, and facilitating result comparisons. Finally, the division of the `wearablehrv` pipeline into individual and group pipelines offers an advantage for different types of users. The individual pipeline is designed for processing data from a single participant, whereas the group pipeline offers comprehensive tools to establish the quality of the signals, device agreement, and validity of the devices across multiple participants. Most common statistical analyses in validation studies, such as mean absolute percentage error, regression analysis, intraclass correlation coefficient, and Bland-Altman analysis, are already incorporated into the pipeline to streamline the validation process [@BRUTON200094;@d66d4fe8-8a27-315e-bbce-f72f450c6450;@MAKRIDAKIS1993527;@Haghayegh_2020].

In summary, provided that a wearable device (either PPG or ECG) allows for the export of the complete time series of recorded IBIs, this Python package makes it relatively easy to establish the validity of a novel wearable in just a few steps. The inclusion of GUI in most functions grants researchers and wearable users the flexibility in validating an unlimited number of wearables across a range of conditions.

# Main Features and Basic Usage

In this section, we provide an overview of the main functions and basic usage of the `wearablehrv` pipeline. The complete documentation of the API and modules can be found [here](https://wearable-hrv.readthedocs.io/en/latest/). Please also refer to the [README.md](https://github.com/Aminsinichi/wearable-hrv/blob/master/README.md), which provides links to several Jupyter Notebook examples.

The pipeline is divided into two parts: the individual pipeline, which focuses on the data from a single individual, and the group pipeline, which combines the data from all the participants in a single study.

![A summary of the main functionalities of wearablehrv Python package.](main_plot2.png)

## Individual pipeline

The individual pipeline allows for the validation of a wearable of interest against a criterion device (e.g., a gold-standard ECG) under unlimited conditions. Data necessary for the pipeline should be in a .csv file with UNIX timestamps (specified in milliseconds) and IBIs, in two separate columns. For defining events and importing recorded data, the pipeline can be initiated with the `define_events` function, either by referencing a pre-specified .csv file or by using the GUI. The `import_data` function is then used to bring the continuous time series from all devices into the pipeline. Data segmentation into smaller, condition-specific chunks is achieved with the `chop_data` function.

One of the primary strengths of `wearablehrv` is the `visual_inspection` function, which allows for simultaneous visualization of IBI signals and addresses the challenge of correcting devices lag when their internal clocks are not in sync. With the assistance of the GUI, adjusting for this lag becomes straightforward.

![The black line shows the IBIs of the criterion device (ECG). The red line represents the IBIs of a given PPG device. By toggling the `Device` and `Conditions` widgets, one can easily explore other devices and conditions. The `Lag` slider allows for lag correction between the devices, and it is possible to crop a part of the signal if necessary.](visual_inspection.png)

Pre-processing and feature extraction are streamlined using the `pre_processing` function and the `data_analysis` function from the `hrv-analysis` python package[@Champseix2021], outputting time domain and frequency domain features for all devices and conditions.
To facilitate comparing the criterion device with the device of interest and visualization of results, several plotting functions are provided, including `result_comparison_plot`, `bar_plot`, `line_plot`, `radar_plot`, and `unfolding_plot`. All time and frequency features for every device and condition can be exported for later use in the `Group Pipeline` via the `save_data` function.

## Group pipeline

Moving to group-level analysis, the `import_data` function of the group module aggregates individual case analyses. A significant advancement in wearable validation is the ability to identify the amount of missing, poor and acceptable data in each device, and generate detailed reports. The `signal_quality` function allows for such signal quality assessment across devices and participants, providing detailed reports for informed decision-making, and meaningfully visualizing them using `signal_quality_plot1` and `signal_quality_plot2`.

![ An example of a signal quality plot, achievable by setting a few thresholds. Each bar corresponds to a device, and the y-axis shows the percentage of poor, acceptable, and missing data in each of them.](signal_quality.PNG)

For visualization across the aggregated dataset, the group module offers `violin_plot`, `box_plot`, `radar_plot`, `hist_plot`, and `matrix_plot`.

![An example of the radar plot at the group level, illustrating a comparison between a criterion device and a specified device for `pnni_50`, `rmssd`, `sdnss` (all distinct extracted features for HRV), and mean heart rate (`mean_hr`). Utilizing the `Device` and `Condition` widgets facilitates easy transitioning between a multitude of devices and conditions.](radar_plot.png)

The culmination of the group pipeline involves the main statistical analyses — mean absolute percentage error, regression analysis, ICC, and Bland-Altman analysis — enabled by the `mape_analysis`, `regression_analysis`, `icc_analysis`, and `blandaltman_analysis` functions, respectively. These are complemented by their corresponding plotting functions, which can be used to make decisions about device validity, or included in a publication.

# Acknowledgements

This study is an output of the ALIVE project (Improving Adolescent mentaL health by reducing the Impact of PoVErty). This work was supported by the Wellcome Trust [221940/Z/20/Z]. We acknowledge contribution from Dr. Denise J. van der Mee and advice from Dr. Marco Altini.

# References
