# wearablehrv - Individual Pipeline Guide

This README provides a brief guide to navigate through the example notebooks included in the "individual" folder for the `wearablehrv` package. These notebooks are designed to assist you in preparing, processing, analyzing, and plotting data from wearable devices for heart rate variability, for once participant.

## Overview

1. **individual_data_preparation.ipynb**
   - Learn how to prepare your datasets for compatibility with `wearablehrv`. This includes naming conventions, data structuring, and importing experimental events.
   - Compatibility Tip: For users of Empatica Embrace Plus, Labfront, or VU-AMS, refer to `individual_compatibility.ipynb` for device-specific data preparation guidance.

2. **individual_data_preprocessing.ipynb**
   - Instructions on preprocessing interbeat intervals, including correcting device lags, segmenting continuous data, visual inspections, and removing outliers and ectopic beats.

3. **individual_data_analysis.ipynb**
   - Analyze preprocessed data to calculate HR and HRV metrics, including both time-domain and frequency-domain features. Learn how to save these calculations for further analysis.

4. **individual_data_plotting.ipynb**
   - Discover various plotting options to visualize your analysis results. This includes comparison plots, unfolding plots, bar plots, line plots, radar plots, and more.

5. **individual_compatibility.ipynb**
   - Specific instructions for converting data from Labfront, Empatica Embrace Plus, and VU-AMS into a format readable by `wearablehrv`. This notebook is essential if your data comes from these systems.