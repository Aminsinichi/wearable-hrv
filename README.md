
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Current version at PyPI](https://img.shields.io/pypi/v/wearablehrv.svg)](https://pypi.org/project/wearablehrv/)
![Supported Python Versions](https://img.shields.io/pypi/pyversions/wearablehrv.svg)
![Last Commit](https://img.shields.io/github/last-commit/Aminsinichi/wearable-hrv)
[![Twitter Follow](https://img.shields.io/twitter/follow/AminSinichi.svg?style=social)](https://twitter.com/AminSinichi)

`wearablehrv` is an open-source Python package tailored for data preparation, pre-processing, feature extraction, comparison, visualization, and both individual and group statistical analyses of heart rate and heart rate variability metrics from wearable devices that transmit raw inter-beat intervals and timestamps. The inclusion of graphical user interfaces in most functions grants researchers the flexibility to easily switch between experimental conditions and devices. This offers versatility in validating an unlimited number of wearables within a single experimental setting and under various conditions. The only required inputs for the pipeline are the inter-beat intervals and timestamps for each device; the rest is fully handled by the `wearablehrv` package. The main functionalities of this Python package are:

![Image Description](https://showme.redstarplugin.com/d/d:N6ru0hU4)

**Individual Pipeline**:

1. Define experimental events by importing raw data from an unlimited number of devices and experimental conditions.
2. Visualize the inter-beat-interval against the criterion device using an intuitive graphical user interface.
3. Correct for lag between devices with millisecond precision and crop the signals when necessary.
4. Pre-process and calculate both time-domain and frequency-domain measures in one go for all devices and conditions.
5. Provide various plotting options to compare the criterion devices with a specific device and establish its validity.

**Group Pipeline**:

1. Import all individual cases, perform an extensive signal quality check and analysis, and exclude outliers if necessary based on modifiable cutoffs.
2. Offer many descriptive plots to visualize the entirety of data for all cases, conditions, and devices.
3. Conduct the most important statistical analyses in one go and with one click for all devices and conditions against the criterion device, including regression analysis, intraclass correlation coefficient, and Bland-Altman analysis.

# Documentation

For an in-depth explanation of the package and sample data, please refer to:

[![Documentation](https://img.shields.io/badge/Read-Documentation-blue)](https://github.com/Aminsinichi/wearable-hrv/blob/master/notebooks/documentation.ipynb)

For the complete documentation of the API and modules, visit:

[![Documentation Status](https://readthedocs.org/projects/wearable-hrv/badge/?version=latest)](https://wearable-hrv.readthedocs.io/en/latest/?badge=latest)

# Questions

For any questions regarding the package, please contact:

- <aminsinichi@gmail.com>
- <m.sinichi@vu.nl>

# Dependencies

**Standard Libraries**

- [datetime](https://docs.python.org/3/library/datetime.html)
- [os](https://docs.python.org/3/library/os.html)
- [json](https://docs.python.org/3/library/json.html)
- [pickle](https://docs.python.org/3/library/pickle.html)

**Data Analysis & Manipulation**

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)

**Visualization**

- [plotly](https://plotly.com/python/)
  - graph_objects
  - express
- [matplotlib](https://matplotlib.org/)
  - pyplot
  - dates
  - transforms
- [seaborn](https://seaborn.pydata.org/)

**User Interface**

- [tkinter](https://docs.python.org/3/library/tkinter.html)
- [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/)
  - IntText, Dropdown, Output, HBox
- [IPython](https://ipython.org/)
  - display, clear_output, Markdown

**Statistical Analysis**

- [pingouin](https://pingouin-stats.org/)
- [scipy](https://www.scipy.org/)
  - stats
  - linregress
- [statsmodels](https://www.statsmodels.org/stable/index.html)
  - api
- [copy](https://docs.python.org/3/library/copy.html)
  - deepcopy

**Heart Rate Variability Analysis**

- [hrvanalysis](https://pypi.org/project/hrv-analysis/)
  - remove_outliers, remove_ectopic_beats, interpolate_nan_values
  - get_time_domain_features
  - get_frequency_domain_features

**Data Serialization**

- [avro](https://avro.apache.org/)
  - datafile.DataFileReader
  - io.DatumReader

# User Installation

The package can be easily installed using `pip`:

    pip install wearablehrv

The repository can be cloned:

    git clone https://github.com/Aminsinichi/wearable-hrv.git

# GitHub

<https://github.com/Aminsinichi/wearable-hrv>

# Development

`wearablehrv` was developed by Amin Sinichi <https://orcid.org/0009-0008-2491-1542>, during his PhD at Vrije Universiteit Amsterdam in Psychophysiology and Neuropsychology.  

**Contributors**

- [Dr. Martin Gevonden](https://orcid.org/0000-0001-7867-1443)
- [Prof dr. Lydia Krabbendam](https://orcid.org/0000-0003-4074-5149)
