import sys
sys.path.append(r"C:\Users\msi401\OneDrive - Vrije Universiteit Amsterdam\PhD\Data\Coding\Validation Study\wearable-hrv")
import unittest
from wearablehrv import individual
import pandas as pd
import os
import time
import pickle
import tkinter as tk
import hrvanalysis
from unittest.mock import patch, MagicMock
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import Dropdown, IntText, Output, HBox
from IPython.display import display, clear_output

path = os.path.dirname(os.path.abspath(__file__)) + "/" 

class TestIndividual(unittest.TestCase):
    
    def setUp(self):
        # Define the setup variables 
        self.path = path  
        self.pp = "test"
        self.criterion = "vu"
        self.conditions = ['sitting', 'arithmetic', 'recovery', 'standing', 'breathing', 
                           'neurotask', 'walking', 'biking']
        self.devices = ["empatica", "heartmath", "kyto", "rhythm", "vu"]

#############################################################################################################################

    def test_define_events_read(self):
        events = individual.define_events(self.path, self.pp, self.conditions, already_saved=True, save_as_csv=True)
        
        # Assert that the output is a DataFrame
        self.assertIsInstance(events, pd.DataFrame)
        
        # Assert that the DataFrame has the expected columns
        expected_columns = ['timestamp', 'conditions', 'datapoint']
        self.assertListEqual(list(events.columns), expected_columns)
        
        # Assert that the conditions in the DataFrame match the expected conditions
        unique_conditions = events['conditions'].unique()
        self.assertListEqual(sorted(unique_conditions), sorted(self.conditions))

        # Test the content of the CSV file
        test_file_events = path + self.pp + "_events.csv"
        saved_data_events = pd.read_csv(test_file_events)
        self.assertEqual(saved_data_events["timestamp"].tolist()[0], "14:09:20")

    @patch('tkinter.Tk.mainloop', MagicMock(side_effect=[None]))  # Mocking the mainloop to exit immediately
    def test_define_events_gui(self):
        # Test if the GUI runs without error
        try:
            events = individual.define_events(self.path, self.pp, self.conditions, already_saved=False, save_as_csv=False)
            gui_works = True
        except:
            gui_works = False
        
        self.assertTrue(gui_works)

#############################################################################################################################

    def test_import_data(self):
        data = individual.import_data(self.path, self.pp, self.devices)
        
        # Assert that the output is a dictionary
        self.assertIsInstance(data, dict)
        
        # Assert that the dictionary has the expected keys
        self.assertListEqual(sorted(data.keys()), sorted(self.devices))
        
        # For each device, verify the returned DataFrame has the expected columns
        for device, df in data.items():
            self.assertIsInstance(df, pd.DataFrame)
            expected_columns = ['timestamp', 'rr']
            self.assertListEqual(list(df.columns), expected_columns)

        # Test the data integrity 
        self.assertEqual(data["empatica"]["rr"][20], [944])
        self.assertEqual(data["vu"]["rr"][20], [1017])

#############################################################################################################################

    def test_chop_data(self):
        # Using the previously imported data and events for this test
        data = individual.import_data(self.path, self.pp, self.devices)
        events = individual.define_events(self.path, self.pp, self.conditions, already_saved=True, save_as_csv=False)
        
        data_chopped = individual.chop_data(data, self.conditions, events, self.devices)
        
        # Assert that the output is a dictionary
        self.assertIsInstance(data_chopped, dict)
        
        # Assert that the dictionary has the expected keys (device names)
        self.assertListEqual(sorted(data_chopped.keys()), sorted(self.devices))
        
        # For each device, verify the returned DataFrame has the expected columns for each condition
        for device, conditions_dict in data_chopped.items():
            self.assertListEqual(sorted(conditions_dict.keys()), sorted(self.conditions))
            for condition, df in conditions_dict.items():
                self.assertIsInstance(df, pd.DataFrame)
                expected_columns = ['timestamp', 'rr']
                self.assertListEqual(list(df.columns), expected_columns)


        # Test the data integrity 
        self.assertEqual (data_chopped["vu"]["sitting"]["rr"].iloc[0], 1137)
        self.assertEqual (data_chopped["heartmath"]["standing"]["rr"].iloc[10], 992)


#############################################################################################################################
    
    def test_calculate_ibi(self):
        # Using the previously imported data, events, and chopped data for this test
        data = individual.import_data(self.path, self.pp, self.devices)
        events = individual.define_events(self.path, self.pp, self.conditions, already_saved=True, save_as_csv=False)
        data_chopped = individual.chop_data(data, self.conditions, events, self.devices)
        
        nibis = individual.calculate_ibi(data_chopped, self.devices, self.conditions)
        
        # Assert that the output is a dictionary
        self.assertIsInstance(nibis, dict)
        
        # Assert that the dictionary has the expected keys (device names)
        self.assertListEqual(sorted(nibis.keys()), sorted(self.devices))
        
        # For each device, verify the returned dictionary has the expected conditions as keys 
        # and the values represent the correct number of IBIs
        for device, conditions_dict in nibis.items():
            self.assertListEqual(sorted(conditions_dict.keys()), sorted(self.conditions))
            for condition, ibi_count in conditions_dict.items():
                expected_ibi_count = len(data_chopped[device][condition])
                self.assertEqual(ibi_count, expected_ibi_count)

        # Test the data integrity (confirmed by manually checking the excel output)
        self.assertEqual (nibis["kyto"]["sitting"], 280)
        self.assertEqual (nibis["rhythm"]["biking"], 251)

#############################################################################################################################

    @patch('tkinter.Tk.mainloop', MagicMock(side_effect=[None]))  # Mocking the mainloop to exit immediately
    def test_visual_inspection_old_gui(self):
        # Test if the GUI runs without error
        try:
            data = individual.import_data(self.path, self.pp, self.devices)
            events = individual.define_events(self.path, self.pp, self.conditions, already_saved=True, save_as_csv=False)
            data_chopped = individual.chop_data(data, self.conditions, events, self.devices)
            
            individual.visual_inspection_old(data_chopped, self.devices, self.conditions, "vu")
            gui_works = True
        except:
            gui_works = False
        
        self.assertTrue(gui_works)

#############################################################################################################################
    
    def test_save_backup(self):
        # Create a mock data_chopped dictionary
        mock_data = {
            "device1": {
                "condition1": pd.DataFrame({
                    "timestamp": ["2021-01-01 12:00:00", "2021-01-01 12:01:00"],
                    "rr": [1000, 1100]
                })
            }
        }
        
        # Call the function to save the mock data
        individual.save_backup(self.pp, self.path, mock_data)
        
        # Load the saved data
        filename = os.path.join(self.path, f"{self.pp}_data_chopped.pkl")
        with open(filename, "rb") as file:
            loaded_data = pickle.load(file)
        
        # Compare the loaded data with the mock data
        for device, conditions in mock_data.items():
            for condition, df in conditions.items():
                self.assertTrue(df.equals(loaded_data[device][condition]))

        # Clean up by removing the saved pickle file
        os.remove(filename)

#############################################################################################################################

    def test_import_backup(self):
        # Create a mock data_chopped dictionary
        mock_data = {
            "device1": {
                "condition1": pd.DataFrame({
                    "timestamp": ["2021-01-01 12:00:00", "2021-01-01 12:01:00"],
                    "rr": [1000, 1100]
                })
            }
        }

        # Save the mock data to a pickle file
        filename = os.path.join(self.path, f"{self.pp}_data_chopped.pkl")
        with open(filename, "wb") as file:
            pickle.dump(mock_data, file)

        # Load the data using the function
        loaded_data = individual.import_backup(self.pp, self.path)

        # Compare the loaded data with the mock data
        for device, conditions in mock_data.items():
            for condition, df in conditions.items():
                self.assertTrue(df.equals(loaded_data[device][condition]))

        # Clean up by removing the saved pickle file
        os.remove(filename)

#############################################################################################################################

    # For pre-processing and analysis of inter-beat intervals, we utilize the hrv-analysis Python package. 
    # Below, I will test whether the integrated function works correctly. For a more comprehensive test, please refer to the following:
    # https://github.com/Aura-healthcare/hrv-analysis/blob/master/tests/tests_preprocessing_methods.py
    
    def test_pre_processing(self):
        # Create mock data_chopped dictionary
        mock_data = {
            "device1": {
                "condition1": pd.DataFrame({
                    "rr": [1000, 1100, 3000, 50, None, 1000]
                })
            }
        }

        # Call the function
        data_pp, data_chopped_returned = individual.pre_processing(mock_data, ["device1"], ["condition1"])

        # Ensure data structures are maintained
        self.assertIsInstance(data_pp, dict)
        self.assertIsInstance(data_chopped_returned, dict)

        # Verify the device and condition keys
        self.assertIn("device1", data_pp)
        self.assertIn("condition1", data_pp["device1"])

        # Verify the preprocessing steps
        self.assertNotIn(3000, data_pp["device1"]["condition1"])  # Check outlier removal
        self.assertNotIn(50, data_pp["device1"]["condition1"])    # Check outlier removal
        self.assertNotIn(None, data_pp["device1"]["condition1"])  # Check NaN interpolation

#############################################################################################################################

    def test_calculate_artefact(self):
        # Create mock data_chopped and data_pp dictionaries
        mock_data_chopped = {
            "device1": {
                "condition1": [1000, 1100, 1200, 1300, 1400]
            }
        }
        mock_data_pp = {
            "device1": {
                "condition1": [1000, 1110, 1200, 1310, 1410]  # 3 values differ from mock_data_chopped
            }
        }

        # Call the function
        artefact_result = individual.calculate_artefact(mock_data_chopped, mock_data_pp, ["device1"], ["condition1"])

        # Ensure the result structure is maintained
        self.assertIsInstance(artefact_result, dict)

        # Verify the device and condition keys
        self.assertIn("device1", artefact_result)
        self.assertIn("condition1", artefact_result["device1"])

        # Verify the number of artifacts
        self.assertEqual(artefact_result["device1"]["condition1"], 3)  # 3 values differ in mock data

#############################################################################################################################

    @patch('matplotlib.pyplot.show', MagicMock())  # Mocking the plot display
    def test_ibi_comparison_plot(self):
        # Create mock data_chopped and data_pp dictionaries
        mock_data_chopped = {
            "device1": {
                "condition1": [1000, 1100, 1200, 1300, 1400]
            },
            "criterion_device": {
                "condition1": [1000, 1110, 1210, 1310, 1410]
            }
        }
        mock_data_pp = {
            "device1": {
                "condition1": [1000, 1110, 1200, 1310, 1410]  # some values differ from mock_data_chopped
            },
            "criterion_device": {
                "condition1": [1000, 1120, 1220, 1320, 1420]  # some values differ from mock_data_chopped
            }
        }

        # Test if the function runs without error
        try:
            individual.ibi_comparison_plot(mock_data_chopped, mock_data_pp, ["device1"], ["condition1"], "criterion_device")
            plot_works = True
        except:
            plot_works = False

        self.assertTrue(plot_works)

#############################################################################################################################

    # Tests to validate the calculations of the hrv-analysis Python module can be found here:
    # https://github.com/Aura-healthcare/hrv-analysis/blob/master/tests/tests_extract_features_methods.py
    # Here, we are solely testing the proper functionality of our function implementation.

    def test_data_analysis(self):
        # Create mock data_pp dictionary
        mock_data_pp = {
            "device1": {
                "condition1": [800, 750, 820, 810, 790, 820, 805, 795, 810, 825,
                815, 805, 790, 800, 820, 815, 790, 810, 825, 805,
                800, 810, 790, 820, 805, 800, 810, 790, 800, 825,
                815, 810, 790, 820, 800, 795, 810, 825, 805, 810,
                820, 790, 800, 810, 795, 815, 825, 800, 810, 790]
            }
        }

        # Call the function
        time_domain_result, frequency_domain_result = individual.data_analysis(mock_data_pp, ["device1"], ["condition1"])

        # Ensure data structures are maintained
        self.assertIsInstance(time_domain_result, dict)
        self.assertIsInstance(frequency_domain_result, dict)

        # Verify the device and condition keys
        self.assertIn("device1", time_domain_result)
        self.assertIn("condition1", time_domain_result["device1"])

        self.assertIn("device1", frequency_domain_result)
        self.assertIn("condition1", frequency_domain_result["device1"])

        # Verify all expected keys in the results (depending on what the hrvanalysis package returns)
        expected_time_domain_keys = ['mean_nni', 'sdnn', 'sdsd', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20', 'rmssd', 'median_nni', 'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr']
        for key in expected_time_domain_keys:
            self.assertIn(key, time_domain_result["device1"]["condition1"])

        expected_frequency_domain_keys = ['lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu', 'total_power', 'vlf']
        for key in expected_frequency_domain_keys:
            self.assertIn(key, frequency_domain_result["device1"]["condition1"])

        # Test data integrity (Tested Against Kubios HRV Standard 3.5.0)
        self.assertEqual (round (time_domain_result["device1"]["condition1"]['mean_nni']), 806)
        self.assertEqual (round (time_domain_result["device1"]["condition1"]['mean_hr']), 74)
        self.assertEqual (round (time_domain_result["device1"]["condition1"]['rmssd']), 21)

#############################################################################################################################

    @patch('matplotlib.pyplot.show', MagicMock(side_effect=[None]))  # Mocking plt.show() to prevent actual plot display
    @patch('ipywidgets.Output', MagicMock())  # Mocking the Output widget to prevent actual display
    @patch('IPython.display.display', MagicMock())  # Mocking the display function to prevent actual display

    def test_result_comparison_plot(self):
        # Create mock data
        mock_data_chopped = {
            "device1": {
                "condition1": [1000, 1100, 1200, 1300, 1400]
            }
        }
        
        mock_time_domain_features = {
            "device1": {
                "condition1": {
                    "mean_nni": 1150,
                    "sdnn": 50,
                    "sdsd": 10
                }
            }
        }

        mock_frequency_domain_features = {
            "device1": {
                "condition1": {
                    "lf": 0.5,
                    "hf": 0.4,
                    "lf_hf_ratio": 1.25
                }
            }
        }
        
        # Check if the function runs without error
        try:
            individual.result_comparison_plot(
                mock_data_chopped, 
                mock_time_domain_features, 
                mock_frequency_domain_features, 
                ["device1"], 
                ["condition1"]
            )
            plot_works = True
        except:
            plot_works = False
            
        self.assertTrue(plot_works)

#############################################################################################################################

    @patch('matplotlib.pyplot.show', MagicMock(side_effect=[None]))  # Mocking plt.show() to prevent actual plot display
    @patch('IPython.display.display', MagicMock())  # Mocking the display function to prevent actual display
    @patch('ipywidgets.Output', MagicMock())  # Mocking the Output widget to prevent actual display
    def test_unfolding_plot(self):
        # Create mock data_pp dictionary for the test
        mock_data_pp = {
            "device1": {
                "condition1": [1000, 1100, 1200, 1300, 1400]
            }
        }
        
        # Check if the function runs without error
        try:
            individual.unfolding_plot(
                mock_data_pp,
                ["device1"],
                ["condition1"]
            )
            plot_works = True
        except:
            plot_works = False
            
        self.assertTrue(plot_works)

#############################################################################################################################
    
    @patch('matplotlib.pyplot.show', MagicMock(side_effect=[None]))  # Mocking plt.show() to prevent actual plot display
    @patch('IPython.display.display', MagicMock())  # Mocking the display function to prevent actual display
    @patch('ipywidgets.Output', MagicMock())  # Mocking the Output widget to prevent actual display
    def test_bar_plot(self):
        # Create mock time_domain_features and frequency_domain_features dictionaries for the test
        mock_time_domain_features = {
            "device1": {
                "condition1": {"rmssd": 50, "mean_nni": 1000},
                "condition2": {"rmssd": 60, "mean_nni": 1100}
            }
        }

        mock_frequency_domain_features = {
            "device1": {
                "condition1": {"hf": 0.5, "lf": 0.4},
                "condition2": {"hf": 0.6, "lf": 0.5}
            }
        }
        
        # Check if the function runs without error
        try:
            individual.bar_plot(
                mock_time_domain_features,
                mock_frequency_domain_features,
                ["device1"],
                ["condition1", "condition2"]
            )
            plot_works = True
        except:
            plot_works = False
            
        self.assertTrue(plot_works)

#############################################################################################################################

    @patch('matplotlib.pyplot.show', MagicMock(side_effect=[None]))  # Mocking plt.show() to prevent actual plot display
    @patch('IPython.display.display', MagicMock())  # Mocking the display function to prevent actual display
    @patch('ipywidgets.Output', MagicMock())  # Mocking the Output widget to prevent actual display
    def test_line_plot(self):
        # Create mock time_domain_features and frequency_domain_features dictionaries for the test
        mock_time_domain_features = {
            "device1": {
                "condition1": {"rmssd": 50, "mean_nni": 1000},
                "condition2": {"rmssd": 60, "mean_nni": 1100}
            }
        }

        mock_frequency_domain_features = {
            "device1": {
                "condition1": {"hf": 0.5, "lf": 0.4},
                "condition2": {"hf": 0.6, "lf": 0.5}
            }
        }
        
        # Check if the function runs without error
        try:
            individual.line_plot(
                mock_time_domain_features,
                mock_frequency_domain_features,
                ["device1"],
                ["condition1", "condition2"]
            )
            plot_works = True
        except:
            plot_works = False
        
        self.assertTrue(plot_works)

#############################################################################################################################

    @patch('plotly.graph_objs.Figure.show', MagicMock())  # Mocking fig.show() to prevent actual display
    @patch('IPython.display.display', MagicMock())  # Mocking display function to prevent actual display
    def test_radar_plot(self):
        # Create mock time_domain_features for testing
        mock_time_domain_features = {
            "criterion_device": {
                "condition1": {"rmssd": 50, "pnni_50": 20, "mean_hr": 70, "sdnn": 30},
                "condition2": {"rmssd": 55, "pnni_50": 22, "mean_hr": 72, "sdnn": 32}
            },
            "device1": {
                "condition1": {"rmssd": 52, "pnni_50": 21, "mean_hr": 71, "sdnn": 31},
                "condition2": {"rmssd": 56, "pnni_50": 23, "mean_hr": 73, "sdnn": 33}
            }
        }

        # Check if the function runs without any errors
        try:
            individual.radar_plot(
                mock_time_domain_features,
                "criterion_device",
                ["device1"],
                ["condition1", "condition2"]
            )
            plot_works = True
        except Exception as e:
            print(e)
            plot_works = False

        self.assertTrue(plot_works)

#############################################################################################################################

    @patch('IPython.display.display', MagicMock())  # Mocking display function to prevent actual display
    @patch('IPython.display.Markdown', MagicMock())  # Mocking Markdown function to prevent actual display
    def test_display_changes(self):
        # Create mock features for testing
        mock_time_domain_features = {
            "Device1": {
                "Condition1": {"rmssd": 50, "mean_hr": 70},
                "Condition2": {"rmssd": 55, "mean_hr": 75}
            },
            "Device2": {
                "Condition1": {"rmssd": 45, "mean_hr": 65},
                "Condition2": {"rmssd": 50, "mean_hr": 70}
            }
        }
        
        mock_frequency_domain_features = {
            "Device1": {"Condition1": {"hf": 20}, "Condition2": {"hf": 25}},
            "Device2": {"Condition1": {"hf": 15}, "Condition2": {"hf": 20}}
        }
        
        # Check if the function runs without any errors
        try:
            individual.display_changes(
                mock_time_domain_features,
                mock_frequency_domain_features,
                ["Device1", "Device2"],
                ["Condition1", "Condition2"]
            )
            display_works = True
        except Exception as e:
            print(e)
            display_works = False

        self.assertTrue(display_works)

#############################################################################################################################

    @patch('pandas.DataFrame.to_csv', autospec=True)  # Mocking to_csv method to prevent actual file save
    def test_save_data(self, mock_to_csv):
        # Create mock data for testing
        mock_time_domain_features = {
            'Device1': {'Condition1': {'rmssd': 50, 'mean_hr': 70}, 'Condition2': {'rmssd': 55, 'mean_hr': 75}},
            'Device2': {'Condition1': {'rmssd': 45, 'mean_hr': 65}, 'Condition2': {'rmssd': 50, 'mean_hr': 70}}
        }

        mock_frequency_domain_features = {
            'Device1': {'Condition1': {'hf': 20}, 'Condition2': {'hf': 25}},
            'Device2': {'Condition1': {'hf': 15}, 'Condition2': {'hf': 20}}
        }

        mock_artefact = {'Device1': {'Condition1': 0, 'Condition2': 1}, 'Device2': {'Condition1': 2, 'Condition2': 3}}

        # Call the function
        df_all = individual.save_data(
            pp='Participant1',
            path='./',
            time_domain_features=mock_time_domain_features,
            frequency_domain_features=mock_frequency_domain_features,
            data_pp=None,
            devices=['Device1', 'Device2'],
            conditions=['Condition1', 'Condition2'],
            events={'timestamp': [1609459200]},
            artefact=mock_artefact,
            save_as_csv=True
        )

        # Assert that the returned DataFrame contains the expected columns
        expected_columns = ['rmssd', 'mean_hr', 'hf', 'artefact', 'nibi_before_cropping', 'nibi_after_cropping', 'conditions', 'device', 'pp', 'time']
        self.assertListEqual(list(df_all.columns), expected_columns)

        # Assert that to_csv was called if save_as_csv=True
        mock_to_csv.assert_called_once()

#############################################################################################################################

if __name__ == '__main__':
    unittest.main()