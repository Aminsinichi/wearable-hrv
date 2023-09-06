import pdb
import sys
sys.path.append(r"C:\Users\msi401\OneDrive - Vrije Universiteit Amsterdam\PhD\Data\Coding\Validation Study\wearable-hrv")
import unittest
from wearablehrv import group
import pandas as pd
import pingouin as pg
import os
import time
import pickle
import tkinter as tk
import hrvanalysis
from unittest.mock import patch, MagicMock, PropertyMock
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import seaborn as sns
from ipywidgets import Dropdown, IntText, Output, HBox
from IPython.display import display, clear_output
from copy import deepcopy

path = os.path.dirname(os.path.abspath(__file__)) + "/" 

class TestGroup(unittest.TestCase):
    
    def setUp(self):
        # Define the setup variables 
        self.path = path  
        self.pp = "test"
        self.criterion = "vu"
        self.conditions = ['sitting', 'arithmetic', 'recovery', 'standing', 'breathing', 
                            'neurotask', 'walking', 'biking']
        self.devices = ["vu", "empatica", "heartmath", "kyto", "rhythm"]
        self.features = ["rmssd", "hf",'pnni_50','mean_hr','sdnn', 'nibi_after_cropping', 'artefact']

#############################################################################################################################
    
    def test_import_data(self):
        # Execute the function
        data, file_names = group.import_data(self.path, self.conditions, self.devices, self.features)
        
        # Test if the function returns a dictionary
        self.assertIsInstance(data, dict)
        
        # Test for specific keys
        self.assertIn("vu", data.keys())
        self.assertIn("rmssd", data["vu"].keys())
        self.assertIn("sitting", data["vu"]["rmssd"].keys())
        
        # Test for participant files
        self.assertIn("P01.csv", file_names)
        self.assertIn("P10.csv", file_names)
        
        # Test the data integrity
        self.assertEqual(data["rhythm"]["rmssd"]["sitting"]["P01.csv"], [55.21269525925547])
        self.assertEqual(data["vu"]["mean_hr"]["breathing"]["P10.csv"], [80.14314824809138])
        
        # Test for messages (this will require patching the print function)
        with patch('builtins.print') as mocked_print:
            group.import_data(self.path, self.conditions, self.devices, self.features)
            mocked_print.assert_any_call('These are the detected .csv files: [\'P01.csv\', \'P02.csv\', \'P03.csv\', \'P04.csv\', \'P05.csv\', \'P06.csv\', \'P07.csv\', \'P08.csv\', \'P09.csv\', \'P10.csv\']')
            mocked_print.assert_any_call('Group dataset imported succesfully')

#############################################################################################################################

    def test_nan_handling(self):
        # Create a sample dictionary that mimics the structure of the 'data' argument
        sample_data = {
            "device1": {
                "feature1": {
                    "condition1": {
                        "P01.csv": [1, 2, 3],
                        "P02.csv": [np.nan]
                    },
                    "condition2": {
                        "P01.csv": [4, 5, 6],
                        "P02.csv": [7, 8, 9]
                    }
                }
            }
        }
        
        # Execute the function
        new_data = group.nan_handling(deepcopy(sample_data), ["device1"], ["feature1"], ["condition1", "condition2"])
        
        # Test if the function returns a dictionary
        self.assertIsInstance(new_data, dict)
        
        # Test NaN replacement
        self.assertEqual(new_data["device1"]["feature1"]["condition1"]["P02.csv"], [])
        
        # Test unchanged values
        self.assertEqual(new_data["device1"]["feature1"]["condition2"]["P02.csv"], [7, 8, 9])
        
        # Test for messages (this will require patching the print function)
        with patch('builtins.print') as mocked_print:
            group.nan_handling(deepcopy(sample_data), ["device1"], ["feature1"], ["condition1", "condition2"])
            mocked_print.assert_any_call('NaN values are removed from the data')


#############################################################################################################################

    def test_save_data(self):
        # Create a sample dictionary that mimics the structure of the 'data' argument
        sample_data = {
            "device1": {
                "feature1": {
                    "condition1": {
                        "P01.csv": [1],
                        "P02.csv": [2]
                    }
                }
            }
        }
        
        # Execute the function
        test_save_path = path
        test_file_path = test_save_path + "group_data.csv"
        group.save_data(sample_data, test_save_path, ["condition1"], ["device1"], ["feature1"], ["P01.csv", "P02.csv"])
        
        # Test if the CSV file is created
        self.assertTrue(os.path.exists(test_file_path))
        
        # Test the content of the CSV file
        saved_data = pd.read_csv(test_file_path)
        self.assertEqual(saved_data["device1_feature1_condition1"].tolist(), [1, 2])
        
        # Test for messages (this will require patching the print function)
        with patch('builtins.print') as mocked_print:
            group.save_data(sample_data, test_save_path, ["condition1"], ["device1"], ["feature1"], ["P01.csv", "P02.csv"])
            mocked_print.assert_any_call('Data Saved Succesfully!')
        
        # Remove the test CSV file
        os.remove(test_file_path)


##############################################################################################################################
    
    def test_signal_quality(self):
        # Loading the original dataset
        sample_data, file_names = group.import_data (path, self.conditions, self.devices, self.features)
        
        # Execute the function
        new_data, new_features = group.signal_quality(
            deepcopy(sample_data), self.path, self.conditions, self.devices, 
            self.features, self.criterion, file_names, exclude = True
        )
        
        # Test if the function returns a tuple of (dict, list)
        self.assertIsInstance(new_data, dict)
        self.assertIsInstance(new_features, list)

        # Test if 'artefact' and 'nibi_after_cropping' are removed from the features and the data
        self.assertNotIn('artefact', new_features)
        self.assertNotIn('nibi_after_cropping', new_features)

        self.assertNotIn('nibi_after_cropping', new_data["vu"])
        self.assertNotIn('artefact', new_data["vu"])

        # Test if the data dictionary is updated correctly 
        self.assertEqual(sample_data['rhythm']['rmssd']['standing']['P05.csv'], [48.39451400218052]) # before excluding
        self.assertEqual(new_data['rhythm']['rmssd']['standing']['P05.csv'], []) # after excluding 
 
        # Test if the CSV file is created
        test_file_1 = path + "quality_report1.csv"
        test_file_2 = path + "quality_report2.csv"
        self.assertTrue(os.path.exists(test_file_1))
        self.assertTrue(os.path.exists(test_file_2))

        # Test the content of the CSV file
        saved_data_1 = pd.read_csv(test_file_1)
        saved_data_2 = pd.read_csv(test_file_2)
        self.assertEqual(saved_data_1["Decision"].tolist()[81], 'Missing')
        self.assertEqual(saved_data_2["Acceptable"].tolist()[0:4], [0, 0, 4, 3])

        # Remove the test CSV file
        os.remove(test_file_1)
        os.remove(test_file_2)

##############################################################################################################################
  
    def test_violin_plot(self):

        # Create mock data dictionary for the test
        mock_data, file_names = group.import_data(path, self.conditions, self.devices, self.features)

        with patch('plotly.graph_objects.Figure.show', MagicMock()):  # Mocking fig.show() to prevent actual plot display
            with patch('IPython.display.display', MagicMock()):  # Mocking the display function to prevent actual display
                with patch('ipywidgets.Output', MagicMock()):  # Mocking the Output widget to prevent actual display
                    try:
                        group.violin_plot(
                            mock_data,
                            self.conditions,
                            self.features,
                            self.devices
                        )
                        plot_works = True
                    except:
                        plot_works = False

        self.assertTrue(plot_works)

##############################################################################################################################

    def test_box_plot(self):

        # Create mock data dictionary for the test
        mock_data, file_names = group.import_data(self.path, self.conditions, self.devices, self.features)

        with patch('plotly.graph_objects.Figure.show', MagicMock()):  # Mocking fig.show() to prevent actual plot display
            with patch('IPython.display.display', MagicMock()):  # Mocking the display function to prevent actual display
                with patch('ipywidgets.Output', MagicMock()):  # Mocking the Output widget to prevent actual display
                    try:
                        group.box_plot(
                            mock_data,
                            self.conditions,
                            self.features,
                            self.devices
                        )
                        plot_works = True
                    except:
                        plot_works = False

        self.assertTrue(plot_works)

##############################################################################################################################
    
    def test_radar_plot(self):

        # Create mock data dictionary for the test
        mock_data, file_names = group.import_data(self.path, self.conditions, self.devices, self.features)

        with patch('plotly.graph_objects.Figure.show', MagicMock()):  # Mocking fig.show() to prevent actual plot display
            with patch('IPython.display.display', MagicMock()):  # Mocking the display function to prevent actual display
                with patch('ipywidgets.Output', MagicMock()):  # Mocking the Output widget to prevent actual display
                    try:
                        group.radar_plot(
                            mock_data,
                            self.criterion,
                            self.conditions,
                            self.features,
                            self.devices
                        )
                        plot_works = True
                    except:
                        plot_works = False

        self.assertTrue(plot_works)

##############################################################################################################################

    def test_hist_plot(self):

        # Create mock data dictionary for the test
        mock_data, file_names = group.import_data(self.path, self.conditions, self.devices, self.features)

        with patch('plotly.graph_objects.Figure.show', MagicMock()):  # Mocking fig.show() to prevent actual plot display
            with patch('IPython.display.display', MagicMock()):  # Mocking the display function to prevent actual display
                with patch('ipywidgets.Output', MagicMock()):  # Mocking the Output widget to prevent actual display
                    try:
                        group.hist_plot(
                            mock_data,
                            self.conditions,
                            self.features,
                            self.devices
                        )
                        plot_works = True
                    except:
                        plot_works = False

        self.assertTrue(plot_works)

##############################################################################################################################
    
    def test_matrix_plot(self):

        # Create mock data dictionary for the test
        mock_data, file_names = group.import_data(self.path, self.conditions, self.devices, self.features)
        mock_data = group.nan_handling (mock_data, self.devices, self.features, self.conditions) # This function is sensitive to NaN values, that's why I am first taking care of them


        with patch.object(plt, 'show', MagicMock()):  # Mocking plt.show() to prevent actual plot display
            with patch('IPython.display.display', MagicMock()):  # Mocking the display function to prevent actual display
                with patch('ipywidgets.Output', MagicMock()):  # Mocking the Output widget to prevent actual display
                    try:
                        group.matrix_plot(
                            mock_data,
                            file_names,
                            self.conditions,
                            self.features,
                            self.devices
                        )
                        plot_works = True
                    except:
                        plot_works = False

        self.assertTrue(plot_works)

##############################################################################################################################

    def test_density_plot(self):

        # Create mock data dictionary for the test
        mock_data, file_names = group.import_data(self.path, self.conditions, self.devices, self.features)

        with patch.object(plt, 'show', MagicMock()):  # Mocking plt.show() to prevent actual plot display
            with patch.object(sns, 'kdeplot', MagicMock()):  # Mocking sns.kdeplot to prevent actual plot display
                with patch('IPython.display.display', MagicMock()):  # Mocking the display function to prevent actual display
                    with patch('ipywidgets.Output', MagicMock()):  # Mocking the Output widget to prevent actual display
                        try:
                            group.density_plot(
                                mock_data,
                                file_names,
                                self.conditions,
                                self.features,
                                self.devices
                            )
                            plot_works = True
                        except:
                            plot_works = False

        self.assertTrue(plot_works)

##############################################################################################################################

    def test_bar_plot(self):

        # Create mock data dictionary for the test
        mock_data, file_names = group.import_data(self.path, self.conditions, self.devices, self.features)

        with patch('plotly.graph_objects.Figure.show', MagicMock()):  # Mocking fig.show() to prevent actual plot display
            with patch('IPython.display.display', MagicMock()):  # Mocking the display function to prevent actual display
                with patch('ipywidgets.Output', MagicMock()):  # Mocking the Output widget to prevent actual display
                    try:
                        group.bar_plot(
                            mock_data,
                            self.conditions,
                            self.features,
                            self.devices
                        )
                        plot_works = True
                    except:
                        plot_works = False

        self.assertTrue(plot_works)

##############################################################################################################################

    def test_regression_analysis(self):
        # Create mock data dictionary for the test
        mock_data, file_names = group.import_data(self.path, self.conditions, self.devices, self.features)
        mock_data = group.nan_handling (mock_data, self.devices, self.features, self.conditions) # This function is sensitive to NaN values, that's why I am first taking care of them
        new_features = ["rmssd", "hf",'pnni_50','mean_hr','sdnn']

        # Run the function
        regression_data = group.regression_analysis(mock_data, self.criterion, self.conditions, self.devices, new_features, self.path, save_as_csv=True)

        # Check that the regression_data dictionary has the expected shape and keys
        self.assertIn("vu", regression_data)
        self.assertIn("rmssd", regression_data["vu"])
        self.assertIn("sitting", regression_data["vu"]["rmssd"])
        self.assertIn("slope", regression_data["vu"]["rmssd"]["sitting"])

        # Test the data integrity (this is tested against Jamovi 2.3.21, Regression Analysis)
        self.assertEqual(round (regression_data["heartmath"]["rmssd"]["breathing"]["slope"], 2), [1.21])
        self.assertEqual(round (regression_data["heartmath"]["rmssd"]["breathing"]["intercept"], 2), [-14.46])
        self.assertEqual(round (regression_data["heartmath"]["rmssd"]["breathing"]["r_value"], 2), [0.95])
        self.assertLess(regression_data["heartmath"]["rmssd"]["breathing"]["p_value"], 0.001)
        self.assertEqual(round (regression_data["heartmath"]["rmssd"]["breathing"]["std_err"], 2), [0.13])

        # Test if the CSV file is created
        test_file_regression = path + "regression_data.csv"
        self.assertTrue(os.path.exists(test_file_regression))
        

        # Test the content of the CSV file
        saved_data_regression = pd.read_csv(test_file_regression)
        self.assertEqual(saved_data_regression["Intercept"].tolist()[58], 3.105166301568409)

        # Remove the test CSV file
        os.remove(test_file_regression)

##############################################################################################################################
    
    def test_regression_plot(self):

        # Create mock data dictionary for the test
        mock_data, file_names = group.import_data(self.path, self.conditions, self.devices, self.features)
        mock_data = group.nan_handling (mock_data, self.devices, self.features, self.conditions) # This function is sensitive to NaN values, that's why I am first taking care of them
        new_features = ["rmssd", "hf",'pnni_50','mean_hr','sdnn']
        regression_data = group.regression_analysis(mock_data, self.criterion, self.conditions, self.devices, new_features, self.path, save_as_csv=False) # Assuming you have this function

        with patch.object(plt, 'show', MagicMock()):  # Mocking plt.show() to prevent actual plot display
            with patch.object(plt, 'scatter', MagicMock()):  # Mocking plt.scatter to prevent actual plot display
                with patch.object(plt, 'plot', MagicMock()):  # Mocking plt.plot to prevent actual plot display
                    with patch('IPython.display.display', MagicMock()):  # Mocking the display function to prevent actual display
                        with patch('ipywidgets.Output', MagicMock()):  # Mocking the Output widget to prevent actual display
                            try:
                                group.regression_plot(
                                    regression_data,
                                    mock_data,
                                    self.criterion,
                                    self.conditions,
                                    self.devices,
                                    self.features
                                )
                                plot_works = True
                            except:
                                plot_works = False

        self.assertTrue(plot_works)


##############################################################################################################################

    def test_icc_analysis(self):
        
        # Create mock data dictionary for the test
        mock_data, file_names = group.import_data(self.path, self.conditions, self.devices, self.features)
        mock_data = group.nan_handling (mock_data, self.devices, self.features, self.conditions) # This function is sensitive to NaN values, that's why I am first taking care of them
        new_features = ["rmssd", "hf",'pnni_50','mean_hr','sdnn']
        icc_data = group.icc_analysis(mock_data, self.criterion, self.devices, self.conditions, new_features, self.path, save_as_csv=True)

        # Verify that the function output matches expected values (These are tested against SPSS 28.0.0.0)
        # In SPSS > Analyze > Scale > Reliability Analysis > Items: heartmath_rmssd_sitting, vu_rmssd_sitting > Statistics: ICC is ticked > ModeL: Two-Way Mixed, Type: Absolute Agreement

        self.assertEqual(round(icc_data['heartmath']['rmssd']['sitting']['ICC'][1], 3), 0.664)
        self.assertEqual(round(icc_data['heartmath']['rmssd']['sitting']['df1'][1], 3), 9)
        self.assertEqual(round(icc_data['heartmath']['rmssd']['sitting']['pval'][1], 3), 0.002)
        self.assertEqual(icc_data['heartmath']['rmssd']['sitting']['CI95%'][1][0], 0.02)
        self.assertEqual(icc_data['heartmath']['rmssd']['sitting']['CI95%'][1][1], 0.91)

        self.assertEqual(round(icc_data['heartmath']['rmssd']['sitting']['ICC'][4], 3), 0.798)
        self.assertEqual(round(icc_data['heartmath']['rmssd']['sitting']['df1'][4], 3), 9)
        self.assertEqual(round(icc_data['heartmath']['rmssd']['sitting']['pval'][4], 3), 0.002)
        self.assertEqual(icc_data['heartmath']['rmssd']['sitting']['CI95%'][4][0], 0.04)
        self.assertEqual(icc_data['heartmath']['rmssd']['sitting']['CI95%'][4][1], 0.95)

        # this works very well even when there are excluding cases (in terms of Kyto for instance, 4 excluded cases):
        self.assertEqual(round(icc_data['kyto']['rmssd']['sitting']['ICC'][1], 3), 0.84)
        self.assertEqual(round(icc_data['kyto']['rmssd']['sitting']['df1'][1], 3), 5)
        self.assertLess(round(icc_data['kyto']['rmssd']['sitting']['pval'][1], 3),  0.001)
        self.assertEqual(icc_data['kyto']['rmssd']['sitting']['CI95%'][1][0], -0.06)
        self.assertEqual(icc_data['kyto']['rmssd']['sitting']['CI95%'][1][1], 0.98)

        self.assertEqual(round(icc_data['kyto']['rmssd']['sitting']['ICC'][4], 3), 0.913)
        self.assertEqual(round(icc_data['kyto']['rmssd']['sitting']['df1'][4], 3), 5)
        self.assertLess(round(icc_data['kyto']['rmssd']['sitting']['pval'][4], 3),  0.001)
        self.assertEqual(icc_data['kyto']['rmssd']['sitting']['CI95%'][4][0], -0.12)
        self.assertEqual(icc_data['kyto']['rmssd']['sitting']['CI95%'][4][1], 0.99)

        # Check if the CSV file is created
        test_file_icc = path + "icc_data.csv"
        self.assertTrue(os.path.exists(test_file_icc))

        # Test the content of the CSV file 
        saved_data_icc = pd.read_csv(test_file_icc)
        self.assertEqual(saved_data_icc["p-value"].tolist()[774], 0.0818837510509474)
            
        # Remove the test CSV file
        os.remove(test_file_icc)

##############################################################################################################################

def test_icc_plot(self):
    mock_data, file_names = group.import_data(self.path, self.conditions, self.devices, self.features)
    mock_data = group.nan_handling(mock_data, self.devices, self.features, self.conditions)  # This function is sensitive to NaN values, that's why I am first taking care of them
    new_features = ["rmssd", "hf", 'pnni_50', 'mean_hr', 'sdnn']
    mock_icc_data = group.icc_analysis(mock_data, self.criterion, self.devices, self.conditions, new_features, self.path, save_as_csv=False)
    
    with patch.object(sns, 'heatmap', MagicMock()):  # Mock sns.heatmap to prevent actual plot display
        with patch.object(plt, 'show', MagicMock()):  # Mock plt.show() to prevent actual plot display
            with patch.object(plt, 'figure', MagicMock()):  # Mock plt.figure() to prevent actual plot display
                with patch('ipywidgets.Output', MagicMock()):  # Mock the Output widget to prevent actual display
                    with patch('IPython.display.display', MagicMock()):  # Mock the display function to prevent actual display
                        try:
                            group.icc_plot(mock_icc_data, self.conditions, self.devices, new_features)
                            plot_works = True
                        except Exception as e:
                            print(f"An error occurred: {e}")
                            plot_works = False

    self.assertTrue(plot_works)

##############################################################################################################################





if __name__ == '__main__':
    unittest.main()