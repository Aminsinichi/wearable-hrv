import sys
sys.path.append(r"C:\Users\msi401\OneDrive - Vrije Universiteit Amsterdam\PhD\Data\Coding\Validation Study\wearable-hrv")
import unittest
from wearablehrv import group
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


#############################################################################################################################
    def test_signal_quality(self):
        # Prepare a sample data dictionary, which should resemble your actual data structure
        sample_data, file_names = group.import_data (path, self.conditions, self.devices, self.features)
        
        # Execute the function
        new_data, new_features = group.signal_quality(
            deepcopy(sample_data), self.path, self.conditions, self.devices, 
            self.features, self.criterion, file_names, exclude = True
        )
        
        # Test if the function returns a tuple of (dict, list)
        self.assertIsInstance(new_data, dict)
        self.assertIsInstance(new_features, list)

        # Test if 'artefact' and 'nibi_after_cropping' are removed from the features
        self.assertNotIn('artefact', new_features)
        self.assertNotIn('nibi_after_cropping', new_features)

        # Test if the data dictionary is updated correctly (this will depend on your thresholds and conditions)
        self.assertEqual(new_data["vu"]["nibi_after_cropping"]["sitting"]["P01.csv"], [0.9])  

        # Test for messages (this will require patching the print function)
        with patch('builtins.print') as mocked_print:
            group.signal_quality(
                deepcopy(sample_data), self.path, self.conditions, self.devices,
                self.features, self.criterion, ["P01.csv", "P02.csv"]
            )
            mocked_print.assert_any_call('Data Saved Succesfully!')






if __name__ == '__main__':
    unittest.main()




    