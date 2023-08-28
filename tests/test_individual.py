import sys
sys.path.append(r"C:\Users\msi401\OneDrive - Vrije Universiteit Amsterdam\PhD\Data\Coding\Validation Study\wearable-hrv")
import unittest
from wearablehrv import individual
import pandas as pd
import os
import time
import tkinter as tk
from unittest.mock import patch, MagicMock
import tempfile

path = os.path.dirname(os.path.abspath(__file__)) + "/" 

class TestIndividual(unittest.TestCase):
    
    def setUp(self):
        # Define the setup variables as provided
        self.path = path  # Adjusted path to match your directory structure
        self.pp = "test"
        self.conditions = ['sitting', 'arithmetic', 'recovery', 'standing', 'breathing', 
                           'neurotask', 'walking', 'biking']
        self.devices = ["vu", "empatica", "heartmath", "kyto", "rhythm"]

#############################################################################################################################

    def test_define_events_read(self):
        events = individual.define_events(self.path, self.pp, self.conditions, already_saved=True, save_as_csv=False)
        
        # Assert that the output is a DataFrame
        self.assertIsInstance(events, pd.DataFrame)
        
        # Assert that the DataFrame has the expected columns
        expected_columns = ['timestamp', 'conditions', 'datapoint']
        self.assertListEqual(list(events.columns), expected_columns)
        
        # Assert that the conditions in the DataFrame match the expected conditions
        unique_conditions = events['conditions'].unique()
        self.assertListEqual(sorted(unique_conditions), sorted(self.conditions))

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

#############################################################################################################################




if __name__ == '__main__':
    unittest.main()