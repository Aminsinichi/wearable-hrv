import sys
sys.path.append(r"C:\Users\msi401\OneDrive - Vrije Universiteit Amsterdam\PhD\Data\Coding\Validation Study\wearable-hrv")
import unittest
from wearablehrv import individual
import pandas as pd
import os
import time
import tkinter as tk
from unittest.mock import patch
import tempfile

path = os.path.dirname(os.path.abspath(__file__)) + "/" 

class TestIndividual(unittest.TestCase):
    
    def setUp(self):
        # Define the setup variables as provided
        self.path = path  # Adjusted path to match your directory structure
        self.pp = "test"
        self.conditions = ['sitting', 'arithmetic', 'recovery', 'standing', 'breathing', 
                           'neurotask', 'walking', 'biking']

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

if __name__ == '__main__':
    unittest.main()