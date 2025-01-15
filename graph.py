"""
0.5 msec/sample	Sample rate =	2000
2 channels
EMG (5 - 250 Hz w/notch)
mV
Clench Force
kg
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('B01_Fatigue.csv')

