import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

a1 = pd.read_csv(r"D:\Data Science\Machine Leaning\CSV\house_price_regression_dataset.csv")

print(a1.describe())
print(a1.head())
print(a1.info())

