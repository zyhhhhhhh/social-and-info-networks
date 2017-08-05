from csv import reader
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import csv
import pandas as pd
with open('market-clearing.csv', "r") as f:
    df = csv.reader(f)
    data = list(df)
data = np.array(data)
# print(data)
print(np.asarray(data[1:,2],dtype = float))
a = np.asarray(data[1:,2],dtype = float).argmax()
print(a)