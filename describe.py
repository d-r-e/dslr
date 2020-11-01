#! /usr/bin/env python3

from sys import argv as av
import pandas as pd
import numpy as np

if __name__ == "__main__":
    if (len(av) < 2):
        print("USAGE: ./describe.py [dataset].csv")
        exit(1)
    df = pd.read_csv(av[1])
    print(df.size)