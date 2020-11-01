#! /usr/bin/env python3

from sys import argv as av
import pandas as pd
import numpy as np
import dslr.math as mth

if __name__ == "__main__":
    if (len(av) < 2):
        print("USAGE: ./describe.py [dataset].csv")
        exit(1)
    df = pd.read_csv(av[1])
    cols = df.shape[1]
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics)
    df.dropna(axis=1, how='all',inplace=True)
    df.drop(columns='Index', inplace=True)
    cols = df.shape[1]
    print("",end='\t')
    for col in df.columns:
        print('{:.5}'.format(col), end='\t\t')
    print()
    print("Count:\t", end='')
    for col in df.columns:
        print(f'{mth.count_(df[col].values)}\t', end='\t')
    print()
    print("Mean:\t", end='')
    for col in df.columns:
        m = mth.mean_(df[col].values)
        print('{:f}'.format(m), end='\t')
    print()