#! /usr/bin/env python3  

# *************************************************************************** #
#                                                                             #
#                                                         :::      ::::::::   #
#    describe.py                                        :+:      :+:    :+:   #
#                                                     +:+ +:+         +:+     #
#    By: darodrig <darodrig@42madrid.com>           +#+  +:+       +#+        #
#                                                 +#+#+#+#+#+   +#+           #
#    Created: 2020/11/01 13:22:39 by darodrig          #+#    #+#             #
#    Updated: 2020/11/01 13:22:39 by darodrig         ###   ########.fr       #
#                                                                             #
# *************************************************************************** #

from sys import argv as av
import pandas as pd
from dslr.data import describe_
from dslr.math import percentile_
import numpy as np

if __name__ == "__main__":
    if (len(av) < 2):
        print("USAGE: ./describe.py [dataset].csv")
        exit(1)
    try:
        df = pd.read_csv(av[1])
    except Exception:
        print("No such file")
        exit(1)
    describe_(df)
    exit(0)
