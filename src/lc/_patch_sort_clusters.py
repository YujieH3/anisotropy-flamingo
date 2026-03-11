# This script sort the csv sample files according to Lcore/Ltot values.
# It is internally called by batch_prep.sh

import pandas as pd
import numpy as np
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Calculate significance map for best fit scans.")

# Add arguments
parser.add_argument('-i', '--input', type=str, help='Input file')
parser.add_argument('-o', '--output', type=str, help='Output file')

# Parse the arguments
args = parser.parse_args()
INPUT_FILE  = args.input
OUTPUT_FILE = args.output

df = pd.read_csv(INPUT_FILE)
df.sort_values('3DLcore/Ltot', ascending=False, inplace=True)
df.to_csv(OUTPUT_FILE, index=False)
