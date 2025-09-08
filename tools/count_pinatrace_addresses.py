# %%
import pandas as pd
import argparse as arg
from os.path import abspath

# %%
if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument('input', type=str, help='Input file')
    args = parser.parse_args()
    df = pd.read_csv(args.input, sep=' ', on_bad_lines='warn')
    columns=['requesting_address', 'R/W', 'address']
    # set header
    df.columns = columns
    absolute_path = abspath(args.input)
    print("Number of unique addresses in {}: ".format(absolute_path))
    print(df['address'].nunique())
