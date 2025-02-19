# %%
import pandas as pd
import argparse as arg
import re

# %%
if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument('input', type=str, help='Input file')
    args = parser.parse_args()
    
    # find all occurrences of On average: followed by a number,
    # with the input length listed between them in the file

    with open(args.input, 'r') as f:
        contents = f.read()
    # cf. https://docs.python.org/3/library/re.html#simulating-scanf
    results = re.findall(r'On average:\s*(\S+)\s*s', contents)
    results = [float(result) for result in results]
    print(results)
    print(len(results))
    input_sizes = re.findall(r'input_size (\d+)', contents)
    input_sizes = [int(size) for size in input_sizes]
    print(input_sizes)
    assert len(results)%len(input_sizes) == 0
    num_variants = len(results)//len(input_sizes)
    print(num_variants)
    assert num_variants == 6
    columns = ['input_size', 'deeprl_avg_dnnl', 'deeprl_avg_libtorch', 'deeprl_avg_pytorch_jit',
               'subgridles_avg_libtorch', 'subgridles_avg_fortran', 'subgridles_avg_pytorch_jit']
    data = []
    for i in range(len(input_sizes)):
        data.append([input_sizes[i]] + results[i*num_variants:(i+1)*num_variants])
    df = pd.DataFrame(data, columns=columns)
    print(df)
    df.to_csv(args.input[:-4] + ".csv", index=None)

