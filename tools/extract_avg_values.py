# %%
import pandas as pd
import argparse as arg
import re


def data_frame_from_file(file_name, label_pattern, value_pattern, floatify=True):
    data = {}
    current_label = None
    with open(file_name, "r") as f:
        for line in f:
            label = re.findall(label_pattern, line)
            if len(label) > 0:
                current_label = int(label[0])
                if current_label not in data:
                    data[current_label] = []
            # Check if the line matches the value pattern
            elif re.search(value_pattern, line):
                values = re.findall(value_pattern, line)
                for value in values:
                    if current_label is not None:
                        if floatify:
                            value = float(value)
                        data[current_label].append(value)
                    else:
                        print(value)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "label"
    print("test \n", df)
    return df


# %%
if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument(
        "type",
        type=str,
        help="either: runtime, memory, or energy",
        nargs="?",
        const="runtime",  # default
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        contents = f.read()
    if "tN" in args.input:
        label_pattern = r"Using OMP_NUM_THREADS=(\d+)"
    else:
        label_pattern = r"input_size (\d+) "
    labels = re.findall(label_pattern, contents)
    labels = [int(label) for label in labels]
    print(labels)

    if args.type == "runtime":
        # find all occurrences of On average: followed by a number,
        # with the input length listed between them in the file
        # cf. https://docs.python.org/3/library/re.html#simulating-scanf
        value_pattern = r"On average:\s*(\S+)\s*s"
        result_df = data_frame_from_file(args.input, label_pattern, value_pattern)
    elif args.type == "memory":
        # the results are listed on a single line,
        # and input lengths are listed on some line above
        value_pattern = r"(\d+)$"
        result_df = data_frame_from_file(
            args.input, label_pattern, value_pattern, floatify=False
        )
    elif args.type == "energy":
        value_pattern = r"\|\s*Energy \[J\]\s*\|\s*(\d+\.?\d*)\s*\|"
        result_df = data_frame_from_file(args.input, label_pattern, value_pattern)
        num_repetitions = 1000
        # divide the results by the number of repetitions
        result_df = result_df / num_repetitions
    elif args.type == "energy_both":
        value_pattern = r"\|\s*Energy \[J\]\s*\|\s*(\d+\.?\d*)\s*\|"
        result_df = data_frame_from_file(args.input, label_pattern, value_pattern)
        value_pattern_dram = r"\|\s*Energy DRAM \[J\]\s*\|\s*(\d+\.?\d*)\s*\|"
        result_df_dram = data_frame_from_file(
            args.input, label_pattern, value_pattern_dram
        )
        num_repetitions = 1000
        [
            print("ratio", float(result_dram) / float(result))
            for result, result_dram in zip(result_df, result_df_dram)
        ]
        assert False
    elif args.type == "flops":
        value_pattern = r"\|\s*SP \[MFLOP/s\]\s*\|\s*(\d+\.?\d*)\s*\|"
        result_df = data_frame_from_file(args.input, label_pattern, value_pattern)
    elif args.type == "intensity":
        value_pattern = (
            r"\|\s*Operational intensity \[FLOP/Byte\]\s*\|\s*(\d+\.?\d*)\s*\|"
        )
        result_df = data_frame_from_file(args.input, label_pattern, value_pattern)

    ## see the data divided by the input sizes
    # pd.set_option("display.precision", 10)
    # print(result_df.div(result_df.index.values, axis=0))
    
    # if the sizes don't match, find the failed runs and add this to the output file:
    """
    On average: 0.0 s
    000000
    | Energy [J] | 0.0 |
    | SP [MFLOP/s] | 0.0 |
    | Operational intensity [FLOP/Byte] | 0.0 |
    """
    if "SubgridLES" in args.input:
        columns = [
            "input_size",
            "subgridles-intel-open-cpp",
            "subgridles-intel-open-fortran",
            "subgridles-intel-open-libtorch",
            "subgridles-intel-mkl-cpp",
            "subgridles-intel-mkl-fortran",
            "subgridles-intel-mkl-pytorch",
        ]
    elif "DeepRLEddy" in args.input:
        columns = [
            "input_size",
            "deeprleddy-intel-open-dnnl",
            "deeprleddy-intel-open-fortran",
            "deeprleddy-intel-open-libtorch",
            "deeprleddy-intel-mkl-dnnl",
            "deeprleddy-intel-mkl-fortran",
            "deeprleddy-intel-mkl-pytorch",
        ]
    if args.type == "memory":
        # no memory measurements for pytorch
        columns = columns[:-1]
    result_df.columns = columns[1:]
    result_df.index.name = columns[0]
    print(result_df)
    result_df.to_csv(args.input[:-4] + ".csv")

    try:
        import subprocess

        # execute latexmk to generate a pdf plot of the data
        subprocess.call(["latexmk", "-pdf", "scalingplot.tex"])
    except FileNotFoundError as e:
        print(
            "Could not generate plot with latexmk, "
            + "check if it is installed and scalingplot.tex exists"
        )
        pass
