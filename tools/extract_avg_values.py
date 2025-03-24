# %%
import pandas as pd
import argparse as arg
import re

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
    input_sizes = re.findall(r"input_size (\d+)", contents)
    input_sizes = [int(size) for size in input_sizes]
    print(input_sizes)

    # later: manually map to column names:
    # input-size, deeprleddy-gcc-open-dnnl, deeprleddy-gcc-open-libtorch, deeprleddy-gcc-open-pytorch,
    # subgridles-gcc-open-cpp, subgridles-gcc-open-fortran,
    # deeprleddy-gcc-mkl-dnnl, subgridles-gcc-open-libtorch, subgridles-gcc-open-pytorch, 
    # subgridles-gcc-mkl-cpp,subgridles-gcc-mkl-fortran
    if args.type == "runtime":
        # find all occurrences of On average: followed by a number,
        # with the input length listed between them in the file
        # cf. https://docs.python.org/3/library/re.html#simulating-scanf
        results = re.findall(r"On average:\s*(\S+)\s*s", contents)
        results = [result[:-1] if result[-1] == "s" else result for result in results]
        results = [float(result) for result in results]
    elif args.type == "memory":
        # the results are listed on a single line,
        # and input lengths are listed on some line above
        results = re.findall(r"\n(\d+)", contents)
        results = [int(result) for result in results]
    elif args.type == "energy":
        results = re.findall(r"\|\s*Energy \[J\]\s*\|\s*(\d+\.?\d*)\s*\|", contents)
        num_repetitions = 1000
        results = [float(result) / num_repetitions for result in results]
    elif args.type == "flops":
        results = re.findall(r"\|\s*SP \[MFLOP/s\]\s*\|\s*(\d+\.?\d*)\s*\|", contents)
        results = [float(result) for result in results]
    elif args.type == "intensity":
        results = re.findall(r"\|\s*Operational intensity \[FLOP/Byte\]\s*\|\s*(\d+\.?\d*)\s*\|", contents)
        results = [float(result) for result in results]
    assert len(results) > 0    
    print(len(input_sizes), len(results), results)

    assert len(results) % len(input_sizes) == 0
    # if the sizes don't match, find the failed runs and add this to the output file:
    """
    On average: 0.0 s
    | Energy [J] | 0.0 |
    | SP [MFLOP/s] | 0.0 |
    | Operational intensity [FLOP/Byte] | 0.0 |
    """

    num_variants = len(results) // len(input_sizes)
    columns = ["input_size"] + [str(i + 1) for i in range(num_variants)]
    data = []
    for i in range(len(input_sizes)):
        data.append(
            [input_sizes[i]] + results[i * num_variants : (i + 1) * num_variants]
        )
    df = pd.DataFrame(data, columns=columns)
    print(df)
    df.to_csv(args.input[:-4] + ".csv", index=None)

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
