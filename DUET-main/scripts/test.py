import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-set-name",
    type=str,
    nargs="+",
    default="ETTh1.csv",
    help="List of dataset names"
)

# args = parser.parse_args(["--data-set-name", "test1", "test2"])
args = parser.parse_args()

print(f"All parsed arguments: {args}")
print(f"Value of args.data_set_name: {args.data_set_name}")
print(f"Type of args.data_set_name: {type(args.data_set_name)}")

if args.data_set_name is None:
    print("data_set_name was not provided, using default (None).")
elif isinstance(args.data_set_name, list):
    print(f"Datasets provided: {', '.join(args.data_set_name)}")
else:
    print("data_set_name has an unexpected type or value.")