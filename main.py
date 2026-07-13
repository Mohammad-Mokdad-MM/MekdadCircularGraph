import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from circular_graph import CircularGraph

BASE_DIR = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize the bundled brain connectivity matrix."
    )
    destination = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Hide connections whose absolute value is at or below this value (default: 0.95).",
    )
    destination.add_argument(
        "--output",
        type=Path,
        help="Write the graph to an image instead of opening the interactive window.",
    )
    destination.add_argument(
        "--interactive",
        action="store_true",
        help="Open the interactive window (the default outside a container).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution of the saved image (default: 150).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.threshold < 0:
        raise SystemExit("--threshold must be zero or greater")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be greater than zero")

    print("Loading data...")
    matrix = pd.read_csv(BASE_DIR / 'surface_native_net_matrix.csv', index_col=0)
    matrix = matrix.values  
    
    labelling = pd.read_csv(BASE_DIR / 'labelling.csv', header=None).values.flatten()
    
    
    names = pd.read_csv(BASE_DIR / 'region_names.csv', header=None).values.flatten()
    
    color_map = pd.read_csv(BASE_DIR / 'color_map.csv', header=None, sep=';').values

    print(f"Original matrix shape: {matrix.shape}")
    print(f"Labelling shape: {labelling.shape}")

    
    if labelling.shape[0] == 124:
        labelling = np.r_[0, 0, labelling[:-1]]   

    if matrix.shape == (123, 123) and labelling.shape[0] == 125:
        padded = np.zeros((125, 125), dtype=matrix.dtype)
        padded[2:, 2:] = matrix           
        matrix = padded

    valid_mask = labelling > 0
    matrix_short = matrix[np.ix_(valid_mask, valid_mask)]            
    labelling_short = labelling[valid_mask].astype(int)             


    perm_matrix = create_permutation_matrix(labelling_short)
    x = perm_matrix.T @ matrix_short @ perm_matrix
    
    
    thresh = args.threshold

    print(f"\nApplying threshold: {thresh}")
    print(f"Values above threshold before: {np.sum(np.abs(x) > thresh)}")
    
    x[np.abs(x) <= thresh] = 0
    
    print(f"Non-zero values after threshold: {np.count_nonzero(x)}")
    print(f"Unique non-zero values: {len(np.unique(x[x != 0]))}")
    
    my_labels = [str(names[i]) for i in range(len(x))]
    
    print("\nCreating circular graph...")
    graph = CircularGraph(
        x,
        colormap=color_map,
        labels=my_labels,
        show=args.output is None,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        graph.fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Graph saved to {args.output}")
    else:
        print("Graph created successfully! Close the window to exit.")


def create_permutation_matrix(v):
    n = len(v)
    m = int(np.max(v))
    M = np.zeros((n, m))
    
    for i in range(n):
        M[i, int(v[i]) - 1] = 1
    
    return M

if __name__ == "__main__":
    main()
