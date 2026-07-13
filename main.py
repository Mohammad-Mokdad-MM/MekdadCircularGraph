import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

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
    destination.add_argument(
        "--web",
        action="store_true",
        help="Serve the interactive graph in a web browser.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MEKDAD_WEB_HOST", "127.0.0.1"),
        help="Address for web mode to listen on (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MEKDAD_WEB_PORT", "8000")),
        help="Port for web mode (default: 8000).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution of the saved image (default: 150).",
    )
    args = parser.parse_args()
    if (
        args.output is None
        and not args.interactive
        and not args.web
        and os.environ.get("MEKDAD_DEFAULT_MODE") == "web"
    ):
        args.web = True
    return args


def configure_matplotlib(args):
    import matplotlib

    if args.output is not None:
        matplotlib.use("Agg")
    elif args.web:
        matplotlib.use("WebAgg")
        matplotlib.rcParams["webagg.address"] = args.host
        matplotlib.rcParams["webagg.port"] = args.port
        matplotlib.rcParams["webagg.port_retries"] = 1
        matplotlib.rcParams["webagg.open_in_browser"] = False


def main():
    args = parse_args()

    if args.threshold < 0:
        raise SystemExit("--threshold must be zero or greater")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be greater than zero")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")

    configure_matplotlib(args)
    from circular_graph import CircularGraph

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
    if args.web:
        browser_host = "localhost" if args.host in {"0.0.0.0", "127.0.0.1"} else args.host
        print(f"Open http://{browser_host}:{args.port} in your browser.", flush=True)
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
    elif args.web:
        print("Web server stopped.")
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
