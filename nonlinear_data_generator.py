import argparse
import csv
import numpy as np

def nonlinear_data_generator(coefs, offset, rnge, N, scale, seed):
    """
    Generate non-linear data suitable for Gradient Boosting testing.
    :param coefs: List of coefficients for the polynomial (e.g., [a, b, c] for ax^2 + bx + c).
    :param offset: Constant offset for the target variable.
    :param rnge: Range of input features as a tuple (min, max).
    :param N: Number of samples to generate.
    :param scale: Scale of the added noise.
    :param seed: Seed for random number generator.
    :return: Tuple (X, y) with features and targets.
    """
    rng = np.random.default_rng(seed=seed)
    X = rng.uniform(low=rnge[0], high=rnge[1], size=(N, len(coefs) - 1))
    
    # Compute the polynomial relationship
    y = np.zeros((N, 1))
    for i, coef in enumerate(coefs):
        y += coef * np.power(X, i).sum(axis=1, keepdims=True)
    
    # Add offset and noise
    y += offset
    noise = rng.normal(loc=0.0, scale=scale, size=y.shape)
    return X, y + noise

def write_data(filename, X, y):
    """
    Write generated data to a CSV file.
    :param filename: Path to the output file.
    :param X: Input features.
    :param y: Target variable.
    """
    with open(filename, "w") as file:
        header = [f"x_{n}" for n in range(X.shape[1])] + ["y"]
        writer = csv.writer(file)
        writer.writerow(header)
        for row in np.hstack((X, y)):
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, help="Number of samples.")
    parser.add_argument("-coefs", nargs='*', type=float, help="Coefficients for polynomial regression (e.g., a b c for ax^2 + bx + c).")
    parser.add_argument("-offset", type=float, help="Offset for the target variable.")
    parser.add_argument("-scale", type=float, help="Scale of the noise.")
    parser.add_argument("-rnge", nargs=2, type=float, help="Range of X values.")
    parser.add_argument("-seed", type=int, help="Seed for reproducibility.")
    parser.add_argument("-output_file", type=str, help="Path to the output CSV file.")
    args = parser.parse_args()

    coefs = np.array(args.coefs)
    X, y = nonlinear_data_generator(coefs, args.offset, args.rnge, args.N, args.scale, args.seed)
    write_data(args.output_file, X, y)

if __name__ == "__main__":
    main()
