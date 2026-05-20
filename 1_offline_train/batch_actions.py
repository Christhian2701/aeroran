import argparse
import csv
import random
import string


# Configuration
duration_ms = 30000
interval_ms = 100
cells = [2, 3, 4, 5, 6, 7, 8]


def index_to_prefix(index: int) -> str:
    """
    Convert 0-based indices to alphabetic prefixes:
    0 -> a, 1 -> b, ..., 25 -> z, 26 -> aa, 27 -> ab, ...
    """
    if index < 0:
        raise ValueError("index must be >= 0")

    alphabet = string.ascii_lowercase
    result = ""
    value = index

    while True:
        value, remainder = divmod(value, 26)
        result = alphabet[remainder] + result
        if value == 0:
            break
        value -= 1

    return result


def build_output_filename(index: int) -> str:
    return f"{index_to_prefix(index)}hierarchical_actions.csv"


def generate_actions(rng: random.Random):
    actions = []

    for timestamp in range(interval_ms, duration_ms + interval_ms, interval_ms):
        for cell_id in cells:
            action_type = 0
            ho_allowed = 1 if rng.random() < 0.7 else 0
            actions.append([timestamp, action_type, cell_id, ho_allowed])

    return actions


def write_actions_file(output_filename: str, actions):
    with open(output_filename, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(actions)


def main():
    parser = argparse.ArgumentParser(
        description="Generate N hierarchical action CSVs with alphabetic prefixes."
    )
    parser.add_argument(
        "n",
        type=int,
        help="Number of CSV files to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base random seed for reproducible output.",
    )
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("n must be greater than 0")

    for index in range(args.n):
        rng = random.Random(None if args.seed is None else args.seed + index)
        output_filename = build_output_filename(index)
        actions = generate_actions(rng)
        write_actions_file(output_filename, actions)

        print(f"Generated {len(actions)} actions and saved to {output_filename}")

        if index == 0:
            print("\nPreview of generated actions:")
            for row in actions[:10]:
                print(f"Timestamp: {row[0]}ms | Cell {row[2]} -> State {row[3]}")


if __name__ == "__main__":
    main()
