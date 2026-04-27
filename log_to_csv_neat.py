import re
import csv

# Input and output filenames
input_file = "temp.txt"
output_file = "experiment_neat.csv"

# Regex patterns to extract key-value pairs
patterns = {
    "epoch_num": r"epoch_num:\s*([0-9.]+)",
    "learning_rate": r"learning_rate:\s*([0-9.]+)",
    "evolution_epoch": r"evolution_epoch:\s*([0-9.]+)",
    "max_fitness": r"max_fitness:\s*([0-9.]+)",
    "average_loss_test_neat": r"average_loss_test_neat:\s*([0-9.]+)",
    "average_loss_test_mlp": r"average_loss_test_mlp:\s*([0-9.]+)",
    "node_num_neat": r"node_num_neat:\s*([0-9.]+)",
    "connection_num_neat": r"connection_num_neat:\s*([0-9.]+)"
}

# The order in which fields should appear in CSV
fieldnames = list(patterns.keys())

def parse_block(block):
    """Extract all fields from a block of text."""
    data = {}
    for key, pat in patterns.items():
        match = re.search(pat, block)
        data[key] = match.group(1) if match else ""
    return data

def main():
    # Read entire file
    with open(input_file, "r") as f:
        content = f.read()

    # Split into experiment blocks
    blocks = content.split("=============================================")
    parsed_rows = []

    for block in blocks:
        if "epoch_num" in block:  # basic filter to confirm it's a valid block
            parsed_rows.append(parse_block(block))

    # Write CSV
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in parsed_rows:
            writer.writerow(row)

    print(f"CSV written to {output_file}")

if __name__ == "__main__":
    main()
