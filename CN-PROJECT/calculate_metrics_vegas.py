def parse_xg_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            try:
                time, value = map(float, line.strip().split())
                if time == 0 and value == 0:
                    continue  # skip dummy entry
                data.append(value)
            except:
                continue
    return data

def print_stats(name, data):
    if not data:
        print(f"{name}: No data")
        return
    print(f"--- {name.upper()} ---")
    print(f"Average: {sum(data)/len(data):.2f}")
    print(f"Maximum: {max(data):.2f}")
    print(f"Minimum: {min(data):.2f}")

def main():
    metrics = {
        "THROUGHPUT": "throughput-vegas.xg",
        "LATENCY": "latency-vegas.xg",
        "PACKET DROP": "drop-vegas.xg",
        "CONGESTION WINDOW": "cwnd-vegas.xg"
    }

    for name, file in metrics.items():
        data = parse_xg_file(file)
        print_stats(name, data)

if __name__ == "__main__":
    main()
