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
        "THROUGHPUT": "throughput-tahoe.xg",
        "LATENCY": "latency-tahoe.xg",
        "PACKET DROP": "drop-tahoe.xg",
        "CONGESTION WINDOW": "cwnd-tahoe.xg"
    }

    for name, file in metrics.items():
        data = parse_xg_file(file)
        print_stats(name, data)

if __name__ == "__main__":
    main()
