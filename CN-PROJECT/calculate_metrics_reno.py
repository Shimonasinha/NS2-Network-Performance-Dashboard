def load_xg_file(filename):
    values = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                time, value = map(float, line.strip().split())
                if time == 0 and value == 0:
                    continue  # Skip dummy starter line
                values.append(value)
            except ValueError:
                continue
    return values

def analyze(name, data):
    if not data:
        print(f"No data in {name}")
        return

    avg = sum(data) / len(data)
    maximum = max(data)
    minimum = min(data)

    print(f"--- {name.upper()} ---")
    print(f"Average: {avg:.2f}")
    print(f"Maximum: {maximum:.2f}")
    print(f"Minimum: {minimum:.2f}")
    print()

def main():
    files = {
        'Throughput': 'throughput-reno.xg',
        'Latency': 'latency-reno.xg',
        'Congestion Window': 'cwnd-reno.xg',
        'Packet Drop': 'drop-reno.xg'
    }

    for name, filename in files.items():
        try:
            data = load_xg_file(filename)
            analyze(name, data)
        except FileNotFoundError:
            print(f"{filename} not found.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
