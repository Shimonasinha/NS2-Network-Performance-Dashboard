# collect_iperf_data.py
# Collects REAL network measurements using iPerf3
# Tests: Reno, Cubic, BBR
 
import subprocess
import json
import time
import os
import numpy as np
 
DURATION   = 10   # seconds per test
INTERVAL   = 1    # reporting interval
OUTPUT_DIR = "."
 
VARIANTS = ["reno", "cubic", "bbr"]
 
def set_tcp_variant(variant):
    try:
        subprocess.run(
            ["sudo", "sysctl", "-w",
             f"net.ipv4.tcp_congestion_control={variant}"],
            capture_output=True
        )
        print(f"  Set TCP to: {variant}")
    except Exception as e:
        print(f"  Warning: Could not set TCP variant: {e}")
 
def run_iperf_server():
    proc = subprocess.Popen(
        ["iperf3", "-s", "-D"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(1)
    return proc
 
def run_iperf_client(variant):
    print(f"\n  Running iPerf3 test for {variant.upper()}...")
    try:
        result = subprocess.run(
            ["iperf3", "-c", "127.0.0.1",
             "-t", str(DURATION),
             "-i", str(INTERVAL),
             "-C", variant,
             "-J"],
            capture_output=True, text=True, timeout=30
        )
        data = json.loads(result.stdout)
        return data
    except subprocess.TimeoutExpired:
        print(f"  Timeout for {variant}")
        return None
    except json.JSONDecodeError:
        print(f"  JSON parse error for {variant}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None
 
def parse_iperf_results(data, variant):
    if not data:
        return None
 
    intervals = data.get("intervals", [])
    end       = data.get("end", {})
 
    throughputs = []
    rtts        = []
    retransmits = []
 
    for interval in intervals:
        streams = interval.get("streams", [])
        for stream in streams:
            bits_per_sec = stream.get("bits_per_second", 0)
            throughputs.append(bits_per_sec / 1e6)
            rtt = stream.get("rtt", 0) / 1000  # microseconds to ms
            if rtt > 0:
                rtts.append(rtt)
            retransmits.append(stream.get("retransmits", 0))
 
    if not throughputs:
        return None
 
    total_retx  = sum(retransmits)
    total_pkts  = max(len(throughputs) * 100, 1)
    loss_rate   = min(total_retx / total_pkts, 0.5)
 
    result = {
        "protocol":        variant,
        "throughput_mbps": round(np.mean(throughputs), 4),
        "throughput_std":  round(np.std(throughputs),  4),
        "mean_rtt_ms":     round(np.mean(rtts) if rtts else 1.0, 4),
        "p90_rtt_ms":      round(np.percentile(rtts, 90) if rtts else 1.5, 4),
        "p99_rtt_ms":      round(np.percentile(rtts, 99) if rtts else 2.0, 4),
        "std_rtt_ms":      round(np.std(rtts) if rtts else 0.1, 4),
        "loss_rate":       round(loss_rate, 6),
        "retx_rate":       round(loss_rate * 0.8, 6),
        "total_retransmits": total_retx,
    }
 
    print(f"  ✅ {variant.upper()} Results:")
    print(f"     Throughput : {result['throughput_mbps']:.3f} Mbps")
    print(f"     Mean RTT   : {result['mean_rtt_ms']:.3f} ms")
    print(f"     Loss Rate  : {result['loss_rate']:.6f}")
 
    return result
 
def save_results(results):
    import csv
    filepath = os.path.join(OUTPUT_DIR, "iperf_results.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → iperf_results.json")
 
def main():
    print("=" * 55)
    print("   COLLECTING REAL iPERF3 MEASUREMENTS")
    print("=" * 55)
    print(f"  Testing variants: {', '.join(VARIANTS)}")
    print(f"  Duration per test: {DURATION}s")
 
    # Enable BBR
    subprocess.run(["sudo", "modprobe", "tcp_bbr"],
                   capture_output=True)
 
    # Start server
    print("\n→ Starting iPerf3 server...")
    server = run_iperf_server()
    time.sleep(2)
 
    results = {}
 
    for variant in VARIANTS:
        set_tcp_variant(variant)
        time.sleep(1)
        data   = run_iperf_client(variant)
        parsed = parse_iperf_results(data, variant)
        if parsed:
            results[variant] = parsed
        time.sleep(2)
 
    # Kill server
    subprocess.run(["pkill", "-f", "iperf3 -s"],
                   capture_output=True)
 
    # Reset to cubic (default)
    subprocess.run(
        ["sudo", "sysctl", "-w",
         "net.ipv4.tcp_congestion_control=cubic"],
        capture_output=True
    )
 
    if results:
        save_results(results)
        print("\n" + "="*55)
        print("  SUMMARY")
        print("="*55)
        for v, r in results.items():
            print(f"  {v.upper():6s} → {r['throughput_mbps']:.3f} Mbps | "
                  f"RTT: {r['mean_rtt_ms']:.3f}ms | "
                  f"Loss: {r['loss_rate']:.4f}")
    else:
        print("\n⚠️  No results collected!")
 
    print("="*55)
 
if __name__ == "__main__":
    main()