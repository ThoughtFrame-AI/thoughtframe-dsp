import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import expon
import os

def run_forensics(filepath="thoughtframe/telemetry/raw_impulses.txt"):
    if not os.path.exists(filepath):
        print("File not found.")
        return

    # 1. Load Data
    timestamps = np.loadtxt(filepath)
    
    # --- DYNAMIC REGIME SELECTOR ---
    # If we have data past 15,000s, assume we want to see the Swarm.
    # Otherwise, assume we are analyzing the Boat/Noise (Current State).
    # if timestamps.max() > 15000:
    #     regime_name = "BIOLOGIC SWARM (t > 15,000s)"
    #     data = timestamps[timestamps > 15000]
    # else:
    #     regime_name = "MECHANICAL / NOISE BASELINE (Start of File)"
    #     data = timestamps # Look at everything we have so far
    #
    # print(f"Analyzing {len(data)} impulses from: {regime_name}")

    start_t = 4850
    end_t = 4950
    
    mask = (timestamps >= start_t) & (timestamps <= end_t)
    data = timestamps[mask]
    
    regime_name = f"BOAT PASSAGE ({start_t}s - {end_t}s)"
        
    print(f"Analyzing {len(data)} impulses from: {regime_name}")


    # 2. Process ICIs
    icis = np.diff(data)
    valid_icis = icis[(icis > 0.002) & (icis < 2.0)]

    # 3. Plotting
    plt.figure(figsize=(12, 7))
    plt.style.use('dark_background') # Professional sonar look

    # Histogram
    sns.histplot(valid_icis, bins=200, kde=False, element="step", 
                 color="#00ffcc", alpha=0.6, label="Observed Rhythm")

    # Random Noise Model (Poisson)
    if len(valid_icis) > 10:
        loc, scale = expon.fit(valid_icis)
        x = np.linspace(valid_icis.min(), valid_icis.max(), 200)
        pdf = expon.pdf(x, loc, scale)
        expected_y = pdf * len(valid_icis) * (x[1] - x[0])
        plt.plot(x, expected_y, 'w--', linewidth=2, alpha=0.8, label="Random Noise Model")

    # --- AXIS LABELS & TITLES (The Fix) ---
    plt.title(f"Forensic Identity: {regime_name}", fontsize=16, color='white', fontweight='bold', pad=20)
    
    plt.xlabel("Inter-Click Interval (Seconds)\n(Time between hits)", fontsize=12, color='white', labelpad=10)
    plt.ylabel("Count (Log Scale)", fontsize=12, color='white', labelpad=10)
    
    # Tick params to ensure visibility
    plt.tick_params(axis='both', colors='white', labelsize=10)
    
    plt.yscale('log') 
    plt.grid(True, which="both", alpha=0.2)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_forensics()