import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1. Config & Setup
# -----------------------------
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MAX_TOKENS = 1500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_FILE = "text_radar_tape3.pt" # Saving the Text Analysis

# DSP SETTINGS (The "Heavy Cruiser" Tune)
TARGET_LAYER = -4      # The "Physics" Layer
EMA_ALPHA = 0.03       # Slow/Heavy Smoothing
LOCK_THRESHOLD = 0.15  # High noise floor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TextSpeedRadar")

# -----------------------------
# 2. Pilot Definition (The Search Targets)
# -----------------------------
PILOTS = {
    # [CONTROL CHANNEL] The target we want to see spike
    "CODE": [
        "def process_stream(data):\n    if not data: return None",
        "int main(int argc, char* argv[]) { std::cout << 0x0F; }",
        "const config = { timeout: 5000, retries: 3 }; console.log(x);",
    ],
    # [CONTEXT CHANNELS]
    "LEGAL": [
        "The defendant knowingly misrepresented material facts.",
        "This constitutes a breach of contractual obligation.",
        "indemnification, liability, and jurisdiction clauses",
    ],
    "FINANCE": [
        "The transaction involved undisclosed liabilities.",
        "Revenue projections were materially overstated.",
        "EBITDA, fiscal quarter, and balance sheet adjustments",
    ],
    "STORY": [
        "The sun sank below the horizon as the ship sailed on.",
        "He remembered the sound of waves against the hull.",
        "She felt a deep sense of longing and regret.",
    ],
}

# -----------------------------
# PHASE 1: ACQUISITION (The Slow Part)
# -----------------------------
print("\n" + "="*60)
print("🧠 PHASE 1: INGESTION (Reading Text & Saving Brain State)")
print("="*60)

t0 = time.perf_counter()

# A. Load Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    output_hidden_states=True,
    trust_remote_code=True
).to(DEVICE)
model.eval()

# B. Load Text (Your sample.txt with the code block at the end)
try:
    with open("sample.txt", "r", encoding="utf-8") as f:
        text = f.read()
except:
    print("Error: sample.txt not found.")
    exit()

# C. Process Text
enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
token_ids = enc["input_ids"][:MAX_TOKENS]
offsets = enc["offset_mapping"][:MAX_TOKENS]
input_ids = torch.tensor([token_ids], device=DEVICE)

# INFERENCE
with torch.no_grad():
    out = model(input_ids)

# D. Extract Trace & Encode Pilots
# We extract the document trace
trace = out.hidden_states[TARGET_LAYER][0].cpu().float().numpy()

# We encode the pilots NOW (using the model) so we don't need the model later
pilot_vectors = {}
for name, sentences in PILOTS.items():
    vecs = []
    for s in sentences:
        e = tokenizer(s, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        with torch.no_grad():
            o = model(**e, output_hidden_states=True)
        vecs.append(o.hidden_states[TARGET_LAYER][0, -1, :].cpu().float().numpy())
    pilot_vectors[name] = np.mean(vecs, axis=0)

# E. Save EVERYTHING to Disk
torch.save({
    "trace": trace,
    "pilots": pilot_vectors,
    "offsets": offsets,
    "text": text
}, CACHE_FILE)

t_acq = time.perf_counter() - t0
print(f"✅ Acquisition Complete: {t_acq:.4f}s")
print(f"💾 Saved Vectors to {CACHE_FILE}")

# CLEANUP (Simulate shutting down the heavy GPU process)
del model
del input_ids
torch.cuda.empty_cache()

# -----------------------------
# PHASE 2: INSTANT REPLAY (The Fast Part)
# -----------------------------
print("\n" + "="*60)
print("🚀 PHASE 2: DSP SCAN (Pure Math - No Model)")
print("="*60)

t1 = time.perf_counter()

# 1. Load Tape (Safety fix included)
data = torch.load(CACHE_FILE, weights_only=False)
H = data["trace"]
P_dict = data["pilots"]
offsets_loaded = data["offsets"]
text_loaded = data["text"]

# 2. Whitening (Global Mean Removal)
global_mean = np.mean(H, axis=0)
H_centered = H - global_mean

pilot_names = list(P_dict.keys())
P_matrix = np.stack([P_dict[k] for k in pilot_names]) - global_mean

# 3. Beamforming
Hn = H_centered / (np.linalg.norm(H_centered, axis=1, keepdims=True) + 1e-9)
Pn = P_matrix / (np.linalg.norm(P_matrix, axis=1, keepdims=True) + 1e-9)

scores = Hn @ Pn.T

# 4. Logic Loop (EMA + Logging)
ema_state = np.zeros(len(pilot_names))
ema_history = np.zeros_like(scores)
is_locked = [False] * len(pilot_names)
lock_counters = np.zeros(len(pilot_names))

print(f"{'TIME':<6} | {'STATUS':<10} | {'CHANNEL':<10} | {'TRIGGER TEXT':<40}")
print("-" * 80)

for t in range(len(scores)):
    # Update EMA
    ema_state = (EMA_ALPHA * scores[t]) + ((1 - EMA_ALPHA) * ema_state)
    ema_history[t] = ema_state
    
    # Text Echo (Proof we are aligned)
    if t > 50:
        start_char = offsets_loaded[max(0, t-15)][0]
        end_char = offsets_loaded[t][1]
        echo_text = text_loaded[start_char:end_char].replace("\n", " ")

        # Check Triggers
        for i, name in enumerate(pilot_names):
            val = ema_state[i]
            
            if val > LOCK_THRESHOLD:
                lock_counters[i] += 1
                if lock_counters[i] >= 10 and not is_locked[i]: # Debounce 10
                    is_locked[i] = True
                    print(f"{t:<6} | \033[92mLOCKED    \033[0m | {name:<10} | \"...{echo_text[-35:]}...\"")
            
            elif val < (LOCK_THRESHOLD * 0.8):
                if is_locked[i]:
                    is_locked[i] = False
                    print(f"{t:<6} | \033[91mRELEASED  \033[0m | {name:<10} | (Signal dropped below {val:.3f})")
                lock_counters[i] = 0

t_scan = time.perf_counter() - t1
print("-" * 80)
print(f"⚡ Scan Complete: {t_scan:.4f}s")
print(f"SPEEDUP FACTOR: {t_acq / t_scan:.0f}x")

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(16, 8))
colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c'] # Code(Red), Legal, Finance, Story

# Align colors to names
name_to_color = {name: colors[i % 4] for i, name in enumerate(pilot_names)}

for i, name in enumerate(pilot_names):
    plt.plot(ema_history[:, i], label=name, color=name_to_color[name], linewidth=2.5)

plt.axhline(LOCK_THRESHOLD, color='gray', linestyle='--', alpha=0.7, label="Threshold")
plt.title(f"Instant Replay (Text Radar) | Speedup: {t_acq / t_scan:.0f}x", fontsize=14)
plt.xlabel("Token Time", fontsize=12)
plt.ylabel("Signal Strength (Whitened)", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()