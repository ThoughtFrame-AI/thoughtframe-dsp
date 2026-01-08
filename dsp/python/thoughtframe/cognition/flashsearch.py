import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1. Config
# -----------------------------
MODEL_ID = "microsoft/phi-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_FILE = "text_radar_tape.pt" # The saved brain state from previous run

# SEARCH SETTINGS
TOP_K = 3               # Number of results to return
CONTEXT_WINDOW = 150    # Characters to print around the hit
EMA_ALPHA = 0.1         # Less smoothing for search (we want sharp spikes)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("FlashSearch")

# -----------------------------
# 2. Setup (Only runs once)
# -----------------------------
print("\n" + "="*60)
print("🔎 FLASH SEARCH ENGINE (Loading Tape...)")
print("="*60)

# Load the "Dead" Tape (No Model required for scanning, but we need model to Encode the Query)
# In a production app, the Query Encoder would be a tiny separate model.
# Here we just reload Phi-2 quickly to encode the pilot.
t_load = time.perf_counter()

# Load Tape
data = torch.load(CACHE_FILE, weights_only=False)
H_tape = data["trace"]
offsets = data["offsets"]
full_text = data["text"]

# Load Model (Just for Query Encoding)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    output_hidden_states=True,
    trust_remote_code=True
).to(DEVICE)
model.eval()

print(f"✅ System Ready ({time.perf_counter() - t_load:.2f}s)")

# -----------------------------
# 3. The Search Function
# -----------------------------
def search(user_query, target_layer=-4):
    print(f"\n❓ QUERY: \"{user_query}\"")
    
    t0 = time.perf_counter()
    
    # A. CREATE PILOT (The "Search Needle")
    # We create a simple vector representation of the query
    inputs = tokenizer(user_query, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    
    # Use the last token of the query as the "Concept Summary"
    # (Or mean pool, but last token is often sharper for Phi-2)
    pilot_vec = out.hidden_states[target_layer][0, -1, :].cpu().float().numpy()
    
    # B. DSP SCAN (The "Haystack Search")
    # 1. Whitening (Match coordinate systems)
    global_mean = np.mean(H_tape, axis=0)
    
    H_centered = H_tape - global_mean
    P_centered = pilot_vec - global_mean
    
    # 2. Normalize
    Hn = H_centered / (np.linalg.norm(H_centered, axis=1, keepdims=True) + 1e-9)
    Pn = P_centered / (np.linalg.norm(P_centered) + 1e-9)
    
    # 3. Beamforming (Dot Product)
    # Shape: [Tokens] @ [Hidden] -> [Tokens]
    scores = Hn @ Pn 
    
    # 4. Smoothing (To find "Regions" not just words)
    ema_scores = np.zeros_like(scores)
    ema = 0
    for i in range(len(scores)):
        ema = (EMA_ALPHA * scores[i]) + ((1 - EMA_ALPHA) * ema)
        ema_scores[i] = ema
        
    t_scan = time.perf_counter() - t0
    
    # C. EXTRACT RESULTS
    # Find indices of the highest peaks
    # We zero out neighbors to avoid finding the same sentence 5 times
    search_scores = ema_scores.copy()
    results = []
    
    for _ in range(TOP_K):
        idx = np.argmax(search_scores)
        score = search_scores[idx]
        
        # If score is too low, stop
        if score < 0.05: break 
        
        # Get Text Context
        start_char = offsets[max(0, idx-20)][0] # Look back 20 tokens
        end_char = offsets[min(len(offsets)-1, idx+20)][1] # Look forward 20 tokens
        
        snippet = full_text[start_char:end_char].replace("\n", " ")
        results.append((idx, score, snippet))
        
        # Suppress this region so we find the NEXT highest peak
        suppress_radius = 50 # tokens
        low = max(0, idx - suppress_radius)
        high = min(len(search_scores), idx + suppress_radius)
        search_scores[low:high] = -1.0

    # D. REPORT
    print(f"⚡ Search Time: {t_scan*1000:.2f} ms")
    print("-" * 60)
    for rank, (idx, score, snippet) in enumerate(results):
        print(f"#{rank+1} [Score: {score:.3f}] @ Token {idx}")
        print(f"   \"...{snippet}...\"")
    print("-" * 60)
    
    return ema_scores

# -----------------------------
# 4. Interactive Loop
# -----------------------------
# We'll run a few predefined queries based on your sample text
# Since your sample text is "Sea -> Legal -> Engine -> Finance -> Story -> Code"

queries = [
    "What are the terms regarding liability and breach of contract?",
    "How do we service the engine assembly?",
    "Describe the financial adjustments and revenue?",
    "Show me the code logic for the stream.",
]

scores_list = []
labels = []

for q in queries:
    s = search(q)
    scores_list.append(s)
    labels.append(q[:20] + "...")

# -----------------------------
# 5. Visualization (Search Heatmap)
# -----------------------------
plt.figure(figsize=(16, 8))
colors = ['purple', 'orange', 'green', 'red']

for i, s in enumerate(scores_list):
    plt.plot(s, label=labels[i], linewidth=2, color=colors[i % len(colors)])

plt.title("Flash Search Results (Semantic Resonance)", fontsize=14)
plt.xlabel("Token Index", fontsize=12)
plt.ylabel("Relevance Score", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()