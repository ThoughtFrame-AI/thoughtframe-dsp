import torch
import numpy as np
import pandas as pd  # <--- NEW: For rolling stats
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "microsoft/phi-2"
EMA_ALPHA = 0.05
MAX_TOKENS = 1200
IF_CONTAMINATION = 0.05
TEXT_CONTEXT_CHARS = 100   # chars before/after spike (Reduced for cleaner prints)
ROLLING_WINDOW = 20        # <--- NEW: Window size for variance

# -----------------------------
def softmax_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    logp = torch.log(probs + 1e-9)
    return -(probs * logp).sum().item()

# -----------------------------
class FrameSession:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.past_key_values = None

    @torch.no_grad()
    def step(self, token_id):
        token = torch.tensor([[token_id]], device=self.device)
        out = self.model(
            input_ids=token,
            past_key_values=self.past_key_values,
            use_cache=True,
            return_dict=True
        )
        self.past_key_values = out.past_key_values
        return out.logits[:, -1, :]

# -----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)
    model.eval()

    # ---- Load text ----
    try:
        with open("sample.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print("❌ Error: 'sample.txt' not found. Please create it with the Moby Dick + Code mix.")
        return

    print("Tokenizing...")
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    token_ids = enc["input_ids"][:MAX_TOKENS]
    offsets = enc["offset_mapping"][:MAX_TOKENS]

    fs = FrameSession(model, device)

    entropies = []
    ema_values = []

    ema = None
    
    print(f"Streaming {len(token_ids)} tokens...")
    t0 = time.perf_counter()

    for i, tid in enumerate(token_ids):
        logits = fs.step(tid)
        entropy = softmax_entropy(logits)

        # Update EMA
        ema = entropy if ema is None else EMA_ALPHA * entropy + (1 - EMA_ALPHA) * ema

        entropies.append(entropy)
        ema_values.append(ema)

    print(f"Done in {time.perf_counter() - t0:.2f}s")

    # -----------------------------
    # Feature Engineering (The Fix)
    # -----------------------------
    # 1. Rolling Variance (Texture)
    rolling_var = pd.Series(entropies).rolling(window=ROLLING_WINDOW).var().fillna(0).values
    
    # 2. Stack Features: [EMA_Entropy, Rolling_Variance]
    X = np.column_stack((ema_values, rolling_var))
    
    # -----------------------------
    # Isolation Forest
    # -----------------------------
    print("Fitting Isolation Forest on [Entropy, Variance]...")
    iforest = IsolationForest(
        n_estimators=200,
        contamination=IF_CONTAMINATION,
        random_state=42
    )
    iforest.fit(X)
    scores = iforest.decision_function(X)

    # Use a dynamic threshold based on the score distribution
    threshold = np.percentile(scores, 5)

    # -----------------------------
    # Print spike contexts
    # -----------------------------
    print("\n==== SEMANTIC REGIME CHANGES ====\n")

    # Simple logic to group contiguous spikes so we don't print 100 times for one block
    in_spike_region = False
    
    for i, score in enumerate(scores):
        if score < threshold:
            if not in_spike_region:
                # START of a new anomaly region
                char_start = offsets[i][0]
                char_end = offsets[i][1]
                
                # Grab context
                lo = max(0, char_start - TEXT_CONTEXT_CHARS)
                hi = min(len(text), char_end + TEXT_CONTEXT_CHARS)
                snippet = text[lo:hi].replace("\n", " ")

                print(f"🔴 ANOMALY DETECTED at Token {i} (Score: {score:.3f})")
                print(f"Context: \"...{snippet}...\"")
                print("-" * 60)
                in_spike_region = True
        else:
            in_spike_region = False

    # -----------------------------
    # Visualization
    # -----------------------------
    plt.figure(figsize=(18, 8))
    
    # Plot 1: The Raw Signals (Entropy + Variance)
    plt.subplot(2, 1, 1)
    plt.plot(ema_values, label="Entropy (EMA)", color="blue", linewidth=1)
    plt.plot(rolling_var, label="Variance (Texture)", color="green", alpha=0.6, linewidth=1)
    plt.title("The Raw Signals: Entropy Level vs. Texture")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: The Anomaly Score
    plt.subplot(2, 1, 2)
    plt.plot(scores, label="Isolation Forest Score", color="black", linewidth=1.5)
    
    # Highlight anomaly regions
    for i, s in enumerate(scores):
        if s < threshold:
            plt.axvline(i, color="red", alpha=0.1)

    plt.axhline(threshold, color='orange', linestyle='--', label="Threshold (5%)")
    plt.title("Anomaly Detection: Identifying Regime Changes")
    plt.xlabel("Token Index")
    plt.ylabel("Anomaly Score (Lower = More Anomalous)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()