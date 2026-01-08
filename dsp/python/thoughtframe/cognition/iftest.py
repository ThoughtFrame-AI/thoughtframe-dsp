import torch
import numpy as np
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
TEXT_CONTEXT_CHARS = 300   # chars before/after spike

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
    with open("sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

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
    prev_entropy = None

    print(f"Streaming {len(token_ids)} tokens...")
    t0 = time.perf_counter()

    for i, tid in enumerate(token_ids):
        logits = fs.step(tid)
        entropy = softmax_entropy(logits)

        ema = entropy if ema is None else EMA_ALPHA * entropy + (1 - EMA_ALPHA) * ema

        entropies.append(entropy)
        ema_values.append(ema)
        prev_entropy = entropy

    print(f"Done in {time.perf_counter() - t0:.2f}s")

    # -----------------------------
    # Isolation Forest
    # -----------------------------
    X = np.array(ema_values).reshape(-1, 1)
    iforest = IsolationForest(
        n_estimators=200,
        contamination=IF_CONTAMINATION,
        random_state=42
    )
    iforest.fit(X)
    scores = iforest.decision_function(X)

    threshold = np.percentile(scores, 5)

    # -----------------------------
    # Print spike contexts
    # -----------------------------
    print("\n==== SEMANTIC SPIKES ====\n")

    for i, score in enumerate(scores):
        if score < threshold:
            char_start = offsets[i][0]
            char_end = offsets[i][1]

            lo = max(0, char_start - TEXT_CONTEXT_CHARS)
            hi = min(len(text), char_end + TEXT_CONTEXT_CHARS)

            snippet = text[lo:hi].replace("\n", " ")

            print(f"\n--- Spike at token {i} (score={score:.3f}) ---")
            print(snippet)
            print("-" * 80)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(18, 6))
    plt.plot(scores, label="IF Anomaly Score")

    for i, s in enumerate(scores):
        if s < threshold:
            plt.axvline(i, color="red", alpha=0.15)

    plt.title("Isolation Forest Anomaly Score (Transformer Semantic Dynamics)")
    plt.xlabel("Token Index")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
