import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================
# 1. CONFIG
# =====================================================
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_FILE = "qwen3_constituion_tape.pt"
MAX_TOKENS = 1600

# --- Tri-band definition (stable offsets, not magic)
BANDS = {
    "LOW":  -8,   # lexical / syntax
    "MID":  -4,   # logic / proposition (anchor)
    "HIGH": -2,   # abstraction
}

BAND_WEIGHTS = {
    "LOW":  0.25,
    "MID":  0.50,
    "HIGH": 0.25,
}

EMA_ALPHA = 0.03
LOCK_THRESHOLD = 0.16

# =====================================================
# 2. ACQUISITION (SLOW / GPU)
# =====================================================
def ingest(text_path):
    print(f"\n🧠 INGESTING WITH {MODEL_ID}")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        output_hidden_states=True,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"][:MAX_TOKENS]
    offsets = enc["offset_mapping"][:MAX_TOKENS]
    input_ids = torch.tensor([ids], device=DEVICE)

    with torch.no_grad():
        out = model(input_ids)

    traces = {}
    for name, layer in BANDS.items():
        traces[name] = out.hidden_states[layer][0].cpu().float().numpy()

    # --- pilot archetypes (encode-only)
    PILOTS = {
        "CODE": [
            "def process(data): return None",
            "struct node { int id; };",
            "const cfg = { timeout: 3000 };"
        ],
        "LEGAL": [
            "This constitutes a breach of contract.",
            "The power to tax is vested by law.",
            "Jurisdiction and liability apply."
        ],
        "STORY": [
            "The ship vanished into the fog.",
            "He felt regret as the sun set.",
            "A quiet sadness filled the room."
        ],
        "PRESIDENT": [
            "No person shall be eligible to the office of President unless",
            "The President must be at least thirty five years old",
            "Eligibility requirements for the office of President include age",
            "No person shall be eligible unless he has attained the age of",
            
        ],
        
        
    }

    pilot_vectors = {b: {} for b in BANDS}

    for pname, sentences in PILOTS.items():
        vecs_by_band = {b: [] for b in BANDS}
        for s in sentences:
            e = tokenizer(s, return_tensors="pt", add_special_tokens=False).to(DEVICE)
            with torch.no_grad():
                o = model(**e, output_hidden_states=True)
            for b, layer in BANDS.items():
                vecs_by_band[b].append(
                    o.hidden_states[layer][0, -1, :].cpu().float().numpy()
                )
        for b in BANDS:
            pilot_vectors[b][pname] = np.mean(vecs_by_band[b], axis=0)

    torch.save({
        "traces": traces,
        "pilots": pilot_vectors,
        "offsets": offsets,
        "text": text
    }, CACHE_FILE)

    del model
    torch.cuda.empty_cache()

    print(f"✅ INGEST COMPLETE ({time.perf_counter() - t0:.2f}s)")

# =====================================================
# 3. TRI-BAND DSP SCAN (FAST / CPU)
# =====================================================
def scan(channel="LEGAL", top_n=8):
    data = torch.load(CACHE_FILE, weights_only=False)

    offsets = data["offsets"]
    text = data["text"]

    # --- whitening per band
    band_scores = {}

    for b in BANDS:
        H = data["traces"][b]
        P = data["pilots"][b][channel]

        mu = np.mean(H, axis=0)
        Hc = H - mu
        Pc = P - mu

        Hn = Hc / (np.linalg.norm(Hc, axis=1, keepdims=True) + 1e-9)
        Pn = Pc / (np.linalg.norm(Pc) + 1e-9)

        band_scores[b] = Hn @ Pn

    # --- fused score (instantaneous)
    fused = np.zeros(len(offsets))
    for b in BANDS:
        fused += BAND_WEIGHTS[b] * band_scores[b]

    # --- EMA (inertial)
    ema = 0.0
    ema_history = np.zeros_like(fused)
    locks = []

    for t in range(len(fused)):
        ema = EMA_ALPHA * fused[t] + (1 - EMA_ALPHA) * ema
        ema_history[t] = ema

        if ema > LOCK_THRESHOLD:
            start = offsets[max(0, t-20)][0]
            end = offsets[t][1]
            snippet = text[start:end].replace("\n", " ")
            locks.append((t, ema, snippet))

    # --- Top-N peaks (inspection)
    top_idx = np.argsort(fused)[-top_n:][::-1]
    top_hits = []

    for idx in top_idx:
        start = offsets[max(0, idx-20)][0]
        end = offsets[min(len(offsets)-1, idx+20)][1]
        snippet = text[start:end].replace("\n", " ")

        top_hits.append({
            "token": idx,
            "score": fused[idx],
            "ema": ema_history[idx],
            "bands": {b: band_scores[b][idx] for b in BANDS},
            "snippet": snippet
        })

    return fused, ema_history, top_hits, locks


# =====================================================
# 4. RUN
# =====================================================
if __name__ == "__main__":
    if not os.path.exists(CACHE_FILE):
        ingest("sample.txt")

    fused, ema, hits, locks = scan("PRESIDENT")

    print("\n🏁 TOP SEMANTIC HITS (Ranked)")
    print("=" * 80)

    for i, h in enumerate(hits, 1):
        print(
            f"[{i}] token={h['token']:4d} | "
            f"raw={h['score']:.3f} | ema={h['ema']:.3f}"
        )
        print(
            f"    bands: "
            f"LOW={h['bands']['LOW']:+.3f}  "
            f"MID={h['bands']['MID']:+.3f}  "
            f"HIGH={h['bands']['HIGH']:+.3f}"
        )
        print(f"    \"...{h['snippet']}...\"\n")

    plt.figure(figsize=(15, 5))
    plt.plot(ema, label="EMA (Fused)", color="purple", linewidth=2)
    plt.plot(fused, alpha=0.25, color="gray", label="Raw Fused")
    plt.axhline(LOCK_THRESHOLD, linestyle="--", color="red", alpha=0.7)
    plt.title("Qwen3-4B Tri-Band Semantic Radar")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()
