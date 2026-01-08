import os
import requests
import torch
import numpy as np
import time
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1. Config
# -----------------------------
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DB_FOLDER = "radar_db_omni3"
DOC_FOLDER = "omni_docs"

CHUNK_SIZE = 1024
OVERLAP = 64

# Tri-band semantic layers
BANDS = {
    "LOW":  -8,
    "MID":  -4,
    "HIGH": -2,
}

BAND_WEIGHTS = {
    "LOW":  0.25,
    "MID":  0.50,
    "HIGH": 0.25,
}

DEFAULT_AGGRESSION = 0.40
DEFAULT_SUPPRESSION = 100

os.makedirs(DB_FOLDER, exist_ok=True)
os.makedirs(DOC_FOLDER, exist_ok=True)

# -----------------------------
# 2. Load model ONCE
# -----------------------------
print(f"\n🚀 BOOTING OMNI-RADAR (Tri-Band) on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
    output_hidden_states=True,
    trust_remote_code=True
).to(DEVICE)
model.eval()

# -----------------------------
# 3. Ingestion (cached)
# -----------------------------
def ingest_corpus():
    files = {
        "constitution.txt": "https://www.gutenberg.org/cache/epub/5/pg5.txt",
        "alice.txt": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "linux_kernel.c": None
    }

    # generate code file
    if not os.path.exists(f"{DOC_FOLDER}/linux_kernel.c"):
        code = """
struct sched_entity {
    unsigned long exec_start;
    unsigned long sum_exec_runtime;
};
""" * 200
        with open(f"{DOC_FOLDER}/linux_kernel.c", "w") as f:
            f.write(code)

    # download text files
    for name, url in files.items():
        if url and not os.path.exists(f"{DOC_FOLDER}/{name}"):
            r = requests.get(url)
            with open(f"{DOC_FOLDER}/{name}", "w", encoding="utf-8") as f:
                f.write(r.text)

    print("\n💾 CHECKING DATABASE...")
    for name in files.keys():
        save_path = f"{DB_FOLDER}/{name}.pt"
        if os.path.exists(save_path):
            continue

        print(f"   🧠 Ingesting {name} (GPU)...", end="", flush=True)

        with open(f"{DOC_FOLDER}/{name}", "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        band_traces = {b: [] for b in BANDS}
        band_offsets = []

        for i in range(0, len(ids), CHUNK_SIZE - OVERLAP):
            chunk = ids[i:i+CHUNK_SIZE]
            if not chunk:
                break

            with torch.no_grad():
                out = model(torch.tensor([chunk], device=DEVICE))

            start = OVERLAP if i > 0 else 0

            for b, layer in BANDS.items():
                band_traces[b].append(
                    out.hidden_states[layer][0, start:].cpu().float().numpy()
                )

            band_offsets.extend(offsets[i+start:i+CHUNK_SIZE])

        traces = {
            b: np.concatenate(band_traces[b], axis=0)
            for b in BANDS
        }

        torch.save(
            {"traces": traces, "offsets": band_offsets, "text": text},
            save_path
        )
        print(f" DONE ({len(band_offsets)} tokens)")

# -----------------------------
# 4. Tri-Band Radar Scan
# -----------------------------
def radar_scan(query, mode="semantic",
               aggression=DEFAULT_AGGRESSION,
               suppression=DEFAULT_SUPPRESSION):

    print(f"\n🔎 {mode.upper()} SCAN: '{query}'")

    # --- pilot
    if mode == "semantic":
        prompt = f"Question: {query}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            out = model(gen, output_hidden_states=True)
        start = inputs["input_ids"].shape[1]
        pilot_tokens = out.hidden_states

        pilots = {
            b: pilot_tokens[layer][0, start:, :].mean(dim=0).cpu().numpy()
            for b, layer in BANDS.items()
        }
    else:
        e = tokenizer(query, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**e, output_hidden_states=True)
        pilots = {
            b: out.hidden_states[layer][0, -1, :].cpu().numpy()
            for b, layer in BANDS.items()
        }

    hits = []

    for tape in glob.glob(f"{DB_FOLDER}/*.pt"):
        data = torch.load(tape, weights_only=False)
        offsets = data["offsets"]
        text = data["text"]

        fused = np.zeros(len(offsets))

        for b in BANDS:
            H = data["traces"][b]
            p = pilots[b]

            mu = H.mean(axis=0)
            Hc = H - mu
            pc = p - mu

            Hn = Hc / (np.linalg.norm(Hc, axis=1, keepdims=True) + 1e-9)
            pn = pc / (np.linalg.norm(pc) + 1e-9)

            fused += BAND_WEIGHTS[b] * (Hn @ pn)

        scores = fused.copy()

        for _ in range(5):
            idx = np.argmax(scores)
            val = scores[idx]
            if val < aggression:
                break

            start = offsets[max(0, idx-40)][0]
            end   = offsets[min(len(offsets)-1, idx+40)][1]
            snippet = text[start:end].replace("\n", " ")

            hits.append({
                "file": os.path.basename(tape),
                "score": val,
                "snippet": snippet
            })

            scores[max(0, idx-suppression):idx+suppression] = -1

    hits.sort(key=lambda x: x["score"], reverse=True)

    for i, h in enumerate(hits[:3]):
        print(f"   [{i+1}] {h['file']:<20} | {h['score']:.3f} | \"...{h['snippet']}...\"")

# -----------------------------
# 5. Run
# -----------------------------
ingest_corpus()

print("\n" + "="*60)
print("🧪 TRI-BAND OMNI-RADAR READY")
print("="*60)

radar_scan("Who has the power to tax?", mode="semantic", aggression=0.35)
radar_scan("How does the scheduler track runtime?", mode="semantic", aggression=0.35)
radar_scan("Article I", mode="exact", aggression=0.50)
radar_scan("struct sched_entity", mode="exact", aggression=0.50)
