import os
import requests
import torch
import numpy as np
import time
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1. Config & Hyperparameters
# -----------------------------
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_FOLDER = "radar_db_omni2"
CHUNK_SIZE = 1024
OVERLAP = 64

# Physics Settings
DEFAULT_AGGRESSION = 0.40   # Confidence threshold (0.0-1.0)
DEFAULT_SUPPRESSION = 100   # Tokens to skip after a find (prevents duplicates)

os.makedirs(DB_FOLDER, exist_ok=True)
os.makedirs("omni_docs", exist_ok=True)

# -----------------------------
# 2. System Boot (Load Model Once)
# -----------------------------
print(f"\n🚀 BOOTING OMNI-RADAR on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32, 
    output_hidden_states=True, 
    trust_remote_code=True
).to(DEVICE)
model.eval()

# -----------------------------
# 3. Smart Ingestion Manager
# -----------------------------
def ingest_corpus():
    # A. Define the Corpus (The Tricolor Test)
    files_to_index = {
        "constitution.txt": "https://www.gutenberg.org/cache/epub/5/pg5.txt",
        "alice.txt": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "linux_kernel.c": None # We generate this locally
    }

    # Generate Code File if missing
    if not os.path.exists("omni_docs/linux_kernel.c"):
        c_code = """
/* Linux Kernel Scheduling Logic */
struct sched_entity {
    struct load_weight load;
    struct rb_node run_node;
    unsigned int on_rq;
    u64 exec_start;
    u64 sum_exec_runtime;
};
static void update_curr(struct cfs_rq *cfs_rq) {
    struct sched_entity *curr = cfs_rq->curr;
    u64 now = rq_clock_task(rq_of(cfs_rq));
    if (unlikely(!curr)) return;
    curr->sum_exec_runtime += (now - curr->exec_start);
}
""" * 100
        with open("omni_docs/linux_kernel.c", "w") as f: f.write(c_code)

    # Download others
    for name, url in files_to_index.items():
        if url and not os.path.exists(f"omni_docs/{name}"):
            try: 
                r = requests.get(url)
                with open(f"omni_docs/{name}", "w", encoding="utf-8") as f: f.write(r.text)
            except: pass

    # B. Ingest Loop (With Caching Check)
    print("\n💾 CHECKING DATABASE...")
    
    for filename in files_to_index.keys():
        save_path = os.path.join(DB_FOLDER, filename + ".pt")
        
        # SMART CACHE CHECK
        if os.path.exists(save_path):
            # print(f"   ✅ {filename} is cached.")
            continue
            
        print(f"   🧠 Ingesting {filename} (GPU Active)...", end="", flush=True)
        
        with open(os.path.join("omni_docs", filename), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        
        full_trace = []
        full_offsets = []

        # Sliding Window
        for i in range(0, len(input_ids), CHUNK_SIZE - OVERLAP):
            chunk_ids = input_ids[i : i + CHUNK_SIZE]
            if not chunk_ids: break
            
            chunk_tensor = torch.tensor([chunk_ids], device=DEVICE)
            with torch.no_grad():
                out = model(chunk_tensor, output_hidden_states=True)
            
            # Extract Layer -4 (The "Concept" Layer)
            vecs = out.hidden_states[-4][0].cpu().float().numpy()
            
            # Stitching (Trim overlap from start)
            start_idx = OVERLAP if i > 0 else 0
            full_trace.append(vecs[start_idx:])
            full_offsets.extend(offsets[i + start_idx : i + CHUNK_SIZE])
            
        if full_trace:
            master_tape = np.concatenate(full_trace, axis=0)
            torch.save({"trace": master_tape, "offsets": full_offsets, "text": text}, save_path)
            print(f" DONE ({len(master_tape)} tokens)")

# Run Ingest Once
ingest_corpus()

# -----------------------------
# 4. The Dual-Mode Search Engine
# -----------------------------
def radar_scan(query, mode="semantic", aggression=DEFAULT_AGGRESSION, suppression=DEFAULT_SUPPRESSION):
    print(f"\n🔎 {mode.upper()} SCAN: '{query}'")
    
    # A. Generate Pilot (The Needle)
    if mode == "semantic":
        # HYPILOT: Dream the answer -> Search for the Dream
        prompt = f"Question: {query}\nAnswer (technical):"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            out = model(gen_out, output_hidden_states=True)
            
        # Use mean of generated tokens
        start_idx = inputs['input_ids'].shape[1]
        pilot = out.hidden_states[-4][0, start_idx:, :].mean(dim=0).cpu().float().numpy()
        
    else:
        # EXACT: Encode query directly -> Search for the Reference
        q_vecs = tokenizer(query, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**q_vecs, output_hidden_states=True)
        # Use last token (sharpest signal)
        pilot = out.hidden_states[-4][0, -1, :].cpu().float().numpy()

    # B. Scan Tapes (The Haystack)
    tapes = glob.glob(os.path.join(DB_FOLDER, "*.pt"))
    hits = []
    
    for tape_path in tapes:
        # Load (Fast - Cached on SSD)
        data = torch.load(tape_path, weights_only=False)
        H = data["trace"]
        offsets = data["offsets"]
        text = data["text"]
        
        # DSP (Whitening + Norm)
        H_centered = H - np.mean(H, axis=0)
        p_centered = pilot - np.mean(H, axis=0)
        
        Hn = H_centered / (np.linalg.norm(H_centered, axis=1, keepdims=True) + 1e-9)
        pn = p_centered / (np.linalg.norm(p_centered) + 1e-9)
        
        scores = Hn @ pn
        
        # C. Peak Finding (Greedy)
        search_scores = scores.copy()
        for _ in range(5): # Max 5 hits per file
            idx = np.argmax(search_scores)
            score = search_scores[idx]
            
            if score < aggression: break
            
            # Extract Context
            start = offsets[max(0, idx-30)][0]
            end = offsets[min(len(offsets)-1, idx+30)][1]
            snippet = text[start:end].replace("\n", " ")
            
            hits.append({
                "file": os.path.basename(tape_path),
                "score": score,
                "snippet": snippet
            })
            
            # Suppress Region
            low = max(0, idx - suppression)
            high = min(len(search_scores), idx + suppression)
            search_scores[low:high] = -1.0

    # D. Sort & Report
    hits.sort(key=lambda x: x["score"], reverse=True)
    
    if not hits:
        print("   ❌ No matches found (Try lowering aggression).")
    else:
        for i, h in enumerate(hits[:3]): # Top 3 Global
            print(f"   [{i+1}] {h['file']:<20} | Conf: {h['score']:.3f} | \"...{h['snippet']}...\"")

# -----------------------------
# 5. The Playground
# -----------------------------
print("\n" + "="*60)
print("🧪 OMNI-RADAR READY. Running Benchmarks...")
print("="*60)

# TEST 1: SEMANTIC (Concepts)
# Note: "Aggression" is lower (0.35) because semantic matches are softer waves.
radar_scan("Who has the power to tax?", mode="semantic", aggression=0.35)
radar_scan("How does the scheduler track runtime?", mode="semantic", aggression=0.35)

# TEST 2: EXACT (References)
# Note: "Aggression" is higher (0.50) because these are sharp impulses.
radar_scan("Article I", mode="exact", aggression=0.50)
radar_scan("struct sched_entity", mode="exact", aggression=0.50)
radar_scan("The Queen of Hearts", mode="exact", aggression=0.45)