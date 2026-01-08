import torch
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1. CONFIG
# -----------------------------
MODEL_ID = "microsoft/phi-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DB_FOLDER = "radar_db_triband"
TOP_N_CANDIDATES = 5   # Send top 5 radar hits to the Judge
CONTEXT_WINDOW = 65    # +/- 65 tokens (Grab the full paragraph)

BANDS = {
    5:  "Syntax (L05)",
    16: "Fact   (L16)",
    28: "Concept (L28)"
}

if 'model' not in globals():
    print(f"🚀 Booting Judge Probe on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        output_hidden_states=True, 
        trust_remote_code=True
    ).to(DEVICE)

# -----------------------------
# 2. INGEST ENGINE
# -----------------------------
def ensure_ingest():
    os.makedirs(DB_FOLDER, exist_ok=True)
    tape_path = os.path.join(DB_FOLDER, "constitution.txt.pt")
    if os.path.exists(tape_path):
        try:
            data = torch.load(tape_path, weights_only=False)
            if 16 in data["bands"]: return 
        except: pass

    print("⚡ Generating Tri-Band Tape...")
    text = requests.get("https://www.gutenberg.org/cache/epub/5/pg5.txt").text
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    
    CHUNK_SIZE = 512
    band_traces = {layer: [] for layer in BANDS.keys()}
    full_offsets = []
    
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    for i in range(0, len(input_ids), CHUNK_SIZE):
        chunk = input_ids[i:i+CHUNK_SIZE]
        if not chunk: break
        with torch.no_grad():
            out = model(torch.tensor([chunk]).to(DEVICE), output_hidden_states=True)
        for layer in BANDS.keys():
            vec = out.hidden_states[layer][0].cpu().numpy().astype(np.float16)
            band_traces[layer].append(vec)
        full_offsets.extend(offsets[i:i+CHUNK_SIZE])
        
    torch.save({
        "bands": {k: np.concatenate(v, axis=0) for k, v in band_traces.items()},
        "offsets": full_offsets,
        "text": text
    }, tape_path)
    print("✅ Ingest Complete.")

# -----------------------------
# 3. HELPER: EXTRACT CANDIDATES
# -----------------------------
def get_candidates(scores, offsets, text, min_height=0.15):
    # Find peaks in the beamformed signal
    peaks, _ = find_peaks(scores, height=min_height, distance=50)
    hits = []
    for idx in peaks:
        score = scores[idx]
        start = offsets[max(0, idx - CONTEXT_WINDOW)][0]
        end = offsets[min(len(offsets)-1, idx + CONTEXT_WINDOW)][1]
        snippet = text[start:end].replace("\n", " ")
        hits.append({"radar_score": score, "idx": idx, "snippet": snippet})
    
    # Sort by raw radar score first
    hits.sort(key=lambda x: x["radar_score"], reverse=True)
    return hits[:TOP_N_CANDIDATES]

# -----------------------------
# 4. THE NEURAL JUDGE (RE-RANKER)
# -----------------------------
def judge_candidates(query, candidates):
    if not candidates: return []
    
    print(f"\n⚖️  The Judge is deliberating on {len(candidates)} candidates...")
    ranked = []
    
    for cand in candidates:
        # Prompt: Ask the model to confirm if the snippet answers the query.
        # We look at the probability of the token "Yes" vs "No".
        prompt = f"Question: {query}\nContext: \"...{cand['snippet']}...\"\nDoes the context answer the question? (Yes/No):"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            
            # Phi-2 Token IDs (Approximate, using string encoding to be safe)
            yes_id = tokenizer.encode("Yes")[0]
            no_id = tokenizer.encode("No")[0]
            
            probs = torch.softmax(logits, dim=0)
            yes_score = probs[yes_id].item()
            no_score = probs[no_id].item()
            
            # Judge Score = Probability of "Yes"
            relevance = yes_score / (yes_score + no_score + 1e-9)
            
        cand["judge_score"] = relevance
        ranked.append(cand)

    # Sort FINAL list by Judge Score
    ranked.sort(key=lambda x: x["judge_score"], reverse=True)
    return ranked

# -----------------------------
# 5. BEAMFORMING ENGINE
# -----------------------------
def run_probe(query):
    ensure_ingest()
    tape_path = os.path.join(DB_FOLDER, "constitution.txt.pt")
    data = torch.load(tape_path, weights_only=False)
    text = data["text"]
    offsets = data["offsets"]
    
    print(f"\n" + "█"*80)
    print(f"📡 PROBE: '{query}'")
    print("█"*80)

    # --- A. DREAM ECHO (DEBUGGING) ---
    prompt = f"Question: {query}\nAnswer (technical):"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        # Generate the dream text
        gen_ids = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=50256)
        out_sem = model(gen_ids, output_hidden_states=True)
        
        # Decode specifically the new tokens
        input_len = inputs['input_ids'].shape[1]
        dream_text = tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True)

    print(f"🧠 DREAM ECHO: \"{dream_text.strip()}\"")
    
    # Extract Pilots
    start_idx = inputs['input_ids'].shape[1]
    pilot_fact    = out_sem.hidden_states[16][0, start_idx:, :].mean(dim=0).cpu().float().numpy()
    pilot_concept = out_sem.hidden_states[28][0, start_idx:, :].mean(dim=0).cpu().float().numpy()
    
    q_vecs = tokenizer(query, return_tensors="pt").to(DEVICE)
    with torch.no_grad(): out_exact = model(**q_vecs, output_hidden_states=True)
    pilot_syntax = out_exact.hidden_states[5][0, -1, :].cpu().float().numpy()

    # --- B. PHYSICS ---
    H5 = data["bands"][5].astype(np.float32)
    s5 = (H5 / np.linalg.norm(H5, axis=1, keepdims=True)) @ (pilot_syntax / np.linalg.norm(pilot_syntax))
    
    H16 = data["bands"][16].astype(np.float32)
    H16c = H16 - np.mean(H16, axis=0)
    p16c = pilot_fact - np.mean(H16, axis=0)
    s16 = (H16c / np.linalg.norm(H16c, axis=1, keepdims=True)) @ (p16c / np.linalg.norm(p16c))
    
    H28 = data["bands"][28].astype(np.float32)
    H28c = H28 - np.mean(H28, axis=0)
    p28c = pilot_concept - np.mean(H28, axis=0)
    s28 = (H28c / np.linalg.norm(H28c, axis=1, keepdims=True)) @ (p28c / np.linalg.norm(p28c))

    # MIX: Fact dominant
    s_mix = (np.clip(s5,0,1)*0.2) + (np.clip(s16,0,1)*0.5) + (np.clip(s28,0,1)*0.3)

    # --- C. RE-RANKING ---
    # 1. Get raw candidates (including errors)
    candidates = get_candidates(s_mix, offsets, text)
    if not candidates: return print("❌ No signal.")

    # 2. Judge them
    final_results = judge_candidates(query, candidates)

    # --- D. REPORT ---
    print(f"\n🏆 FINAL RANKING (Verified):")
    for i, hit in enumerate(final_results):
        # Highlight the winner with formatting
        marker = "✅ WINNER" if i == 0 else f"Rank #{i+1}"
        print(f"   [{marker}] Judge: {hit['judge_score']:.2f} | Radar: {hit['radar_score']:.2f} | Loc: {hit['idx']}")
        print(f"       \"...{hit['snippet']}...\"")
        print("-" * 40)

# -----------------------------
# 5. RUN
# -----------------------------

# 1. The President Test (Should promote the Rank #3 "President" hit over "Senator")
run_probe("How old must the President be?")

# 2. The Tax Test (Should downrank the Copyright Footer)
run_probe("Can the government take money from people?")