import torch
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


class FrameSession:
    def __init__(self, model_id, model, tokenizer, device):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.input_ids = None
        self.past_key_values = None
        self._next_logits = None

    @torch.no_grad()
    def ingest(self, text: str):
        token_ids = self.tokenizer(
            text, return_tensors="pt"
        ).input_ids.to(self.device)

        self._forward(token_ids)
        return token_ids.shape[1]

    @torch.no_grad()
    def continue_token(self, token_id: int):
        token_ids = torch.tensor([[token_id]], device=self.device)
        self._forward(token_ids)

    def _forward(self, token_ids: torch.Tensor):
        outputs = self.model(
            input_ids=token_ids,
            past_key_values=self.past_key_values,
            use_cache=True,
            return_dict=True
        )

        self.past_key_values = outputs.past_key_values
        self._next_logits = outputs.logits[:, -1, :].detach()

        if self.input_ids is None:
            self.input_ids = token_ids
        else:
            self.input_ids = torch.cat([self.input_ids, token_ids], dim=-1)

    def next_token_argmax(self) -> int:
        return int(torch.argmax(self._next_logits[0]).item())

    def save(self, path: str):
        torch.save(
            {
                "model_id": self.model_id,
                "input_ids": self.input_ids,
                "past_key_values": self.past_key_values,
                "_next_logits": self._next_logits,
            },
            path
        )

    @classmethod
    def load_into(cls, path, model, tokenizer, device):
        data = torch.load(path, map_location=device, weights_only=False)

        fs = cls(
            model_id=data["model_id"],
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        fs.input_ids = data["input_ids"].to(device)
        fs._next_logits = data["_next_logits"].to(device)
        fs.past_key_values = cls._move_recursive(
            data["past_key_values"], device
        )

        return fs

    @staticmethod
    def _move_recursive(data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, tuple):
            return tuple(FrameSession._move_recursive(x, device) for x in data)
        elif isinstance(data, list):
            return [FrameSession._move_recursive(x, device) for x in data]
        return data


def generate_ids(fs: FrameSession, n: int):
    out = []
    t0 = time.perf_counter()

    for _ in range(n):
        tid = fs.next_token_argmax()
        out.append(tid)
        fs.continue_token(tid)

    return out, time.perf_counter() - t0


def main():
    model_id = "microsoft/phi-2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading model ONCE on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)
    model.eval()

    base_text = (
        "The history of the Roman Empire is long and complex. "
        "It began as a small city-state in Italy and grew to control "
        "the entire Mediterranean basin. "
    )
    prompt = base_text * 60

    print("\n[1] Ingesting document...")
    fs = FrameSession(model_id, model, tokenizer, device)

    t0 = time.perf_counter()
    num_tokens = fs.ingest(prompt)
    ingest_time = time.perf_counter() - t0

    print(f"Tokens: {num_tokens}")
    print(f"Ingest time: {ingest_time:.2f}s")
    print(f"Speed: {num_tokens / ingest_time:.2f} tok/s")

    print("\n[2] Saving FrameSession cache...")
    t0 = time.perf_counter()
    fs.save("heavy_bench.framesession")
    save_time = time.perf_counter() - t0

    size_mb = os.path.getsize("heavy_bench.framesession") / (1024 * 1024)
    print(f"Cache size: {size_mb:.2f} MB")
    print(f"Save time: {save_time:.3f}s")

    del fs

    print("\n[3] Loading cache ONLY (model already hot)...")
    t0 = time.perf_counter()
    fs_loaded = FrameSession.load_into(
        "heavy_bench.framesession",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    load_time = time.perf_counter() - t0

    print(f"Cache load time: {load_time:.3f}s")

    print("\n[4] Generating 10 tokens...")
    ids, gen_time = generate_ids(fs_loaded, 10)

    print(f"Generation time: {gen_time:.3f}s")
    print(f"Speed: {10 / gen_time:.2f} tok/s")
    print(f"Output: ...{tokenizer.decode(ids)}")

    print("\n========================================")
    print(" FINAL RESULTS ")
    print("========================================")
    print(f"Ingest compute: {ingest_time:.2f}s")
    print(f"Cache hydrate:  {load_time:.3f}s")
    print(f"Speedup:        {ingest_time / load_time:.1f}x")

    os.remove("heavy_bench.framesession")


if __name__ == "__main__":
    main()
