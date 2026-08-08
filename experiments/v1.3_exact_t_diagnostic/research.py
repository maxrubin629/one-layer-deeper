"""Research-only exact-T controller diagnostic; blocked from submission."""

from __future__ import annotations

import argparse
import json
import time
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state


D_MODEL = 128
NUM_HEADS = 4
MAX_LOOPS = 64
N_TOKEN, X_TOKEN, T_TOKEN, ANS_TOKEN, DIGIT_OFFSET = 2, 3, 4, 5, 7


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class EncoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=x.device, dtype=torch.bool)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


def field_masks(input_ids: Tensor, attention_mask: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    valid = input_ids.ne(0) if attention_mask is None else attention_mask.bool()
    digits = input_ids.ge(DIGIT_OFFSET) & input_ids.lt(DIGIT_OFFSET + 10) & valid
    seen_n = input_ids.eq(N_TOKEN).cumsum(1).gt(0)
    seen_x = input_ids.eq(X_TOKEN).cumsum(1).gt(0)
    seen_t = input_ids.eq(T_TOKEN).cumsum(1).gt(0)
    seen_answer = input_ids.eq(ANS_TOKEN).cumsum(1).gt(0)
    return digits & seen_n & ~seen_x, digits & seen_x & ~seen_t, digits & seen_t & ~seen_answer


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def parse_exact_t(input_ids: Tensor, t_mask: Tensor) -> Tensor:
    """Parse decimal T on the current device. This makes the file research-only."""

    depth = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    for position in range(input_ids.shape[1]):
        digit = (input_ids[:, position] - DIGIT_OFFSET).clamp(0, 9)
        depth = torch.where(t_mask[:, position], depth * 10 + digit, depth)
    return depth.clamp(0, MAX_LOOPS)


class ContextualGELUTransition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = RMSNorm(D_MODEL)
        self.context = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)
        self.residual_scale = nn.Parameter(torch.empty(()))
        nn.init.uniform_(self.residual_scale, 0.2, 0.3)

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        features = self.norm(state) + self.context(context)
        return state + self.residual_scale.tanh() * self.down(F.gelu(self.up(features)))


class Model(nn.Module):
    num_loops = MAX_LOOPS

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.encoder = EncoderBlock()
        self.initial_projection = nn.Linear(2 * D_MODEL, D_MODEL)
        self.context_projection = nn.Linear(D_MODEL, D_MODEL)
        self.transition = ContextualGELUTransition()
        self.output_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        token_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        token_states = self.encoder(token_states, attention_mask)
        n_mask, x_mask, t_mask = field_masks(input_ids, attention_mask)
        n_state = masked_mean(token_states, n_mask)
        x_state = masked_mean(token_states, x_mask)
        context = self.context_projection(n_state)
        state = self.initial_projection(torch.cat((n_state, x_state), -1))
        exact_t = parse_exact_t(input_ids, t_mask)
        for step in range(MAX_LOOPS):
            proposal = self.transition(state, context)
            state = torch.where(exact_t.gt(step).unsqueeze(-1), proposal, state)
        logits = self.head(self.output_norm(token_states + state.unsqueeze(1)))
        return logits, {"exact_t": exact_t}


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    ))


SUBMISSION = Submission(build_model=build_model, build_optimizer=build_optimizer)


def run_diagnostic(device: str = "cpu") -> dict[str, object]:
    torch_device = torch.device(device)
    torch.manual_seed(74)
    input_ids = torch.tensor(
        [
            [2, 8, 10, 3, 9, 4, 8, 0, 0, 0],
            [2, 8, 11, 10, 3, 12, 4, 13, 11, 0],
        ],
        dtype=torch.long,
        device=torch_device,
    )
    attention_mask = input_ids.ne(0)
    model = build_model(ModelSpec(17, 10, 500_000_000)).to(torch_device).eval()
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits, auxiliary = model(input_ids, attention_mask)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    return {
        "experiment_id": "v1.3_exact_t_diagnostic",
        "eligibility": "research-only",
        "device": str(torch_device),
        "parsed_exact_t": auxiliary["exact_t"].tolist(),
        "maximum_masked_steps": MAX_LOOPS,
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "model_state_elements": sum(
            tensor.numel() for tensor in model.state_dict().values()
        ),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "torch_version": torch.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_diagnostic(args.device), sort_keys=True))


if __name__ == "__main__":
    main()
