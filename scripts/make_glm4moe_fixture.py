#!/usr/bin/env python3
"""Generate the tiny synthetic `glm4moe` GGUF used by ferrox's GLM-4.5-MoE
refusal test.

`glm4moe` is what GLM-4.5, GLM-4.5-Air and GLM-4.6 tag. ferrox refuses it
today, which is right, but the refusal used to say "use
`ferrox_models::glm52_decoder` / `glm52_gguf_loader`" -- and that loader
cannot read a `glm4moe` file at all: `read_glm52_hparams` requires
`{arch}.attention.q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim` and
`qk_rope_head_dim`, and **GLM-4.5 is not an MLA model**. This fixture is
the proof: a file llama.cpp itself loads and runs as `glm4moe`, carrying
none of those four keys, because `src/models/glm4-moe.cpp`'s
`load_arch_hparams` never asks for them and its `load_arch_tensors` calls
`create_tensor_qkv` (plain Q/K/V) rather than creating any `attn_kv_a_mqa`
/ `attn_kv_b` / `attn_q_a` / `attn_q_b`.

The fixture is deliberately small (2 layers: one leading dense, one MoE;
32-wide, 6 experts) and carries the structure of `glm4-moe.cpp` that
decides where it can and cannot run:

  * `blk.N.post_attention_norm.weight` and **no** `blk.N.ffn_norm.weight`
    (glm4-moe.cpp:75). llama.cpp norms `ffn_inp` -- the post-residual
    sum -- with it (:215), i.e. it is the *pre-FFN* norm, gpt-oss's slot
    and not Gemma's. ferrox's generic decoder puts
    `post_attention_norm` in Gemma's slot (on the attention branch,
    before the residual add) and separately requires `ffn_norm`, so the
    generic path is a different graph AND cannot even find its tensors.
  * per-head Q/K RMSNorm of length `head_dim` (:69,71, optional there).
  * required Q/K/V biases via `create_tensor_qkv`, and no output bias.
  * a leading dense block, a shared expert, `exp_probs_b.bias`, sigmoid
    gating and `expert_weights_scale` -- the DeepSeek-V3-shaped routing
    ferrox already validates on `dots1`.
  * partial RoPE: `rope.dimension_count` is half `head_dim`, GLM's
    `partial_rotary_factor = 0.5`.

Weights are pseudo-random from a fixed seed so the file is byte-stable.

Usage:
    PYTHONPATH=/path/to/llama.cpp/gguf-py \\
        python3 scripts/make_glm4moe_fixture.py OUT.gguf

That the file is a *valid* glm4moe checkpoint was checked by running
llama.cpp's own loader over it (`scripts/gptoss_reference_logits.cpp`
against a real `libllama`): it loads, prints `n_expert = 6` /
`rope type = 2`, and decodes. No golden logits are checked in, because
ferrox has no glm4moe graph to compare against yet -- see
`crates/ferrox-models/tests/glm4moe_refusal.rs`.
"""

import sys

import numpy as np

import gguf

ARCH = "glm4moe"

N_LAYER = 2
N_DENSE_LEAD = 1
N_EMBD = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 16
# GLM's partial_rotary_factor = 0.5: only the first half of each head is
# rotated.
ROPE_DIM = HEAD_DIM // 2
N_EXPERT = 6
N_EXPERT_USED = 2
N_EXPERT_SHARED = 1
N_FF = 40
N_FF_EXP = 16
N_VOCAB = 48
CTX = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5
EXPERT_WEIGHTS_SCALE = 2.5


def main(out_path: str) -> None:
    rng = np.random.default_rng(0x64114)

    def rnd(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.25).astype(np.float32)

    w = gguf.GGUFWriter(out_path, ARCH)
    w.add_name("ferrox-glm4moe-fixture")
    w.add_block_count(N_LAYER)
    w.add_context_length(CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_freq_base(ROPE_BASE)
    w.add_rope_dimension_count(ROPE_DIM)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_shared_count(N_EXPERT_SHARED)
    w.add_expert_feed_forward_length(N_FF_EXP)
    w.add_leading_dense_block_count(N_DENSE_LEAD)
    w.add_expert_weights_scale(EXPERT_WEIGHTS_SCALE)
    w.add_expert_weights_norm(True)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    # Deliberately absent, and that absence is the point of the fixture:
    # `{arch}.attention.q_lora_rank`, `.kv_lora_rank`,
    # `.qk_nope_head_dim`, `.qk_rope_head_dim`. GLM-4.5 is plain GQA.

    tokens = ["<unk>", "<s>", "</s>"] + [f"tok{i}" for i in range(3, N_VOCAB)]
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores([0.0] * N_VOCAB)
    types = [gguf.TokenType.CONTROL if i < 3 else gguf.TokenType.NORMAL for i in range(N_VOCAB)]
    w.add_token_types([int(t) for t in types])
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)

    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))

    n_embd_q = N_HEAD * HEAD_DIM
    n_embd_kv = N_HEAD_KV * HEAD_DIM

    for il in range(N_LAYER):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", rnd(N_EMBD))

        w.add_tensor(p + "attn_q.weight", rnd(n_embd_q, N_EMBD))
        w.add_tensor(p + "attn_k.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_v.weight", rnd(n_embd_kv, N_EMBD))
        w.add_tensor(p + "attn_q.bias", rnd(n_embd_q))
        w.add_tensor(p + "attn_k.bias", rnd(n_embd_kv))
        w.add_tensor(p + "attn_v.bias", rnd(n_embd_kv))
        w.add_tensor(p + "attn_q_norm.weight", rnd(HEAD_DIM))
        w.add_tensor(p + "attn_k_norm.weight", rnd(HEAD_DIM))

        w.add_tensor(p + "attn_output.weight", rnd(N_EMBD, n_embd_q))

        # The pre-FFN norm, spelled `post_attention_norm`. There is no
        # `ffn_norm` in a glm4moe checkpoint.
        w.add_tensor(p + "post_attention_norm.weight", rnd(N_EMBD))

        if il < N_DENSE_LEAD:
            w.add_tensor(p + "ffn_gate.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_up.weight", rnd(N_FF, N_EMBD))
            w.add_tensor(p + "ffn_down.weight", rnd(N_EMBD, N_FF))
            continue

        w.add_tensor(p + "ffn_gate_inp.weight", rnd(N_EXPERT, N_EMBD))
        w.add_tensor(
            p + "exp_probs_b.bias",
            (rng.standard_normal(N_EXPERT) * 0.6).astype(np.float32),
        )

        w.add_tensor(p + "ffn_gate_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(N_EXPERT, N_FF_EXP, N_EMBD))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(N_EXPERT, N_EMBD, N_FF_EXP))

        n_ff_sh = N_FF_EXP * N_EXPERT_SHARED
        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(n_ff_sh, N_EMBD))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(n_ff_sh, N_EMBD))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(N_EMBD, n_ff_sh))

    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "glm4moe-fixture.gguf")
