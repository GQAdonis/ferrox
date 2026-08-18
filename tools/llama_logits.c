// Reference logits dumper: llama.cpp's own last-position logits for an
// EXPLICIT token-id sequence, written as raw f32 to a file.
//
// Why token ids and not text: `ferrox` and llama.cpp were once compared
// by greedy text, and it proved nothing — the two tokenizers had to agree
// first, and once they did the sequences still diverged after ~3 tokens
// from ordinary numeric drift flipping a single argmax. Text is the wrong
// medium. This dumps the logit vector at one position, so the comparison
// is over a distribution instead of over one sampled draw, and the
// tokenizer is removed from the experiment entirely by passing the ids in.
//
// Build (macOS, Homebrew llama.cpp):
//   ./tools/build_llama_logits.sh
//
// or by hand (macOS, Homebrew llama.cpp):
//   P=$(brew --prefix llama.cpp)
//   cc -std=c11 -O2 -I"$P/include" -L"$P/lib" -lllama \
//       -Wl,-rpath,"$P/lib" tools/llama_logits.c -o target/llama_logits
//
// Usage:
//   llama_logits <model.gguf> <out.bin> <tok0> <tok1> ...
// Writes n_vocab f32 to <out.bin> and prints "n_vocab <N>" on stdout.
//
// CPU only (n_gpu_layers = 0): the ferrox side of the comparison is its
// CPU path, which is the one cross-validated against NumPy. Comparing a
// GPU reference against a CPU candidate would confound two differences.

#include "llama.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bin> <tok0> [tok1 ...]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];

    const int n_tokens = argc - 3;
    if (n_tokens <= 0) {
        fprintf(stderr, "no tokens given\n");
        return 2;
    }
    llama_token * tokens = (llama_token *) malloc(sizeof(llama_token) * (size_t) n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        tokens[i] = (llama_token) atoi(argv[i + 3]);
    }

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }

    struct llama_context_params cparams = llama_context_default_params();
    // Context and batch must both cover the whole prompt: this decodes it
    // in ONE call, because splitting it would change the batch shape and
    // therefore which kernels llama.cpp picks — the thing under test.
    cparams.n_ctx    = (uint32_t) n_tokens + 8;
    cparams.n_batch  = (uint32_t) n_tokens;
    cparams.n_ubatch = (uint32_t) n_tokens;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    struct llama_batch batch = llama_batch_init((int32_t) n_tokens, 0, 1);
    for (size_t i = 0; i < (size_t) n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        // Only the last position's logits are wanted; asking for all of
        // them would allocate n_tokens*n_vocab floats for nothing.
        batch.logits[i]    = (i + 1 == (size_t) n_tokens);
    }
    batch.n_tokens = (int32_t) n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    const float * logits = llama_get_logits_ith(ctx, (int32_t) n_tokens - 1);
    if (!logits) {
        fprintf(stderr, "llama_get_logits_ith returned NULL\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    FILE * f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s for writing\n", out_path);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    fwrite(logits, sizeof(float), (size_t) n_vocab, f);
    fclose(f);

    printf("n_vocab %d\n", n_vocab);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
