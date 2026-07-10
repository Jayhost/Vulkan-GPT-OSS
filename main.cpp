#include "gguf.h"
#include "model.h"
#include "tokenizer.h"
#include "vulkan_backend.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <vector>
#include <random>  

// Returns a uniform random float in [0, 1)
inline float rng_next() {
    static thread_local std::mt19937       rng{ std::random_device{}() };
    static thread_local std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}

struct SamplerArgs {
    float temp = 1.0f;
    int top_k = 0;       
    float top_p = 1.0f;  
    float repeat_penalty = 1.0f;
};

int sample(float* logits, int vocab_size, const SamplerArgs& sa,
           const std::vector<int>& recent_tokens, float rng_val) {
    if (sa.repeat_penalty != 1.0f && !recent_tokens.empty()) {
        for (int t : recent_tokens) {
            if (t >= 0 && t < vocab_size) {
                if (logits[t] > 0) logits[t] /= sa.repeat_penalty;
                else logits[t] *= sa.repeat_penalty;
            }
        }
    }

    if (sa.temp != 1.0f) {
        for (int i = 0; i < vocab_size; i++) logits[i] /= sa.temp;
    }

    if (sa.top_k > 0 && sa.top_k < vocab_size) {
        static std::vector<float> tmp;
        if ((int)tmp.size() != vocab_size) tmp.resize(vocab_size);
        memcpy(tmp.data(), logits, vocab_size * 4);
        std::nth_element(tmp.begin(), tmp.begin() + vocab_size - sa.top_k, tmp.end());
        float threshold = tmp[vocab_size - sa.top_k];
        for (int i = 0; i < vocab_size; i++)
            if (logits[i] < threshold) logits[i] = -1e30f;
    }

    float mx = logits[0];
    for (int i = 1; i < vocab_size; i++) mx = std::max(mx, logits[i]);
    
    float sum = 0;
    static std::vector<float> probs;
    if ((int)probs.size() != vocab_size) probs.resize(vocab_size);

    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf(logits[i] - mx);
        sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++) probs[i] *= inv_sum;

    if (sa.top_p < 1.0f) {
        static std::vector<int> idx;
        if ((int)idx.size() != vocab_size) idx.resize(vocab_size);
        for (int i = 0; i < vocab_size; i++) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });
        float cumsum = 0;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[idx[i]];
            if (cumsum > sa.top_p) {
                for (int j = i + 1; j < vocab_size; j++) probs[idx[j]] = 0;
                break;
            }
        }
        sum = 0;
        for (int i = 0; i < vocab_size; i++) sum += probs[i];
        inv_sum = 1.0f / sum;
        for (int i = 0; i < vocab_size; i++) probs[i] *= inv_sum;
    }

    float r = rng_val;
    float acc = 0;
    for (int i = 0; i < vocab_size; i++) {
        acc += probs[i];
        if (r <= acc) return i;
    }
    return vocab_size - 1;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> \"prompt\" [max_tokens=128] [temp=1.0] [top_k=0] [top_p=1.0]\n", argv[0]);
        return 1;
    }

    std::string model_path = argv[1];
    std::string raw_prompt = argv[2];
    int max_tokens = argc > 3 ? atoi(argv[3]) : 128;
    SamplerArgs sa;
    if (argc > 4) sa.temp = atof(argv[4]);
    if (argc > 5) sa.top_k = atoi(argv[5]);
    if (argc > 6) sa.top_p = atof(argv[6]);

    GGUFFile gf;
    if (!gf.load(model_path)) return 1;
    printf("GGUF loaded: %zu tensors, %zu KV pairs\n", gf.tensors().size(), gf.kv().size());

    Model model;
    model.load(gf);

    Tokenizer tok;
    if (!tok.load(gf)) return 1;

    VulkanBackend vk;
    vk.init();

    Inference infer;

    auto loadShader = [&](const char* path) -> VkShaderModule {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            fprintf(stderr, "Failed to open shader: %s\n", path);
            exit(1);
        }
        auto size = f.tellg();
        std::vector<uint32_t> spirv(size / 4);
        f.seekg(0);
        f.read((char*)spirv.data(), size);
        return vk.createShader(spirv);
    };

    printf("Loading shaders...\n");
    infer.rmsnorm_shader = loadShader("shaders/rmsnorm.spv");
    infer.matvec_f32_shader = loadShader("shaders/matvec_f32.spv");
    infer.matvec_f16_shader = loadShader("shaders/matvec_f16.spv");
    infer.matvec_q4_0_shader = loadShader("shaders/matvec_q4_0.spv");
    infer.matvec_f32_expert_shader = loadShader("shaders/matvec_f32_expert.spv");
    infer.matvec_f16_expert_shader = loadShader("shaders/matvec_f16_expert.spv");
    infer.matvec_q4_0_expert_shader = loadShader("shaders/matvec_q4_0_expert.spv");
    infer.rope_shader = loadShader("shaders/rope.spv");
    infer.copy_to_kv_shader = loadShader("shaders/copy_to_kv.spv");
    infer.attention_shader = loadShader("shaders/attention.spv");
    infer.swiglu_shader = loadShader("shaders/swiglu.spv");
    infer.add_shader = loadShader("shaders/add.spv");
    infer.moe_router_shader = loadShader("shaders/moe_router.spv");
    infer.add_moe_shader = loadShader("shaders/add_moe.spv");
    infer.embed_lookup_shader = loadShader("shaders/embed_lookup.spv");
    infer.tanh_cap_shader     = loadShader("shaders/tanh_cap.spv");
    infer.reduce_moe_shader = loadShader("shaders/reduce_moe.spv"); 

    infer.init(&model, &vk);

    std::string prompt = "<|im_start|>user\n" + raw_prompt + "<|im_end|>\n<|im_start|>assistant\n";
    
    std::vector<int> tokens = tok.encode(prompt);
    printf("Prompt tokens: %zu\n", tokens.size());
    printf("Tokens: ");
    for (int t : tokens) printf("%d ", t);
    printf("\n");

    std::vector<float> logits(model.cfg.vocab_size);

    // --- Prefill Loop ---
    // We don't need logits for intermediate prefill steps. Submit everything
    // back-to-back without blocking the CPU. waitFence() in forward() ensures
    // we don't overwrite scratch buffers still in use by the GPU.
    auto t0 = std::chrono::steady_clock::now();
    int pos = 0;
    for (int t : tokens) {
        infer.forward(t, pos);
        pos++;
    }
    // Wait for the very last prefill token to finish so we can read logits
    infer.get_logits(logits.data());

    auto t1 = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Prefill: %d tokens in %.0f ms (%.1f tok/s)\n",
           (int)tokens.size(), prefill_ms, tokens.size() / (prefill_ms/1000));

    printf("\n--- Output ---\n");
    std::vector<int> generated;
    std::vector<int> recent;

    // Sample first generated token immediately from prefill output
    int next = sample(logits.data(), model.cfg.vocab_size, sa, recent, rng_next());

        // --- Generation Loop ---
    for (int i = 0; i < max_tokens; i++) {
        if (next == model.cfg.eos_id) {
            printf("<EOS>\n");
            break;
        }

        auto cpu_start = std::chrono::steady_clock::now();
        
        // 1. Submit forward pass for the token. GPU starts working immediately.
        infer.forward(next, pos);
        pos++;

        auto cpu_end = std::chrono::steady_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // 2. Overlap CPU side string decode/print with GPU execution
        std::string piece = tok.decode({next});
        printf("%s", piece.c_str());
        fflush(stdout);

        generated.push_back(next);
        recent.push_back(next);
        if (recent.size() > 64) recent.erase(recent.begin());

        // 3. Block only now that we actually need the logits to sample the next token
        auto gpu_wait_start = std::chrono::steady_clock::now();
        infer.get_logits(logits.data());
        auto gpu_wait_end = std::chrono::steady_clock::now();
        double gpu_wait_ms = std::chrono::duration<double, std::milli>(gpu_wait_end - gpu_wait_start).count();

        // Print timing for the first 5 tokens to see the split
        // if (i < 5) {
        //     printf("\n[Token %d] CPU build: %.2f ms | GPU wait: %.2f ms\n", i, cpu_ms, gpu_wait_ms);
        // }

        next = sample(logits.data(), model.cfg.vocab_size, sa, recent, rng_next());
    }
    printf("\n");

    auto t2 = std::chrono::steady_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("\nGeneration: %d tokens in %.0f ms (%.1f tok/s)\n",
           (int)generated.size(), gen_ms, generated.size() / (gen_ms/1000));

    // Cleanup shaders to prevent ASAN memory leaks
    vk.destroyShader(infer.rmsnorm_shader);
    vk.destroyShader(infer.matvec_f32_shader);
    vk.destroyShader(infer.matvec_f16_shader);
    vk.destroyShader(infer.matvec_q4_0_shader);
    vk.destroyShader(infer.matvec_f32_expert_shader);
    vk.destroyShader(infer.matvec_f16_expert_shader);
    vk.destroyShader(infer.matvec_q4_0_expert_shader);
    vk.destroyShader(infer.rope_shader);
    vk.destroyShader(infer.copy_to_kv_shader);
    vk.destroyShader(infer.attention_shader);
    vk.destroyShader(infer.swiglu_shader);
    vk.destroyShader(infer.add_shader);
    vk.destroyShader(infer.moe_router_shader);
    vk.destroyShader(infer.add_moe_shader);
    vk.destroyShader(infer.embed_lookup_shader);
vk.destroyShader(infer.tanh_cap_shader);
vk.destroyShader(infer.reduce_moe_shader); 

    vk.cleanup();
    return 0;
}