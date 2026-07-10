#include "model.h"
#include "vulkan_backend.h"
#include <cmath>
#include <algorithm>
#include <cstring>

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else {
            int e = -1;
            do { e++; mant <<= 1; } while (!(mant & 0x400));
            mant &= 0x3ff;
            f = (sign<<31) | ((uint32_t)(127-15-e)<<23) | (mant<<13);
        }
    } else if (exp == 31) {
        f = (sign<<31) | (0xff<<23) | (mant<<13);
    } else {
        f = (sign<<31) | ((exp+127-15)<<23) | (mant<<13);
    }
    float r; memcpy(&r, &f, 4); return r;
}

VkBuffer uploadTensor(VulkanBackend& vk, const GGUFTensor* t) {
    if (!t) return VK_NULL_HANDLE;
    VkBuffer buf = vk.createBuffer(t->nbytes(), 
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); 
    vk.uploadData(buf, t->data, t->nbytes());
    return buf;
}

void Model::load(GGUFFile& gf) {
    // (Unchanged, omitted for brevity)
    auto* arch = gf.get("general.architecture");
    cfg.arch = arch ? arch->s : "gpt-oss";
    std::string p = cfg.arch + ".";

    auto get = [&](const std::string& k, int def=0) -> int {
        auto* v = gf.get(k); return v ? v->i32() : def;
    };
    auto getf = [&](const std::string& k, float def=0) -> float {
        auto* v = gf.get(k); return v ? v->f32() : def;
    };
    auto getb = [&](const std::string& k, bool def=false) -> bool {
        auto* v = gf.get(k); return v ? v->b() : def;
    };

    cfg.n_layers   = get(p+"block_count");
    cfg.n_ctx      = 4096;
    cfg.n_embd     = get(p+"embedding_length");
    cfg.n_ff       = get(p+"feed_forward_length");
    cfg.n_head     = get(p+"attention.head_count");
    cfg.n_head_kv  = get(p+"attention.head_count_kv", cfg.n_head);
    cfg.head_dim   = get(p+"attention.key_length", 64);
    cfg.n_experts  = get(p+"expert_count", 0);
    cfg.n_experts_per_tok = get(p+"expert_used_count", 0);
    
    cfg.rope_freq_base = getf(p+"rope.freq_base", 150000.f); 
    cfg.rms_eps    = getf(p+"attention.layer_norm_rms_epsilon", 1e-6f);
    cfg.sliding_window = get(p+"attention.sliding_window", 0);
    
    cfg.attn_cap = getf(p+"attn_cap", 0.0f);
    cfg.final_cap = getf(p+"final_cap", 0.0f);
    cfg.q_pre_attn_scalar = getf(p+"q_pre_attn_scalar", 0.0f);
    cfg.vocab_size = get(p+"vocab_size", 0);
    cfg.bos_id     = get("tokenizer.ggml.bos_token_id", 0);
    cfg.eos_id     = get("tokenizer.ggml.eos_token_id", 0);
    cfg.add_bos    = getb("tokenizer.ggml.add_bos_token", false);

    cfg.rope_scaling_factor    = getf(p+"rope.scaling.factor", 32.0f);
    cfg.rope_ntk_alpha         = getf(p+"rope.ntk_alpha", 1.0f);
    cfg.rope_ntk_beta          = getf(p+"rope.ntk_beta", 32.0f);
    cfg.initial_context_length = get(p+"rope.scaling.original_context_length", 4096);

    token_embd  = gf.tensor("token_embd.weight");
    output_norm = gf.tensor("output_norm.weight");
    output      = gf.tensor("output.weight");

    if (cfg.n_layers == 0) {
        int max_blk = -1;
        for (const auto& t : gf.tensors()) {
            if (t.name.rfind("blk.", 0) == 0) {
                int idx = std::stoi(t.name.substr(4));
                if (idx > max_blk) max_blk = idx;
            }
        }
        cfg.n_layers = max_blk + 1;
    }
    if (cfg.n_embd == 0 && token_embd) cfg.n_embd = token_embd->ne[0];
    if (cfg.vocab_size == 0 && token_embd) cfg.vocab_size = token_embd->ne[1];

    const GGUFTensor* q_b = gf.tensor("blk.0.attn_q.bias");
    if (q_b) {
        if (cfg.n_head == 0) cfg.n_head = q_b->ne[0] / cfg.head_dim;
    }
    const GGUFTensor* k_b = gf.tensor("blk.0.attn_k.bias");
    if (k_b) {
        if (cfg.n_head_kv == 0) cfg.n_head_kv = k_b->ne[0] / cfg.head_dim;
    }
    const GGUFTensor* gate_inp = gf.tensor("blk.0.ffn_gate_inp.weight");
    if (gate_inp) {
        if (cfg.n_experts == 0) cfg.n_experts = gate_inp->ne[1];
        if (cfg.n_experts_per_tok == 0) cfg.n_experts_per_tok = cfg.n_experts;
    }
    const GGUFTensor* gate_exps = gf.tensor("blk.0.ffn_gate_exps.weight");
    if (gate_exps && cfg.n_ff == 0) cfg.n_ff = gate_exps->ne[1];

    if (cfg.n_ctx <= 0) cfg.n_ctx = 4096;
    if (cfg.n_embd == 0) cfg.n_embd = 2880;
    if (cfg.n_ff == 0) cfg.n_ff = 2880;
    if (cfg.n_layers == 0) cfg.n_layers = 24;
    if (cfg.head_dim == 0) cfg.head_dim = 64;
    if (cfg.n_head == 0) cfg.n_head = 64;
    if (cfg.n_head_kv == 0) cfg.n_head_kv = 8;
    if (cfg.n_experts == 0) cfg.n_experts = 4;
    if (cfg.n_experts_per_tok == 0) cfg.n_experts_per_tok = 4;

    layers.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; i++) {
        auto& L = layers[i];
        std::string b = "blk." + std::to_string(i) + ".";
        L.attn_norm = gf.tensor(b+"attn_norm.weight");
        L.post_attn_norm = gf.tensor(b+"post_attention_norm.weight");
        if (!L.post_attn_norm) L.post_attn_norm = gf.tensor(b+"ffn_norm.weight"); 
        
        L.q = gf.tensor(b+"attn_q.weight");
        L.k = gf.tensor(b+"attn_k.weight");
        L.v = gf.tensor(b+"attn_v.weight");
        L.o = gf.tensor(b+"attn_output.weight");
        
        L.q_bias = gf.tensor(b+"attn_q.bias");
        L.k_bias = gf.tensor(b+"attn_k.bias");
        L.v_bias = gf.tensor(b+"attn_v.bias");
        L.o_bias = gf.tensor(b+"attn_output.bias");
        
        L.attn_sinks = gf.tensor(b+"attn_sinks.weight");
        
        L.ffn_gate_inp = gf.tensor(b+"ffn_gate_inp.weight");
        L.ffn_gate_inp_bias = gf.tensor(b+"ffn_gate_inp.bias");
        L.gate_exps = gf.tensor(b+"ffn_gate_exps.weight");
        L.up_exps   = gf.tensor(b+"ffn_up_exps.weight");
        L.down_exps = gf.tensor(b+"ffn_down_exps.weight");
        L.gate_exps_bias = gf.tensor(b+"ffn_gate_exps.bias");
        L.up_exps_bias   = gf.tensor(b+"ffn_up_exps.bias");
        L.down_exps_bias = gf.tensor(b+"ffn_down_exps.bias");
    }

    is_moe = (cfg.n_experts > 0);
}

void Inference::init(Model* m, VulkanBackend* vk) {
    model = m;
    vk_backend = vk;
    auto& cfg = m->cfg;

    // All scratch buffers are strictly DEVICE_LOCAL now. No host visible mappings.
    h_buf = vk_backend->createBuffer(cfg.n_embd * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    normed_buf = vk_backend->createBuffer(cfg.n_embd * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    q_buf = vk_backend->createBuffer(cfg.n_head * cfg.head_dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    k_buf = vk_backend->createBuffer(cfg.n_head_kv * cfg.head_dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    v_buf = vk_backend->createBuffer(cfg.n_head_kv * cfg.head_dim * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    attn_out_buf = vk_backend->createBuffer(std::max(cfg.n_embd, cfg.n_head * cfg.head_dim) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tmp_buf = vk_backend->createBuffer(std::max(cfg.n_embd, cfg.n_ff) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    logits_buf = vk_backend->createBuffer(cfg.vocab_size * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    gate_out_buf = vk_backend->createBuffer(cfg.n_ff * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    up_out_buf = vk_backend->createBuffer(cfg.n_ff * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    down_out_buf = vk_backend->createBuffer(cfg.n_embd * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    router_buf = vk_backend->createBuffer(cfg.n_experts * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    moe_weights_buf = vk_backend->createBuffer(cfg.n_experts * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Batch all initial uploads via the ring buffer in a single command buffer
    vk_backend->beginCommandBuffer();

    token_embd_buf = uploadTensor(*vk_backend, m->token_embd);
    output_norm_buf = uploadTensor(*vk_backend, m->output_norm);
    output_buf = uploadTensor(*vk_backend, m->output);
    if (output_buf == VK_NULL_HANDLE) output_buf = token_embd_buf;

    vk_layers.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; i++) {
        auto& L = m->layers[i];
        auto& VL = vk_layers[i];
        VL.attn_norm = uploadTensor(*vk_backend, L.attn_norm);
        VL.post_attn_norm = uploadTensor(*vk_backend, L.post_attn_norm);
        VL.q = uploadTensor(*vk_backend, L.q);
        VL.k = uploadTensor(*vk_backend, L.k);
        VL.v = uploadTensor(*vk_backend, L.v);
        VL.o = uploadTensor(*vk_backend, L.o);
        VL.q_bias = uploadTensor(*vk_backend, L.q_bias);
        VL.k_bias = uploadTensor(*vk_backend, L.k_bias);
        VL.v_bias = uploadTensor(*vk_backend, L.v_bias);
        VL.o_bias = uploadTensor(*vk_backend, L.o_bias);
        VL.attn_sinks = uploadTensor(*vk_backend, L.attn_sinks);
        VL.ffn_gate_inp = uploadTensor(*vk_backend, L.ffn_gate_inp);
        VL.ffn_gate_inp_bias = uploadTensor(*vk_backend, L.ffn_gate_inp_bias);
        VL.gate_exps = uploadTensor(*vk_backend, L.gate_exps);
        VL.up_exps = uploadTensor(*vk_backend, L.up_exps);
        VL.down_exps = uploadTensor(*vk_backend, L.down_exps);
        VL.gate_exps_bias = uploadTensor(*vk_backend, L.gate_exps_bias);
        VL.up_exps_bias = uploadTensor(*vk_backend, L.up_exps_bias);
        VL.down_exps_bias = uploadTensor(*vk_backend, L.down_exps_bias);
    }

    kv_k_buf.resize(cfg.n_layers);
    kv_v_buf.resize(cfg.n_layers);
    size_t kv_size = (size_t)cfg.n_ctx * cfg.n_head_kv * cfg.head_dim * 4;
    for (int i = 0; i < cfg.n_layers; i++) {
        kv_k_buf[i] = vk_backend->createBuffer(kv_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        kv_v_buf[i] = vk_backend->createBuffer(kv_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vk_backend->fillBuffer(kv_k_buf[i], kv_size, 0);
        vk_backend->fillBuffer(kv_v_buf[i], kv_size, 0);
    }

    int hd = cfg.head_dim;
    int half = hd / 2;
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; i++) {
        inv_freq[i] = 1.0f / powf(cfg.rope_freq_base, (float)(2*i) / hd);
    }

    float concentration = 1.0f;
    if (cfg.rope_scaling_factor > 1.0f) {
        concentration = 0.1f * logf(cfg.rope_scaling_factor) + 1.0f;
        float low = (float)half * logf((float)cfg.initial_context_length / (cfg.rope_ntk_beta * 2.0f * (float)M_PI)) / logf(cfg.rope_freq_base);
        float high = (float)half * logf((float)cfg.initial_context_length / (cfg.rope_ntk_alpha * 2.0f * (float)M_PI)) / logf(cfg.rope_freq_base);
        for (int i = 0; i < half; i++) {
            float freq_extra = 1.0f / powf(cfg.rope_freq_base, (float)(2*i) / hd);
            float freq_inter = 1.0f / (cfg.rope_scaling_factor * powf(cfg.rope_freq_base, (float)(2*i) / hd));
            float ramp = ((float)i - low) / (high - low);
            float mask = 1.0f - std::max(0.0f, std::min(1.0f, ramp));
            inv_freq[i] = freq_inter * (1.0f - mask) + freq_extra * mask;
        }
    }

    cos_table.resize((size_t)cfg.n_ctx * half);
    sin_table.resize((size_t)cfg.n_ctx * half);
    for (int pos = 0; pos < cfg.n_ctx; pos++) {
        for (int i = 0; i < half; i++) {
            float angle = pos * inv_freq[i];
            cos_table[(size_t)pos*half + i] = cosf(angle) * concentration;
            sin_table[(size_t)pos*half + i] = sinf(angle) * concentration;
        }
    }

    cos_buf = vk_backend->createBuffer(cos_table.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    sin_buf = vk_backend->createBuffer(sin_table.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vk_backend->uploadData(cos_buf, cos_table.data(), cos_table.size() * 4);
    vk_backend->uploadData(sin_buf, sin_table.data(), sin_table.size() * 4);

    vk_backend->endAndSubmitCommandBuffer();
    vk_backend->waitFence();
}


void Inference::forward(int token, int pos) {
    // beginCommandBuffer handles waitFence for the previous token.
    // We record the entire forward pass into a single command buffer.
    vk_backend->beginCommandBuffer();

    auto& cfg = model->cfg;
    int hd = cfg.head_dim;
    int half = hd / 2;
    int n_embd = cfg.n_embd;

    auto get_matvec_shader = [&](const GGUFTensor* t, bool expert) -> VkShaderModule {
        if (!t) return expert ? matvec_q4_0_expert_shader : matvec_q4_0_shader;
        if (t->type == GT_F32) return expert ? matvec_f32_expert_shader : matvec_f32_shader;
        if (t->type == GT_F16) return expert ? matvec_f16_expert_shader : matvec_f16_shader;
        return expert ? matvec_q4_0_expert_shader : matvec_q4_0_shader;
    };

    struct RmsNormPC { int n; float eps; } rms_pc;
    struct MatVecPC { int M; int K; int has_bias; } matvec_pc;
    struct RoPEPC { int n_head; int head_dim; int pos; } rope_pc;
    struct CopyKVPC { int pos; int n_head_kv; int head_dim; } ckv_pc;
    struct AttnPC {
        int n_head; int n_head_kv; int head_dim; int pos;
        int sliding_window; int layer_idx; float scale; float attn_cap;
    } attn_pc;
    struct ExpertPC { int M; int K; int has_bias; int expert_idx; int M_per_expert; } exp_pc;
    struct AddPC { int n; float weight; } add_pc;
    struct MoeRouterPC { int n_experts; int top_k; } moe_router_pc;
    struct AddMoePC { int n; int expert_idx; } add_moe_pc;

    for (int li = 0; li < cfg.n_layers; li++) {
        auto& L = model->layers[li];
        auto& VL = vk_layers[li];

        if (li == 0) {
            // 1. Embedding lookup (GPU)
            struct EmbedPC { int token_id; int n_embd; int type; } embed_pc;
            embed_pc.token_id = token;
            embed_pc.n_embd   = n_embd;
            embed_pc.type     = (model->token_embd->type == GT_F32) ? 0 :
                                (model->token_embd->type == GT_F16) ? 1 : 2;
            vk_backend->dispatch(embed_lookup_shader, (n_embd + 31) / 32, 1, 1,
                                 { token_embd_buf, h_buf },
                                 &embed_pc, sizeof(EmbedPC));
            vk_backend->barrier({ h_buf });
        }

        // --- Attention RMSNorm ---
        rms_pc.n = n_embd; rms_pc.eps = cfg.rms_eps;
        vk_backend->dispatch(rmsnorm_shader, 1, 1, 1,
                             {h_buf, VL.attn_norm, normed_buf}, &rms_pc, sizeof(RmsNormPC));
        vk_backend->barrier({normed_buf});

        // --- Q, K, V projections ---
        matvec_pc.K = n_embd; matvec_pc.has_bias = 1;

        matvec_pc.M = cfg.n_head * hd;
        vk_backend->dispatch(get_matvec_shader(L.q, false), matvec_pc.M, 1, 1,
                             {VL.q, normed_buf, q_buf, VL.q_bias}, &matvec_pc, sizeof(MatVecPC));

        matvec_pc.M = cfg.n_head_kv * hd;
        vk_backend->dispatch(get_matvec_shader(L.k, false), matvec_pc.M, 1, 1,
                             {VL.k, normed_buf, k_buf, VL.k_bias}, &matvec_pc, sizeof(MatVecPC));
        vk_backend->dispatch(get_matvec_shader(L.v, false), matvec_pc.M, 1, 1,
                             {VL.v, normed_buf, v_buf, VL.v_bias}, &matvec_pc, sizeof(MatVecPC));
        vk_backend->barrier({q_buf, k_buf, v_buf});

        // --- RoPE on Q and K ---
        rope_pc.head_dim = hd; rope_pc.pos = pos;
        rope_pc.n_head = cfg.n_head;
        vk_backend->dispatch(rope_shader, rope_pc.n_head, 1, 1, {q_buf, cos_buf, sin_buf}, &rope_pc, sizeof(RoPEPC));
        rope_pc.n_head = cfg.n_head_kv;
        vk_backend->dispatch(rope_shader, rope_pc.n_head, 1, 1, {k_buf, cos_buf, sin_buf}, &rope_pc, sizeof(RoPEPC));
        vk_backend->barrier({q_buf, k_buf});

        // --- Copy K, V into KV cache ---
        ckv_pc.pos = pos; ckv_pc.n_head_kv = cfg.n_head_kv; ckv_pc.head_dim = hd;
        vk_backend->dispatch(copy_to_kv_shader, (cfg.n_head_kv * hd + 255)/256, 1, 1,
                             {k_buf, kv_k_buf[li]}, &ckv_pc, sizeof(CopyKVPC));
        vk_backend->dispatch(copy_to_kv_shader, (cfg.n_head_kv * hd + 255)/256, 1, 1,
                             {v_buf, kv_v_buf[li]}, &ckv_pc, sizeof(CopyKVPC));
        vk_backend->barrier({q_buf, kv_k_buf[li], kv_v_buf[li]});

        // --- Attention ---
        attn_pc.n_head = cfg.n_head; attn_pc.n_head_kv = cfg.n_head_kv;
        attn_pc.head_dim = hd; attn_pc.pos = pos;
        attn_pc.sliding_window = cfg.sliding_window; attn_pc.layer_idx = li;
        attn_pc.scale = cfg.q_pre_attn_scalar > 0.0f ? cfg.q_pre_attn_scalar : (1.0f / sqrtf((float)hd));
        attn_pc.attn_cap = cfg.attn_cap;
        vk_backend->dispatch(attention_shader, cfg.n_head, 1, 1,
                             {q_buf, kv_k_buf[li], kv_v_buf[li], VL.attn_sinks, attn_out_buf},
                             &attn_pc, sizeof(AttnPC));
        vk_backend->barrier({attn_out_buf});

        // --- Output projection ---
        matvec_pc.M = n_embd; matvec_pc.K = cfg.n_head * hd;
        vk_backend->dispatch(get_matvec_shader(L.o, false), matvec_pc.M, 1, 1,
                             {VL.o, attn_out_buf, tmp_buf, VL.o_bias}, &matvec_pc, sizeof(MatVecPC));
        vk_backend->barrier({tmp_buf});

        // --- Residual add ---
        add_pc.n = n_embd; add_pc.weight = 1.0f;
        vk_backend->dispatch(add_shader, (n_embd + 255)/256, 1, 1, {h_buf, tmp_buf}, &add_pc, sizeof(AddPC));
        vk_backend->barrier({h_buf});

        // --- FFN RMSNorm ---
        rms_pc.n = n_embd;
        vk_backend->dispatch(rmsnorm_shader, 1, 1, 1,
                             {h_buf, VL.post_attn_norm, normed_buf}, &rms_pc, sizeof(RmsNormPC));
        vk_backend->barrier({normed_buf});

        // --- Router ---
        matvec_pc.M = cfg.n_experts; matvec_pc.K = n_embd; matvec_pc.has_bias = 1;
        vk_backend->dispatch(get_matvec_shader(L.ffn_gate_inp, false), matvec_pc.M, 1, 1,
                             {VL.ffn_gate_inp, normed_buf, router_buf, VL.ffn_gate_inp_bias}, &matvec_pc, sizeof(MatVecPC));
        vk_backend->barrier({router_buf});

        // --- MoE routing (softmax + top-k) ---
        moe_router_pc.n_experts = cfg.n_experts;
        moe_router_pc.top_k = cfg.n_experts_per_tok;
        vk_backend->dispatch(moe_router_shader, 1, 1, 1,
                             {router_buf, moe_weights_buf}, &moe_router_pc, sizeof(MoeRouterPC));
        vk_backend->barrier({moe_weights_buf});

        // --- Zero accumulator ---
        vk_backend->fillBuffer(down_out_buf, n_embd * 4, 0);
        vk_backend->barrier({down_out_buf});

        // --- Expert loop (dispatch ALL experts, no readback required) ---
        exp_pc.K = n_embd; exp_pc.has_bias = 1; exp_pc.M_per_expert = cfg.n_ff;

        for (int e = 0; e < cfg.n_experts; e++) {
            // Gate + Up
            exp_pc.M = cfg.n_ff; exp_pc.expert_idx = e;
            vk_backend->dispatch(get_matvec_shader(L.gate_exps, true), exp_pc.M, 1, 1,
                                 {VL.gate_exps, normed_buf, gate_out_buf, VL.gate_exps_bias}, &exp_pc, sizeof(ExpertPC));
            vk_backend->dispatch(get_matvec_shader(L.up_exps, true), exp_pc.M, 1, 1,
                                 {VL.up_exps, normed_buf, up_out_buf, VL.up_exps_bias}, &exp_pc, sizeof(ExpertPC));
            vk_backend->barrier({gate_out_buf, up_out_buf});

            // SwiGLU
            int n_ff = cfg.n_ff;
            vk_backend->dispatch(swiglu_shader, (n_ff + 255)/256, 1, 1, {gate_out_buf, up_out_buf}, &n_ff, sizeof(int));
            vk_backend->barrier({gate_out_buf});

            // Down projection
            exp_pc.M = n_embd;
            exp_pc.K = cfg.n_ff;
            exp_pc.M_per_expert = n_embd;
            vk_backend->dispatch(get_matvec_shader(L.down_exps, true), exp_pc.M, 1, 1,
                                 {VL.down_exps, gate_out_buf, tmp_buf, VL.down_exps_bias}, &exp_pc, sizeof(ExpertPC));
            vk_backend->barrier({tmp_buf});

            // Accumulate into down_out_buf (add_moe reads moe_weights_buf, non-active are 0.0)
            add_moe_pc.n = n_embd; add_moe_pc.expert_idx = e;
            vk_backend->dispatch(add_moe_shader, (n_embd + 255)/256, 1, 1,
                                 {down_out_buf, tmp_buf, moe_weights_buf}, &add_moe_pc, sizeof(AddMoePC));

            // WAW barrier so the next expert can safely overwrite gate/up/tmp/down_out
            if (e < cfg.n_experts - 1) {
                vk_backend->barrier({gate_out_buf, up_out_buf, tmp_buf, down_out_buf});
            }
        }

        // --- Final residual add ---
        vk_backend->barrier({down_out_buf, normed_buf}); // normed_buf barrier for WAR safety on next layer
        add_pc.n = n_embd; add_pc.weight = 1.0f;
        vk_backend->dispatch(add_shader, (n_embd + 255)/256, 1, 1, {h_buf, down_out_buf}, &add_pc, sizeof(AddPC));
        vk_backend->barrier({h_buf});
    }

    // ================================================================
    // Final norm + output projection
    // ================================================================
    rms_pc.n = cfg.n_embd; rms_pc.eps = cfg.rms_eps;
    vk_backend->dispatch(rmsnorm_shader, 1, 1, 1,
                         {h_buf, output_norm_buf, normed_buf}, &rms_pc, sizeof(RmsNormPC));
    vk_backend->barrier({normed_buf});

    matvec_pc.M = cfg.vocab_size; matvec_pc.K = cfg.n_embd; matvec_pc.has_bias = 0;
    const GGUFTensor* out_tensor = model->output ? model->output : model->token_embd;
    VkShaderModule out_shader = matvec_q4_0_shader;
    if (out_tensor) {
        if (out_tensor->type == GT_F32) out_shader = matvec_f32_shader;
        else if (out_tensor->type == GT_F16) out_shader = matvec_f16_shader;
    }
    vk_backend->dispatch(out_shader, matvec_pc.M, 1, 1,
                         {output_buf, normed_buf, logits_buf, VK_NULL_HANDLE}, &matvec_pc, sizeof(MatVecPC));

    // Apply final tanh cap on the GPU (if enabled)
    if (cfg.final_cap > 0.0f) {
        vk_backend->barrier({ logits_buf });
        struct TanhCapPC { int n; float cap; } tc_pc;
        tc_pc.n   = cfg.vocab_size;
        tc_pc.cap = cfg.final_cap;
        vk_backend->dispatch(tanh_cap_shader,
                             (cfg.vocab_size + 255) / 256, 1, 1,
                             { logits_buf },
                             &tc_pc, sizeof(TanhCapPC));
    }

    // Double-buffered readback: copy logits to alternating offsets in the 4MB readback buffer
    vk_backend->barrier({logits_buf});
    VkDeviceSize logits_offset = readback_flip ? 0 : 1048576; // 1MB offset
    vk_backend->readbackData(logits_buf, logits_offset, cfg.vocab_size * 4);

    // Submit the entire token at once
    vk_backend->endAndSubmitCommandBuffer();
}

void Inference::get_logits(float* logits) {
    auto& cfg = model->cfg;
    
    vk_backend->waitFence();
    
    VkDeviceSize logits_offset = readback_flip ? 0 : 1048576;
    void* ptr = (uint8_t*)vk_backend->getReadbackPtr() + logits_offset;
    memcpy(logits, ptr, cfg.vocab_size * 4);

    readback_flip = 1 - readback_flip;
}