#pragma once
#include "gguf.h"
#include "vulkan_backend.h"
#include <vector>
#include <string>

float fp16_to_fp32(uint16_t h);
struct block_q4_0 { uint16_t d; uint8_t qs[16]; };

struct ModelConfig {
    std::string arch;
    int n_layers=0, n_ctx=0, n_embd=0, n_ff=0;
    int n_head=0, n_head_kv=0, head_dim=0;
    int n_experts=0, n_experts_per_tok=0;
    float rope_freq_base=150000.f, rms_eps=1e-5f;
    int sliding_window=0;
    float attn_cap=0.f, final_cap=0.f;
    float q_pre_attn_scalar=0.f;
    int vocab_size=0, bos_id=0, eos_id=0;
    bool add_bos=true;
    
    int initial_context_length=4096;
    float rope_scaling_factor=32.0f;
    float rope_ntk_alpha=1.0f;
    float rope_ntk_beta=32.0f;
};

struct LayerTensors {
    const GGUFTensor *attn_norm=nullptr, *post_attn_norm=nullptr;
    const GGUFTensor *q=nullptr, *k=nullptr, *v=nullptr, *o=nullptr;
    const GGUFTensor *q_bias=nullptr, *k_bias=nullptr, *v_bias=nullptr, *o_bias=nullptr;
    const GGUFTensor *attn_sinks=nullptr;
    const GGUFTensor *ffn_gate_inp=nullptr, *ffn_gate_inp_bias=nullptr;
    const GGUFTensor *gate_exps=nullptr, *up_exps=nullptr, *down_exps=nullptr;
    const GGUFTensor *gate_exps_bias=nullptr, *up_exps_bias=nullptr, *down_exps_bias=nullptr;
};

struct Model {
    ModelConfig cfg;
    const GGUFTensor *token_embd=nullptr;
    const GGUFTensor *output_norm=nullptr;
    const GGUFTensor *output=nullptr;
    std::vector<LayerTensors> layers;
    bool is_moe=false;

    void load(GGUFFile& gf);
};

struct KVCache {
    std::vector<float> k, v;
    int n_head_kv, head_dim, max_seq;
    void init(int nk, int hd, int ms) {
        n_head_kv=nk; head_dim=hd; max_seq=ms;
        k.resize((size_t)ms*nk*hd, 0);
        v.resize((size_t)ms*nk*hd, 0);
    }
    float* kptr(int pos) { return k.data() + (size_t)pos*n_head_kv*head_dim; }
    float* vptr(int pos) { return v.data() + (size_t)pos*n_head_kv*head_dim; }
};

struct Inference {
    Model* model;
    VulkanBackend* vk_backend = nullptr;
    std::vector<KVCache> kv; 
    std::vector<float> cos_table, sin_table;

    VkBuffer h_buf, normed_buf, q_buf, k_buf, v_buf, attn_out_buf, tmp_buf, logits_buf;
    VkBuffer gate_out_buf, up_out_buf, down_out_buf, router_buf;
    VkBuffer token_embd_buf, output_norm_buf, output_buf;
    VkBuffer cos_buf, sin_buf;
    std::vector<VkBuffer> kv_k_buf, kv_v_buf;

    struct VulkanLayerTensors {
        VkBuffer attn_norm, post_attn_norm, q, k, v, o;
        VkBuffer q_bias, k_bias, v_bias, o_bias, attn_sinks;
        VkBuffer ffn_gate_inp, ffn_gate_inp_bias;
        VkBuffer gate_exps, up_exps, down_exps;
        VkBuffer gate_exps_bias, up_exps_bias, down_exps_bias;
    };
    std::vector<VulkanLayerTensors> vk_layers;

    VkShaderModule rmsnorm_shader, matvec_f32_shader, matvec_f16_shader, matvec_q4_0_shader;
    VkShaderModule matvec_f32_expert_shader, matvec_f16_expert_shader, matvec_q4_0_expert_shader;
    VkShaderModule rope_shader, copy_to_kv_shader, attention_shader, swiglu_shader, add_shader;

    void init(Model* m, VulkanBackend* vk);
    void forward(int token, int pos, float* logits);
};