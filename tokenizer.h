#pragma once
#include "gguf.h"
#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    bool load(const GGUFFile& gf);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    int bos_id=0, eos_id=0;
    bool add_bos=false;
    int vocab_size=0;

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<uint64_t, int> merge_rank_;
    std::unordered_map<std::string, int> special_tokens_;

    std::string byte_encoder_[256];
    int byte_token_id_[256]; // Fast lookup for byte tokens
    void initByteEncoder();

    std::vector<int> bpe(const std::string& token) const;
    static uint64_t pairKey(int a, int b) { return (uint64_t)a << 32 | (uint64_t)b; }

    std::vector<std::string> pretokenize(const std::string& text) const;
};