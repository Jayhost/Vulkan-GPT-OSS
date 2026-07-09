#include "tokenizer.h"
#include <cctype>
#include <algorithm>
#include <cstring>
#include <regex>
#include <climits>

void Tokenizer::initByteEncoder() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }
    for (size_t i = 0; i < bs.size() && i < 256; i++) {
        int cp = cs[i];
        if (cp < 0x80) byte_encoder_[bs[i]] = std::string(1, (char)cp);
        else if (cp < 0x800) {
            byte_encoder_[bs[i]] = std::string{(char)(0xC0 | (cp >> 6)), (char)(0x80 | (cp & 0x3F))};
        } else {
            byte_encoder_[bs[i]] = std::string{(char)(0xE0 | (cp >> 12)),
                                                (char)(0x80 | ((cp >> 6) & 0x3F)),
                                                (char)(0x80 | (cp & 0x3F))};
        }
    }
}

bool Tokenizer::load(const GGUFFile& gf) {
    initByteEncoder();

    auto* model = gf.get("tokenizer.ggml.model");
    std::string model_type = model ? model->s : "gpt2";

    auto* tokens = gf.get("tokenizer.ggml.tokens");
    auto* types = gf.get("tokenizer.ggml.token_type");
    if (!tokens) { fprintf(stderr, "No tokenizer tokens in GGUF\n"); return false; }

    vocab_.resize(tokens->arr.size());
    for (size_t i = 0; i < tokens->arr.size(); i++) {
        vocab_[i] = tokens->arr[i].s;
        token_to_id_[tokens->arr[i].s] = i;
        int tt = types ? types->arr[i].i : 0;
        if (tt >= 2) {
            special_tokens_[tokens->arr[i].s] = i;
        }
    }
    vocab_size = vocab_.size();

    auto* merges = gf.get("tokenizer.ggml.merges");
    if (merges) {
        for (size_t i = 0; i < merges->arr.size(); i++) {
            std::string m = merges->arr[i].s;
            auto sp = m.find(' ');
            if (sp == std::string::npos) continue;
            std::string a = m.substr(0, sp);
            std::string b = m.substr(sp + 1);
            auto ia = token_to_id_.find(a);
            auto ib = token_to_id_.find(b);
            if (ia != token_to_id_.end() && ib != token_to_id_.end()) {
                merge_rank_[pairKey(ia->second, ib->second)] = i;
            }
        }
    }

    auto* bos = gf.get("tokenizer.ggml.bos_token_id");
    auto* eos = gf.get("tokenizer.ggml.eos_token_id");
    auto* ab = gf.get("tokenizer.ggml.add_bos_token");
    bos_id = bos ? bos->i32() : 0;
    eos_id = eos ? eos->i32() : 0;
    add_bos = ab ? ab->b() : false;

    // Initialize byte_token_id_ lookup array
    for (int b = 0; b < 256; b++) {
        byte_token_id_[b] = -1;
        auto it = token_to_id_.find(byte_encoder_[b]);
        if (it != token_to_id_.end()) {
            byte_token_id_[b] = it->second;
        }
    }

    printf("Tokenizer: %s, vocab=%d, merges=%zu, bos=%d, eos=%d, add_bos=%d\n",
           model_type.c_str(), vocab_size, merges ? merges->arr.size() : 0,
           bos_id, eos_id, add_bos);
    return true;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    std::vector<std::string> result;
    size_t i = 0;
    size_t n = text.size();

    auto isLetter = [](unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80; };
    auto isDigit  = [](unsigned char c) { return c >= '0' && c <= '9'; };
    auto isSpace  = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; };

    while (i < n) {
        if (text[i] == '\'' && i + 1 < n) {
            char c = text[i+1];
            if (c == 's' || c == 't' || c == 'm' || c == 'd') {
                result.push_back(text.substr(i, 2)); i += 2; continue;
            }
            if (i + 2 < n && ((c == 'r' && text[i+2] == 'e') ||
                              (c == 'v' && text[i+2] == 'e') ||
                              (c == 'l' && text[i+2] == 'l'))) {
                result.push_back(text.substr(i, 3)); i += 3; continue;
            }
        }

        size_t start = i;
        if (text[i] == ' ') i++;
        
        if (i < n && isLetter((unsigned char)text[i])) {
            while (i < n && isLetter((unsigned char)text[i])) i++;
            result.push_back(text.substr(start, i - start));
            continue;
        }
        if (i < n && isDigit((unsigned char)text[i])) {
            while (i < n && isDigit((unsigned char)text[i])) i++;
            result.push_back(text.substr(start, i - start));
            continue;
        }
        if (i < n && !isSpace((unsigned char)text[i])) {
            while (i < n && !isSpace((unsigned char)text[i]) &&
                   !isLetter((unsigned char)text[i]) && !isDigit((unsigned char)text[i])) i++;
            result.push_back(text.substr(start, i - start));
            continue;
        }
        
        // If we consumed a leading space but didn't find a word/digit/symbol to attach it to,
        // it means we hit the end of the string or more spaces. Group them together.
        if (i > start) {
            while (i < n && isSpace((unsigned char)text[i])) i++;
            result.push_back(text.substr(start, i - start));
            continue;
        }
        
        if (i < n && isSpace((unsigned char)text[i])) {
            while (i < n && isSpace((unsigned char)text[i])) i++;
            result.push_back(text.substr(start, i - start));
            continue;
        }
        
        i = start + 1;
    }
    return result;
}

std::vector<int> Tokenizer::bpe(const std::string& token) const {
    // Byte-encode the token string
    std::string encoded;
    for (unsigned char c : token) encoded += byte_encoder_[c];

    // Direct lookup first
    auto it = token_to_id_.find(encoded);
    if (it != token_to_id_.end()) return {it->second};

    // FIX: Correct BPE initialization. Must start with purely byte-level tokens.
    std::vector<int> ids;
    for (unsigned char c : token) {
        if (byte_token_id_[c] >= 0) {
            ids.push_back(byte_token_id_[c]);
        }
    }

    // Iteratively apply the lowest rank merge
    if (merge_rank_.empty()) return ids;

    while (ids.size() > 1) {
        int best_rank = INT_MAX;
        int best_idx = -1;
        for (size_t j = 0; j + 1 < ids.size(); j++) {
            auto it = merge_rank_.find(pairKey(ids[j], ids[j+1]));
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = j;
            }
        }
        if (best_idx < 0) break;

        std::string merged = vocab_[ids[best_idx]] + vocab_[ids[best_idx+1]];
        auto mit = token_to_id_.find(merged);
        if (mit == token_to_id_.end()) break;

        ids[best_idx] = mit->second;
        ids.erase(ids.begin() + best_idx + 1);
    }

    return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> result;

    if (add_bos) result.push_back(bos_id);

    size_t search_pos = 0;
    while (search_pos < text.size()) {
        bool found_special = false;
        for (auto& [tok, id] : special_tokens_) {
            if (search_pos + tok.size() <= text.size() &&
                text.compare(search_pos, tok.size(), tok) == 0) {
                result.push_back(id);
                search_pos += tok.size();
                found_special = true;
                break;
            }
        }
        if (found_special) continue;

        size_t next_special = text.size();
        for (auto& [tok, id] : special_tokens_) {
            size_t p = text.find(tok, search_pos);
            if (p != std::string::npos && p < next_special) next_special = p;
        }

        std::string segment = text.substr(search_pos, next_special - search_pos);
        auto pretokens = pretokenize(segment);
        for (auto& pt : pretokens) {
            auto ids = bpe(pt);
            for (auto id : ids) result.push_back(id);
        }
        search_pos = next_special;
    }

    return result;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    static std::unordered_map<std::string, uint8_t> byte_decoder;
    if (byte_decoder.empty()) {
        Tokenizer* self = const_cast<Tokenizer*>(this);
        for (int b = 0; b < 256; b++) {
            byte_decoder[self->byte_encoder_[b]] = (uint8_t)b;
        }
    }

    for (int id : ids) {
        if (id < 0 || id >= (int)vocab_.size()) continue;
        const std::string& tok = vocab_[id];
        size_t i = 0;
        while (i < tok.size()) {
            bool decoded = false;
            for (int l = std::min((size_t)3, tok.size() - i); l >= 1; l--) {
                std::string sub = tok.substr(i, l);
                auto it = byte_decoder.find(sub);
                if (it != byte_decoder.end()) {
                    result += (char)it->second;
                    i += l;
                    decoded = true;
                    break;
                }
            }
            if (!decoded) {
                result += tok[i];
                i++;
            }
        }
    }
    return result;
}