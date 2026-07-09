#include "gguf.h"
#include <fstream>
#include <cstring>
#include <cassert>

uint64_t GGUFTensor::nelem() const {
    uint64_t n = 1;
    for (auto d : ne) n *= d;
    return n;
}

uint64_t GGUFTensor::nbytes() const {
    uint64_t n = nelem();
    switch (type) {
        case GT_F32: return n * 4;
        case GT_F16: return n * 2;
        case GT_Q4_0: return (n / 32) * 18;
        case GT_Q4_1: return (n / 32) * 20;
        case GT_Q5_0: return (n / 32) * 22;
        case GT_Q5_1: return (n / 32) * 24;
        case GT_Q8_0: return (n / 32) * 34;
        default: return n * 4;
    }
}

static std::string readStr(std::ifstream& f) {
    uint64_t n; f.read((char*)&n, 8);
    std::string s(n, '\0');
    f.read(s.data(), n);
    return s;
}

GGUFValue GGUFFile::readValue(std::ifstream& f, GGUFValueType vt) {
    GGUFValue v; v.type = vt;
    switch (vt) {
        case GV_UINT8:  { uint8_t x;  f.read((char*)&x,1); v.u=x; break; }
        case GV_INT8:   { int8_t x;   f.read((char*)&x,1); v.i=x; break; }
        case GV_UINT16: { uint16_t x; f.read((char*)&x,2); v.u=x; break; }
        case GV_INT16:  { int16_t x;  f.read((char*)&x,2); v.i=x; break; }
        case GV_UINT32: { uint32_t x; f.read((char*)&x,4); v.u=x; break; }
        case GV_INT32:  { int32_t x;  f.read((char*)&x,4); v.i=x; break; }
        case GV_FLOAT32:{ float x;    f.read((char*)&x,4); v.f=x; break; }
        case GV_BOOL:   { uint8_t x;  f.read((char*)&x,1); v.i=x; break; }
        case GV_STRING: { v.s = readStr(f); break; }
        case GV_ARRAY: {
            uint32_t at; f.read((char*)&at, 4);
            uint64_t al; f.read((char*)&al, 8);
            v.arr_type = (GGUFValueType)at;
            v.arr.reserve(al);
            for (uint64_t i = 0; i < al; i++)
                v.arr.push_back(readValue(f, (GGUFValueType)at));
            break;
        }
        case GV_UINT64: { uint64_t x; f.read((char*)&x,8); v.u=x; break; }
        case GV_INT64:  { int64_t x;  f.read((char*)&x,8); v.i=x; break; }
        case GV_FLOAT64:{ double x;   f.read((char*)&x,8); v.f=x; break; }
    }
    return v;
}

bool GGUFFile::readKV(std::ifstream& f) {
    uint64_t nkv; f.read((char*)&nkv, 8);
    for (uint64_t i = 0; i < nkv; i++) {
        std::string key = readStr(f);
        uint32_t vt; f.read((char*)&vt, 4);
        kv_[key] = readValue(f, (GGUFValueType)vt);
    }
    return true;
}

bool GGUFFile::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); return false; }

    auto fsize = f.tellg();
    f.seekg(0);

    char magic[4]; f.read(magic, 4);
    if (memcmp(magic, "GGUF", 4) != 0) { fprintf(stderr, "Not GGUF\n"); return false; }

    f.read((char*)&version_, 4);
    uint64_t ntensor, nkv;
    f.read((char*)&ntensor, 8);
    f.read((char*)&nkv, 8);

    // Read KV metadata
    for (uint64_t i = 0; i < nkv; i++) {
        std::string key = readStr(f);
        uint32_t vt; f.read((char*)&vt, 4);
        kv_[key] = readValue(f, (GGUFValueType)vt);
    }

    // Read tensor info
    tensors_.resize(ntensor);
    for (uint64_t i = 0; i < ntensor; i++) {
        auto& t = tensors_[i];
        t.name = readStr(f);
        uint32_t ndim; f.read((char*)&ndim, 4);
        t.ne.resize(ndim);
        for (uint32_t d = 0; d < ndim; d++) f.read((char*)&t.ne[d], 8);
        f.read((char*)&t.type, 4);
        f.read((char*)&t.offset, 8);
        tensor_idx_[t.name] = i;
    }

    // Align to 32 bytes
    auto pos = f.tellg();
    uint64_t aligned = ((uint64_t)pos + 31) & ~31ULL;
    f.seekg(aligned);
    data_offset_ = aligned;

    // Read rest into buffer
    auto remaining = fsize - (std::streamoff)aligned;
    buf_.resize(remaining);
    f.read((char*)buf_.data(), remaining);

    // Set data pointers
    for (auto& t : tensors_)
        t.data = buf_.data() + t.offset;

    return true;
}

const GGUFValue* GGUFFile::get(const std::string& key) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? nullptr : &it->second;
}

const GGUFTensor* GGUFFile::tensor(const std::string& name) const {
    auto it = tensor_idx_.find(name);
    if (it == tensor_idx_.end()) return nullptr;
    return &tensors_[it->second];
}