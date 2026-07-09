#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

enum GGUFValueType : uint32_t {
    GV_UINT8=0, GV_INT8, GV_UINT16, GV_INT16, GV_UINT32, GV_INT32,
    GV_FLOAT32, GV_BOOL, GV_STRING, GV_ARRAY, GV_UINT64, GV_INT64, GV_FLOAT64
};

struct GGUFValue {
    GGUFValueType type;
    int64_t  i = 0;
    uint64_t u = 0;
    double   f = 0;
    std::string s;
    std::vector<GGUFValue> arr;
    GGUFValueType arr_type = GV_UINT8;

    // FIX: Handle unsigned 32-bit integers properly
    int32_t  i32() const { return type == GV_UINT32 ? (int32_t)u : (int32_t)i; }
    uint32_t u32() const { return type == GV_UINT32 ? (uint32_t)u : (uint32_t)i; }
    float    f32() const { return (float)f; }
    bool     b()   const { return i != 0; }
};

enum GGMLType : uint32_t { GT_F32=0, GT_F16=1, GT_Q4_0=2, GT_Q4_1=3, GT_Q5_0=6, GT_Q5_1=7, GT_Q8_0=8 };

struct GGUFTensor {
    std::string name;
    std::vector<uint64_t> ne;  // dimensions (ne[0] = innermost)
    uint32_t type;
    uint64_t offset;
    const uint8_t* data = nullptr;

    uint64_t nelem() const;
    uint64_t nbytes() const;
};

class GGUFFile {
public:
    bool load(const std::string& path);
    const GGUFValue* get(const std::string& key) const;
    const GGUFTensor* tensor(const std::string& name) const;
    const std::unordered_map<std::string, GGUFValue>& kv() const { return kv_; }
    const std::vector<GGUFTensor>& tensors() const { return tensors_; }

private:
    uint32_t version_ = 0;
    std::unordered_map<std::string, GGUFValue> kv_;
    std::vector<GGUFTensor> tensors_;
    std::unordered_map<std::string, size_t> tensor_idx_;
    std::vector<uint8_t> buf_;  // whole file in memory
    uint64_t data_offset_ = 0;

    bool readKV(std::ifstream& f);
    GGUFValue readValue(std::ifstream& f, GGUFValueType vt);
};