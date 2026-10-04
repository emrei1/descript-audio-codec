// Minimal read-only GGUF loader (F32 tensors, scalar and int-array metadata).
#include "model.h"

#include <cstring>

namespace dacn {
namespace {

enum : uint32_t { T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 };

size_t scalar_size(uint32_t t) {
    switch (t) {
        case T_U8: case T_I8: case T_BOOL: return 1;
        case T_U16: case T_I16: return 2;
        case T_U32: case T_I32: case T_F32: return 4;
        case T_U64: case T_I64: case T_F64: return 8;
        default: return 0;
    }
}

struct reader {
    const uint8_t * p;
    size_t n, off = 0;
    void need(size_t k) { if (k > n - off) throw error("corrupt model file (truncated header)"); }
    template <class T> T get() { T v; need(sizeof v); memcpy(&v, p + off, sizeof v); off += sizeof v; return v; }
    std::string str() {
        const uint64_t len = get<uint64_t>();
        need((size_t) len);
        std::string s((const char *) p + off, (size_t) len);
        off += (size_t) len;
        return s;
    }
};

}  // namespace

bool model::load(const fs::path & path, std::string & err) {
    try {
        FILE * f = open_file(path, "rb");
        if (!f) throw error("cannot open model " + path.u8string());
        fseek(f, 0, SEEK_END);
#ifdef _WIN32
        const int64_t size = _ftelli64(f);
#else
        const int64_t size = (int64_t) ftello(f);
#endif
        fseek(f, 0, SEEK_SET);
        if (size < 64) { fclose(f); throw error("model file too small"); }
        buf_ = alloc_floats((size_t) size / sizeof(float) + 1);
        uint8_t * base = (uint8_t *) buf_.get();
        const size_t got = fread(base, 1, (size_t) size, f);
        fclose(f);
        if (got != (size_t) size) throw error("cannot read model file");

        reader r{base, (size_t) size};
        if (r.get<uint32_t>() != 0x46554747u) throw error("not a GGUF file");
        if (r.get<uint32_t>() < 2) throw error("unsupported GGUF version");
        const uint64_t n_tensors = r.get<uint64_t>(), n_kv = r.get<uint64_t>();
        uint32_t alignment = 32;
        std::map<std::string, uint64_t> u;
        std::map<std::string, std::vector<int>> arr;
        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = r.str();
            const uint32_t t = r.get<uint32_t>();
            if (t == T_STR) { r.str(); continue; }
            if (t == T_ARR) {
                const uint32_t et = r.get<uint32_t>();
                const uint64_t cnt = r.get<uint64_t>();
                if (et == T_STR) { for (uint64_t k = 0; k < cnt; ++k) r.str(); continue; }
                const size_t es = scalar_size(et);
                if (!es || cnt > (r.n - r.off) / es) throw error("corrupt model metadata");
                if (et == T_I32 || et == T_U32) {
                    std::vector<int> v((size_t) cnt);
                    memcpy(v.data(), r.p + r.off, (size_t) cnt * 4);
                    arr[key] = v;
                }
                r.off += (size_t) (cnt * es);
                continue;
            }
            const size_t es = scalar_size(t);
            if (!es) throw error("corrupt model metadata");
            uint64_t v = 0;
            r.need(es);
            memcpy(&v, r.p + r.off, es);
            r.off += es;
            u[key] = v;
            if (key == "general.alignment" && v) alignment = (uint32_t) v;
        }
        struct info { std::string name; std::vector<int64_t> ne; uint32_t type; uint64_t off; };
        std::vector<info> infos;
        for (uint64_t i = 0; i < n_tensors; ++i) {
            info ti;
            ti.name = r.str();
            const uint32_t nd = r.get<uint32_t>();
            if (nd > 4) throw error("corrupt tensor info");
            for (uint32_t d = 0; d < nd; ++d) ti.ne.push_back((int64_t) r.get<uint64_t>());
            ti.type = r.get<uint32_t>();
            ti.off = r.get<uint64_t>();
            infos.push_back(std::move(ti));
        }
        const size_t data_off = (r.off + alignment - 1) / alignment * alignment;
        for (auto & ti : infos) {
            if (ti.type != 0) throw error("tensor " + ti.name + " is not F32");
            size_t cnt = 1;
            for (int64_t d : ti.ne) cnt *= (size_t) d;
            if (data_off + ti.off + cnt * 4 > (size_t) size) throw error("tensor " + ti.name + " out of bounds");
            if ((data_off + ti.off) % 4) throw error("misaligned tensor " + ti.name);
            tensors_[ti.name] = tensor{ti.ne, (const float *) (base + data_off + ti.off), cnt};
        }
        auto U = [&](const char * k, int d) { auto it = u.find(std::string("dac.") + k); return it == u.end() ? d : (int) it->second; };
        sample_rate = U("sample_rate", 44100);
        n_codebooks = U("n_codebooks", 9);
        codebook_size = U("codebook_size", 1024);
        codebook_dim = U("codebook_dim", 8);
        latent_dim = U("latent_dim", 1024);
        encoder_dim = U("encoder_dim", 64);
        decoder_dim = U("decoder_dim", 1536);
        if (arr.count("dac.encoder_rates")) encoder_rates = arr["dac.encoder_rates"];
        if (arr.count("dac.decoder_rates")) decoder_rates = arr["dac.decoder_rates"];
        hop_length = 1;
        for (int s : encoder_rates) hop_length *= s;
        hop_length = U("hop_length", hop_length);
        if (!has("dec.conv_in.weight") || !has("quant.0.table")) throw error("not a DAC model file");
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

const tensor & model::get(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw error("model is missing tensor " + name);
    return it->second;
}

}  // namespace dacn
