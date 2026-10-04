#include "dacfile.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>

namespace dacn {
namespace {

constexpr size_t MAX_FILE = (size_t) 1 << 31;   // 2 GiB
constexpr size_t MAX_STACK = 1 << 20;
constexpr size_t MAX_MEMO = 1 << 20;

struct obj;
using P = std::shared_ptr<obj>;

enum class kind { none, boolean, integer, real, str, bytes, tuple, list, dict, global, ndarray, dtype };

struct obj {
    kind k = kind::none;
    bool b = false;
    int64_t i = 0;
    double f = 0;
    std::string s;                          // str / bytes payload / global "module name" / dtype code
    std::vector<P> items;                   // tuple / list / object-array elements
    std::vector<std::pair<P, P>> entries;   // dict
    // ndarray
    std::vector<int64_t> shape;
    P dtype;
    std::string raw;
    bool built = false;
    // dtype
    char endian = '=';
};

P mk(kind k) { auto o = std::make_shared<obj>(); o->k = k; return o; }

struct fail : std::runtime_error { using std::runtime_error::runtime_error; };

bool allowed_global(const std::string & mod, const std::string & name) {
    if ((mod == "numpy.core.multiarray" || mod == "numpy._core.multiarray") && name == "_reconstruct") return true;
    if (mod == "numpy" && (name == "ndarray" || name == "dtype")) return true;
    return false;
}

class unpickler {
public:
    unpickler(const uint8_t * p, size_t n) : p_(p), n_(n) {}

    P run() {
        while (true) {
            const uint8_t op = u8();
            switch (op) {
                case 0x80: { if (u8() > 5) throw fail("unsupported pickle protocol"); break; }   // PROTO
                case 0x95: take(8); break;                                                      // FRAME
                case '.': { P r = pop(); return r; }                                             // STOP
                case '(': marks_.push_back(stack_.size()); break;                                // MARK
                case ')': push(mk(kind::tuple)); break;                                          // EMPTY_TUPLE
                case ']': push(mk(kind::list)); break;                                           // EMPTY_LIST
                case '}': push(mk(kind::dict)); break;                                           // EMPTY_DICT
                case 'N': push(mk(kind::none)); break;
                case 0x88: case 0x89: { P o = mk(kind::boolean); o->b = op == 0x88; push(o); break; }
                case 'K': push_int(u8()); break;                                                 // BININT1
                case 'M': { uint16_t v; memcpy(&v, take(2), 2); push_int(v); break; }             // BININT2
                case 'J': { int32_t v; memcpy(&v, take(4), 4); push_int(v); break; }              // BININT
                case 0x8a: {                                                                     // LONG1
                    const uint8_t n = u8();
                    if (n > 8) throw fail("integer too large");
                    const uint8_t * b = take(n);
                    uint64_t v = 0;
                    for (int k = 0; k < n; ++k) v |= (uint64_t) b[k] << (8 * k);
                    if (n && n < 8 && (b[n - 1] & 0x80)) v |= ~0ull << (8 * n);                    // sign-extend
                    push_int((int64_t) v);
                    break;
                }
                case 'G': {                                                                      // BINFLOAT (big endian)
                    const uint8_t * b = take(8);
                    uint64_t v = 0;
                    for (int k = 0; k < 8; ++k) v = (v << 8) | b[k];
                    P o = mk(kind::real); memcpy(&o->f, &v, 8); push(o);
                    break;
                }
                case 'X': push_str(kind::str, u32()); break;                                     // BINUNICODE
                case 0x8c: push_str(kind::str, u8()); break;                                     // SHORT_BINUNICODE
                case 0x8d: push_str(kind::str, u64()); break;                                    // BINUNICODE8
                case 'C': push_str(kind::bytes, u8()); break;                                    // SHORT_BINBYTES
                case 'B': push_str(kind::bytes, u32()); break;                                   // BINBYTES
                case 0x8e: push_str(kind::bytes, u64()); break;                                  // BINBYTES8
                case 'c': {                                                                      // GLOBAL
                    std::string mod = line(), name = line();
                    push_global(mod, name);
                    break;
                }
                case 0x93: {                                                                     // STACK_GLOBAL
                    P name = pop(), mod = pop();
                    if (name->k != kind::str || mod->k != kind::str) throw fail("bad STACK_GLOBAL");
                    push_global(mod->s, name->s);
                    break;
                }
                case 'q': memo_put(u8()); break;                                                 // BINPUT
                case 'r': memo_put(u32()); break;                                                // LONG_BINPUT
                case 0x94: memo_put(memo_.size()); break;                                        // MEMOIZE
                case 'h': memo_get(u8()); break;                                                 // BINGET
                case 'j': memo_get(u32()); break;                                                // LONG_BINGET
                case 0x85: case 0x86: case 0x87: {                                               // TUPLE1/2/3
                    const size_t n = op - 0x84;
                    if (stack_.size() < n) throw fail("stack underflow");
                    P t = mk(kind::tuple);
                    t->items.assign(stack_.end() - n, stack_.end());
                    stack_.resize(stack_.size() - n);
                    push(t);
                    break;
                }
                case 't': { P t = mk(kind::tuple); t->items = pop_mark(); push(t); break; }      // TUPLE
                case 'a': { P v = pop(); top(kind::list)->items.push_back(v); break; }           // APPEND
                case 'e': { auto v = pop_mark(); auto & l = top(kind::list)->items; l.insert(l.end(), v.begin(), v.end()); break; }
                case 's': { P v = pop(), k = pop(); top(kind::dict)->entries.emplace_back(k, v); break; }  // SETITEM
                case 'u': {                                                                      // SETITEMS
                    auto v = pop_mark();
                    if (v.size() % 2) throw fail("odd SETITEMS");
                    P d = top(kind::dict);
                    for (size_t k = 0; k < v.size(); k += 2) d->entries.emplace_back(v[k], v[k + 1]);
                    break;
                }
                case 'R': reduce(); break;
                case 'b': build(); break;
                default: {
                    char msg[64];
                    snprintf(msg, sizeof msg, "unsupported pickle opcode 0x%02x", op);
                    throw fail(msg);
                }
            }
        }
    }

private:
    const uint8_t * p_;
    size_t n_, off_ = 0;
    std::vector<P> stack_;
    std::vector<size_t> marks_;
    std::vector<P> memo_;

    const uint8_t * take(uint64_t k) {
        if (k > n_ - off_) throw fail("truncated pickle");
        const uint8_t * r = p_ + off_;
        off_ += (size_t) k;
        return r;
    }
    uint8_t u8() { return *take(1); }
    uint32_t u32() { uint32_t v; memcpy(&v, take(4), 4); return v; }
    uint64_t u64() { uint64_t v; memcpy(&v, take(8), 8); return v; }
    std::string line() {
        const size_t start = off_;
        while (off_ < n_ && p_[off_] != '\n') ++off_;
        if (off_ >= n_) throw fail("truncated pickle");
        std::string s((const char *) p_ + start, off_ - start);
        ++off_;
        return s;
    }
    void push(P o) {
        if (stack_.size() >= MAX_STACK) throw fail("pickle stack too deep");
        stack_.push_back(std::move(o));
    }
    P pop() {
        if (stack_.empty() || (!marks_.empty() && stack_.size() <= marks_.back())) throw fail("stack underflow");
        P o = stack_.back();
        stack_.pop_back();
        return o;
    }
    P top(kind k) {
        if (stack_.empty() || stack_.back()->k != k) throw fail("unexpected object type");
        return stack_.back();
    }
    std::vector<P> pop_mark() {
        if (marks_.empty()) throw fail("missing MARK");
        const size_t m = marks_.back();
        marks_.pop_back();
        std::vector<P> v(stack_.begin() + m, stack_.end());
        stack_.resize(m);
        return v;
    }
    void push_int(int64_t v) { P o = mk(kind::integer); o->i = v; push(o); }
    void push_str(kind k, uint64_t len) {
        const uint8_t * b = take(len);
        P o = mk(k);
        o->s.assign((const char *) b, (size_t) len);
        push(o);
    }
    void push_global(const std::string & mod, const std::string & name) {
        if (!allowed_global(mod, name)) throw fail("refusing pickle global " + mod + "." + name);
        P o = mk(kind::global);
        o->s = mod + " " + name;
        push(o);
    }
    void memo_put(size_t idx) {
        if (stack_.empty()) throw fail("memo put on empty stack");
        if (idx >= MAX_MEMO) throw fail("memo index too large");
        if (memo_.size() <= idx) memo_.resize(idx + 1);
        memo_[idx] = stack_.back();
    }
    void memo_get(size_t idx) {
        if (idx >= memo_.size() || !memo_[idx]) throw fail("bad memo reference");
        push(memo_[idx]);
    }

    // REDUCE only for the two whitelisted constructors; nothing is ever "called".
    void reduce() {
        P args = pop(), fn = pop();
        if (fn->k != kind::global || args->k != kind::tuple) throw fail("bad REDUCE");
        const std::string & name = fn->s.substr(fn->s.find(' ') + 1);
        if (name == "_reconstruct") {
            if (args->items.size() != 3 || args->items[0]->k != kind::global) throw fail("bad ndarray reconstruct");
            push(mk(kind::ndarray));
        } else if (name == "dtype") {
            if (args->items.empty() || args->items[0]->k != kind::str) throw fail("bad dtype");
            P d = mk(kind::dtype);
            d->s = args->items[0]->s;
            push(d);
        } else {
            throw fail("refusing to call " + fn->s);
        }
    }

    void build() {
        P state = pop();
        if (stack_.empty()) throw fail("BUILD on empty stack");
        P o = stack_.back();
        if (o->k == kind::dtype) {
            // (version, endian, subarray, names, fields, elsize, alignment, flags)
            if (state->k != kind::tuple || state->items.size() < 2 || state->items[1]->k != kind::str ||
                state->items[1]->s.size() != 1)
                throw fail("bad dtype state");
            o->endian = state->items[1]->s[0];
        } else if (o->k == kind::ndarray) {
            // (version, shape, dtype, is_fortran, rawdata | list-of-objects)
            if (state->k != kind::tuple || state->items.size() != 5) throw fail("bad ndarray state");
            P shape = state->items[1], dt = state->items[2], fortran = state->items[3], data = state->items[4];
            if (shape->k != kind::tuple || dt->k != kind::dtype || fortran->k != kind::boolean) throw fail("bad ndarray state");
            if (fortran->b) throw fail("fortran-order arrays are not supported");
            for (const P & d : shape->items) {
                if (d->k != kind::integer || d->i < 0) throw fail("bad ndarray shape");
                o->shape.push_back(d->i);
            }
            o->dtype = dt;
            if (data->k == kind::bytes) o->raw = data->s;
            else if (data->k == kind::list) o->items = data->items;
            else throw fail("bad ndarray data");
            o->built = true;
        } else {
            throw fail("BUILD on unsupported object");
        }
    }
};

const obj * dict_get(const obj & d, const char * key) {
    for (const auto & kv : d.entries)
        if (kv.first->k == kind::str && kv.first->s == key) return kv.second.get();
    return nullptr;
}

int64_t as_int(const obj * o, const char * what) {
    if (!o) throw fail(std::string("missing field ") + what);
    if (o->k == kind::integer) return o->i;
    if (o->k == kind::boolean) return o->b;
    throw fail(std::string("field ") + what + " is not an integer");
}

int64_t elem_count(const obj & a) {
    int64_t n = 1;
    for (int64_t d : a.shape) {
        if (d && n > (int64_t) MAX_FILE / d) throw fail("array too large");
        n *= d;
    }
    return n;
}

double as_scalar_f(const obj * o, const char * what) {
    if (!o) throw fail(std::string("missing field ") + what);
    if (o->k == kind::real) return o->f;
    if (o->k == kind::integer) return (double) o->i;
    if (o->k == kind::ndarray && o->built && elem_count(*o) == 1) {
        const std::string & c = o->dtype->s;
        if ((c == "f4" || c == "<f4") && o->raw.size() == 4) { float v; memcpy(&v, o->raw.data(), 4); return v; }
        if ((c == "f8" || c == "<f8") && o->raw.size() == 8) { double v; memcpy(&v, o->raw.data(), 8); return v; }
    }
    throw fail(std::string("field ") + what + " is not a float scalar");
}

}  // namespace

bool dac_file_parse(const uint8_t * data, size_t size, dac_file & out, std::string & err) {
    try {
        // ---- .npy envelope ----
        static const uint8_t magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
        if (size < 10 || memcmp(data, magic, 6) != 0) throw fail("not a .dac/.npy file");
        const uint8_t major = data[6];
        size_t hl = 0, off = 0;
        if (major == 1) { hl = data[8] | (data[9] << 8); off = 10; }
        else if (major == 2 || major == 3) {
            if (size < 12) throw fail("truncated header");
            hl = data[8] | (data[9] << 8) | ((size_t) data[10] << 16) | ((size_t) data[11] << 24);
            off = 12;
        } else throw fail("unsupported .npy version");
        if (hl > size - off) throw fail("truncated header");
        const std::string header((const char *) data + off, hl);
        if (header.find("'descr': '|O'") == std::string::npos || header.find("'shape': ()") == std::string::npos)
            throw fail("not a DAC artifact (expected a pickled dict)");
        off += hl;

        // ---- restricted pickle ----
        P root = unpickler(data + off, size - off).run();
        if (root->k != kind::ndarray || !root->built || root->items.size() != 1 || root->items[0]->k != kind::dict)
            throw fail("unexpected .dac payload layout");
        const obj & art = *root->items[0];

        const obj * codes = dict_get(art, "codes");
        const obj * meta = dict_get(art, "metadata");
        if (!codes || codes->k != kind::ndarray || !codes->built) throw fail("missing codes array");
        if (!meta || meta->k != kind::dict) throw fail("missing metadata");

        // ---- codes [B, n_codebooks, T] (saved as uint16) ----
        if (codes->shape.size() != 3) throw fail("codes must be 3-D [batch, codebooks, frames]");
        std::string code = codes->dtype->s;
        const char endian = codes->dtype->endian;
        if (code.size() == 3 && (code[0] == '<' || code[0] == '|' || code[0] == '=')) code = code.substr(1);
        if (endian == '>') throw fail("big-endian codes are not supported");
        size_t isz = 0;
        bool is_signed = false;
        if (code == "u1" || code == "i1") isz = 1;
        else if (code == "u2" || code == "i2") isz = 2;
        else if (code == "u4" || code == "i4") isz = 4;
        else if (code == "u8" || code == "i8") isz = 8;
        else throw fail("unsupported codes dtype " + code);
        is_signed = code[0] == 'i';
        const int64_t n = elem_count(*codes);
        if ((size_t) n * isz != codes->raw.size()) throw fail("codes buffer size mismatch");
        out.B = codes->shape[0];
        out.n_codebooks = codes->shape[1];
        out.T = codes->shape[2];
        out.codes.resize((size_t) n);
        const uint8_t * rb = (const uint8_t *) codes->raw.data();
        for (int64_t k = 0; k < n; ++k) {
            int64_t v = 0;
            switch (isz) {
                case 1: v = is_signed ? (int64_t) (int8_t) rb[k] : rb[k]; break;
                case 2: { uint16_t u; memcpy(&u, rb + 2 * k, 2); v = is_signed ? (int16_t) u : u; break; }
                case 4: { uint32_t u; memcpy(&u, rb + 4 * k, 4); v = is_signed ? (int32_t) u : u; break; }
                case 8: { uint64_t u; memcpy(&u, rb + 8 * k, 8); v = (int64_t) u; break; }
            }
            if (v < 0 || v > 65535) throw fail("code index out of range");
            out.codes[(size_t) k] = (int32_t) v;
        }

        // ---- metadata ----
        out.chunk_length = as_int(dict_get(*meta, "chunk_length"), "chunk_length");
        out.original_length = as_int(dict_get(*meta, "original_length"), "original_length");
        out.sample_rate = as_int(dict_get(*meta, "sample_rate"), "sample_rate");
        out.channels = as_int(dict_get(*meta, "channels"), "channels");
        out.padding = as_int(dict_get(*meta, "padding"), "padding") != 0;
        out.input_db = as_scalar_f(dict_get(*meta, "input_db"), "input_db");
        const obj * ver = dict_get(*meta, "dac_version");
        out.dac_version = (ver && ver->k == kind::str) ? ver->s : "";

        if (out.dac_version != "1.0.0") throw fail("unsupported dac_version '" + out.dac_version + "'");
        if (out.chunk_length <= 0 || out.channels <= 0 || out.sample_rate <= 0 || out.original_length < 0)
            throw fail("invalid metadata");
        if (out.B % out.channels) throw fail("codes batch is not a multiple of channels");
        if (out.channels > 32 || out.sample_rate > 768000) throw fail("unsupported channel count / sample rate");
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

bool dac_file_load(const fs::path & path, dac_file & out, std::string & err) {
    std::vector<uint8_t> buf;
    if (!read_file(path, buf, err, MAX_FILE)) return false;
    return dac_file_parse(buf.data(), buf.size(), out, err);
}

// ---------------------------------------------------------------- writer
//
// Emits the pickle numpy writes for np.save(f, {"codes": uint16 array, "metadata": {...}}):
// a 0-d object array whose single element is the dict, each ndarray as
// _reconstruct(ndarray, (0,), b'b') + BUILD((1, shape, dtype, False, raw_bytes)).

namespace {

struct pickler {
    std::vector<uint8_t> b;

    void op(uint8_t c) { b.push_back(c); }
    void raw(const void * p, size_t n) { b.insert(b.end(), (const uint8_t *) p, (const uint8_t *) p + n); }
    void u32le(uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((uint8_t) (v >> (8 * i))); }
    void global(const char * mod, const char * name) {
        op('c'); raw(mod, strlen(mod)); op('\n'); raw(name, strlen(name)); op('\n');
    }
    void integer(int64_t v) {
        if (v >= 0 && v < 256) { op('K'); op((uint8_t) v); }
        else if (v >= 0 && v < 65536) { op('M'); op((uint8_t) v); op((uint8_t) (v >> 8)); }
        else if (v >= INT32_MIN && v <= INT32_MAX) { op('J'); u32le((uint32_t) (int32_t) v); }
        else { op(0x8a); op(8); for (int i = 0; i < 8; ++i) op((uint8_t) ((uint64_t) v >> (8 * i))); }   // LONG1
    }
    void str(const std::string & s) { op('X'); u32le((uint32_t) s.size()); raw(s.data(), s.size()); }
    void bytes(const void * p, size_t n) {
        if (n < 256) { op('C'); op((uint8_t) n); }
        else { op('B'); u32le((uint32_t) n); }
        raw(p, n);
    }
    void boolean(bool v) { op(v ? 0x88 : 0x89); }
    // numpy.dtype(code, False, True) with state (3, endian, None, None, None, -1, -1, flags)
    void dtype(const char * code, const char * endian, int flags) {
        global("numpy", "dtype");
        str(code); boolean(false); boolean(true); op(0x87); op('R');
        op('('); integer(3); str(endian); op('N'); op('N'); op('N'); integer(-1); integer(-1); integer(flags); op('t');
        op('b');
    }
    void ndarray_begin() {
        global("numpy.core.multiarray", "_reconstruct");
        global("numpy", "ndarray");
        integer(0); op(0x85); bytes("b", 1); op(0x87); op('R');
        op('('); integer(1);
    }
    void shape(const std::vector<int64_t> & dims) {
        op('(');
        for (int64_t d : dims) integer(d);
        op('t');
    }
    void ndarray_end() { op('t'); op('b'); }
};

}  // namespace

std::vector<uint8_t> dac_file_serialize(const dac_file & f) {
    pickler p;
    p.op(0x80); p.op(3);                                    // PROTO 3
    p.ndarray_begin();                                      // outer 0-d object array
    p.shape({});
    p.dtype("O8", "|", 63);
    p.boolean(false);
    p.op(']');                                              // object list with one element: the dict
    p.op('}'); p.op('(');
    {
        p.str("codes");
        std::vector<uint16_t> u16(f.codes.size());
        for (size_t i = 0; i < f.codes.size(); ++i) u16[i] = (uint16_t) f.codes[i];
        p.ndarray_begin();
        p.shape({f.B, f.n_codebooks, f.T});
        p.dtype("u2", "<", 0);
        p.boolean(false);
        std::vector<uint8_t> le(u16.size() * 2);
        for (size_t i = 0; i < u16.size(); ++i) { le[2 * i] = (uint8_t) u16[i]; le[2 * i + 1] = (uint8_t) (u16[i] >> 8); }
        p.bytes(le.data(), le.size());
        p.ndarray_end();

        p.str("metadata");
        p.op('}'); p.op('(');
        p.str("input_db");
        p.ndarray_begin();
        p.shape({1});
        p.dtype("f4", "<", 0);
        p.boolean(false);
        const float db = (float) f.input_db;
        uint32_t dbits;
        memcpy(&dbits, &db, 4);
        uint8_t dle[4] = {(uint8_t) dbits, (uint8_t) (dbits >> 8), (uint8_t) (dbits >> 16), (uint8_t) (dbits >> 24)};
        p.bytes(dle, 4);
        p.ndarray_end();
        p.str("original_length"); p.integer(f.original_length);
        p.str("sample_rate"); p.integer(f.sample_rate);
        p.str("chunk_length"); p.integer(f.chunk_length);
        p.str("channels"); p.integer(f.channels);
        p.str("padding"); p.boolean(f.padding);
        p.str("dac_version"); p.str(f.dac_version);
        p.op('u');
    }
    p.op('u');
    p.op('a');                                              // list.append(dict)
    p.ndarray_end();
    p.op('.');

    // .npy v1.0 envelope; header padded with spaces so the payload starts 64-byte aligned
    std::string header = "{'descr': '|O', 'fortran_order': False, 'shape': (), }";
    const size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header.push_back('\n');
    std::vector<uint8_t> out = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, (uint8_t) header.size(), (uint8_t) (header.size() >> 8)};
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), p.b.begin(), p.b.end());
    return out;
}

bool dac_file_save(const fs::path & path, const dac_file & f, std::string & err) {
    const std::vector<uint8_t> data = dac_file_serialize(f);
    FILE * fp = open_file(path, "wb");
    if (!fp) { err = "cannot create " + path.u8string(); return false; }
    const bool ok = fwrite(data.data(), 1, data.size(), fp) == data.size();
    if (fclose(fp) != 0 || !ok) { err = "write failed"; return false; }
    return true;
}

}  // namespace dacn
