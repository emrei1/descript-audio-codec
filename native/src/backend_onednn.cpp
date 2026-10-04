// oneDNN backend: direct (JIT brgemm) convolution and deconvolution primitives.
#include "backend.h"

#include <dnnl.hpp>

#include <unordered_map>

namespace dacn {
namespace {

using namespace dnnl;
using tag = memory::format_tag;
using dt  = memory::data_type;

class onednn_backend final : public backend {
public:
    const char * name() const override { return "oneDNN"; }
    void sync() override { strm_.wait(); }

    std::function<void()> conv(const conv_params & p, view src, view dst) override {
        try {
            return build(p, src, dst);
        } catch (const dnnl::error & e) {
            throw error(std::string("oneDNN: ") + e.what());
        }
    }

private:
    engine eng_{engine::kind::cpu, 0};
    stream strm_{eng_};
    std::vector<memory> keep_;

    memory act(const view & v) { return memory({{1, v.C, v.T}, dt::f32, tag::nwc}, eng_, v.ptr); }

    memory prep_weights(const memory::desc & want, const memory::dims & dims, tag wtag, const float * data) {
        memory user({dims, dt::f32, wtag}, eng_, (void *) data);
        if (want == user.get_desc()) return user;
        memory w(want, eng_);
        reorder(user, w).execute(strm_, user, w);
        strm_.wait();
        keep_.push_back(w);
        return w;
    }

    std::function<void()> build(const conv_params & p, view src, view dst) {
        memory s = act(src), d = act(dst);
        const memory::dims wdims = {p.OC, p.IC, p.K};
        const memory::desc w_any(wdims, dt::f32, tag::any);
        memory b({{p.OC}, dt::f32, tag::x}, eng_, (void *) p.b);
        post_ops po;
        if (p.sum_into) po.append_sum(1.0f);
        if (p.tanh) po.append_eltwise(algorithm::eltwise_tanh, 0.f, 0.f);
        primitive_attr attr;
        attr.set_post_ops(po);

        std::shared_ptr<primitive> prim;
        memory w;
        if (p.transposed) {
            deconvolution_forward::primitive_desc pd(eng_, prop_kind::forward_inference, algorithm::deconvolution_direct,
                s.get_desc(), w_any, b.get_desc(), d.get_desc(), {p.stride}, {p.dil - 1}, {p.pad}, {p.pad}, attr);
            w = prep_weights(pd.weights_desc(), wdims, tag::bac, p.w);   // PyTorch [IC][OC][K]
            prim = std::make_shared<deconvolution_forward>(pd);
        } else {
            convolution_forward::primitive_desc pd(eng_, prop_kind::forward_inference, algorithm::convolution_direct,
                s.get_desc(), w_any, b.get_desc(), d.get_desc(), {p.stride}, {p.dil - 1}, {p.pad}, {p.pad}, attr);
            w = prep_weights(pd.weights_desc(), wdims, tag::oiw, p.w);   // PyTorch [OC][IC][K]
            prim = std::make_shared<convolution_forward>(pd);
        }
        std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, s}, {DNNL_ARG_WEIGHTS, w}, {DNNL_ARG_BIAS, b}, {DNNL_ARG_DST, d}};
        stream * st = &strm_;
        return [prim, args, st]() { prim->execute(*st, args); };
    }
};

}  // namespace

std::unique_ptr<backend> make_backend() { return std::make_unique<onednn_backend>(); }

}  // namespace dacn
