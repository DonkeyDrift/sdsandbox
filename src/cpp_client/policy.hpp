// ONNX Runtime(C API)策略推理封装:obs(1x8 float32) -> [steering, throttle]。
// 输入/输出名与 src/export_onnx.py 的导出一致("obs"/"action"),单线程执行
// 配置优先低延迟。
#pragma once

#include <array>
#include <stdexcept>
#include <string>

#include <onnxruntime_c_api.h>

namespace dk {

class Policy {
public:
    explicit Policy(const std::string& model_path) {
        api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        OrtStatus* st = api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "dkcpp", &env_);
        check(st);
        OrtSessionOptions* opts = nullptr;
        st = api_->CreateSessionOptions(&opts);
        check(st);
        api_->SetIntraOpNumThreads(opts, 1);
        api_->SetInterOpNumThreads(opts, 1);
        api_->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
#ifdef _WIN32
        std::wstring wpath(model_path.begin(), model_path.end());
        st = api_->CreateSession(env_, wpath.c_str(), opts, &sess_);
#else
        st = api_->CreateSession(env_, model_path.c_str(), opts, &sess_);
#endif
        check(st);
        api_->ReleaseSessionOptions(opts);

        st = api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &meminfo_);
        check(st);

        size_t n_in = 0, n_out = 0;
        api_->SessionGetInputCount(sess_, &n_in);
        api_->SessionGetOutputCount(sess_, &n_out);
        if (n_in != 1 || n_out < 1)
            throw std::runtime_error("unexpected onnx io counts: in=" +
                                     std::to_string(n_in) + " out=" + std::to_string(n_out));
    }

    ~Policy() {
        if (meminfo_) api_->ReleaseMemoryInfo(meminfo_);
        if (sess_) api_->ReleaseSession(sess_);
        if (env_) api_->ReleaseEnv(env_);
    }
    Policy(const Policy&) = delete;
    Policy& operator=(const Policy&) = delete;

    std::array<float, 2> infer(const std::array<float, 8>& obs) {
        std::array<int64_t, 2> shape{1, 8};
        OrtValue* tensor = nullptr;
        OrtStatus* st = api_->CreateTensorWithDataAsOrtValue(
            meminfo_, const_cast<float*>(obs.data()), obs.size() * sizeof(float),
            shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &tensor);
        check(st);

        const char* in_names[] = {"obs"};
        const char* out_names[] = {"action"};
        OrtValue* outs[1] = {nullptr};
        st = api_->Run(sess_, nullptr, in_names, &tensor, 1, out_names, 1, outs);
        api_->ReleaseValue(tensor);
        check(st);

        float* data = nullptr;
        api_->GetTensorMutableData(outs[0], reinterpret_cast<void**>(&data));
        std::array<float, 2> action{data[0], data[1]};
        api_->ReleaseValue(outs[0]);
        return action;
    }

private:
    void check(OrtStatus* st) {
        if (st != nullptr) {
            std::string msg = api_->GetErrorMessage(st);
            api_->ReleaseStatus(st);
            throw std::runtime_error("onnxruntime: " + msg);
        }
    }

    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSession* sess_ = nullptr;
    OrtMemoryInfo* meminfo_ = nullptr;
};

}  // namespace dk
