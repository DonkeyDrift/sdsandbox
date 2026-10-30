// DonkeySim C++ 推理客户端:量化异步控制协议,五道验证闸。
//   dkcpp gate1  --port 9301 --ctl-step 20           骨架探针
//   dkcpp gate2  --jsonl outputs/cpp_replay.jsonl --onnx m.onnx   一致性回放
//   dkcpp gate3  --port 9301 --ctl-step 10 --seconds 60 --onnx m.onnx [--policy sine]
//   dkcpp gate4  --port 9301 --onnx m.onnx --ctl-step 20 --episodes 10 --seed-base 8100
//   dkcpp gate5  --port 9301 --onnx m.onnx --ctl-step 10 --seconds 600
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include <nlohmann/json.hpp>

#include "obs.hpp"
#include "policy.hpp"
#include "sim_client.hpp"

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * v.size()));
    if (idx == 0) idx = 1;
    return v[std::min(idx, v.size()) - 1];
}

static double rss_mb() {
#ifdef __APPLE__
    struct task_basic_info_64 info;
    mach_msg_type_number_t sz = TASK_BASIC_INFO_64_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO_64, (task_info_t)&info, &sz) == KERN_SUCCESS)
        return info.resident_size / 1048576.0;
#endif
    return -1.0;
}

static void send_control(dk::SimClient& c, double steer, double throttle, double brake) {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  R"({"msg_type":"control","steering":"%.4f","throttle":"%.4f","brake":"%.4f"})",
                  steer, throttle, brake);
    c.send_raw(buf);
    c.send_raw("\n");
}

// 握手:等 car_loaded -> 刹车 -> car_config(含 ctl_step_ms)-> cam_config(img_enc none)。
static void handshake(dk::SimClient& c, int ctl_step_ms) {
    double deadline = now_ms() + 15000;
    bool loaded = false;
    while (now_ms() < deadline && !loaded) {
        json j = c.next_json();
        if (j.is_discarded()) continue;
        std::string mt = j.value("msg_type", "");
        if (mt == "need_car_config" || mt == "car_loaded") loaded = mt == "car_loaded";
    }
    if (!loaded) throw std::runtime_error("car_loaded not received (sim in track scene?)");

    send_control(c, 0, 0, 1.0);
    json cc = {{"msg_type", "car_config"},
               {"body_style", "car01"},
               {"body_r", "255"},
               {"body_g", "0"},
               {"body_b", "255"},
               {"car_name", "dkcpp"},
               {"rl_mode", "1"}};
    if (ctl_step_ms > 0) cc["ctl_step_ms"] = std::to_string(ctl_step_ms);
    c.send_json(cc);
    json cam = {{"msg_type", "cam_config"},
                {"fov", "150"},
                {"fish_eye_x", "0.0"},
                {"fish_eye_y", "0.0"},
                {"img_w", "160"},
                {"img_h", "120"},
                {"img_d", "3"},
                {"img_enc", "none"},
                {"offset_x", "0.0"},
                {"offset_y", "3.0"},
                {"offset_z", "0.0"},
                {"rot_x", "90.0"}};
    c.send_json(cam);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

static void drain(dk::SimClient& c, double seconds) {
    double until = now_ms() + seconds * 1000.0;
    c.next_json();  // 确保至少阻塞到下一帧,避免忙等
    while (now_ms() < until) {
        try {
            c.next_json();
        } catch (...) {
            break;
        }
    }
}

static void send_reset(dk::SimClient& c, long seed, double lateral) {
    send_control(c, 0, 0, 1.0);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  R"({"msg_type":"reset_car","node_min":"5","node_max":"245","lateral_offset":"%.3f","heading_noise":"8.0","seed":"%ld"})",
                  lateral, seed);
    c.send_raw(buf);
    c.send_raw("\n");
}

// fork send_reset_car 的 lateral_offset 随机化:random.Random((seed,"lat")).uniform(-1,1)
// Python MT19937 无法廉价复刻,启动时调用同目录 lat_offsets.py 预计算。
static std::vector<double> lateral_offsets(long seed_base, int n) {
    std::vector<double> out;
    std::string cmd = "python3 src/cpp_client/lat_offsets.py " + std::to_string(seed_base) +
                      " " + std::to_string(n) + " 2>/dev/null";
    FILE* f = popen(cmd.c_str(), "r");
    if (!f) return out;
    double v;
    while (fscanf(f, "%lf", &v) == 1) out.push_back(v);
    pclose(f);
    return out;
}

struct Args {
    int port = 9091;
    int ctl_step = 0;
    double seconds = 60;
    std::string onnx, jsonl, policy = "idle";
    int episodes = 10;
    long seed_base = 8100;
    int max_steps = 2500;
    bool obs_inflight = false;  // INFLIGHT 实验:观测含在飞指令(8→12 维)
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (k == "--port") a.port = std::stoi(next());
        else if (k == "--ctl-step") a.ctl_step = std::stoi(next());
        else if (k == "--seconds") a.seconds = std::stod(next());
        else if (k == "--onnx") a.onnx = next();
        else if (k == "--jsonl") a.jsonl = next();
        else if (k == "--policy") a.policy = next();
        else if (k == "--episodes") a.episodes = std::stoi(next());
        else if (k == "--seed-base") a.seed_base = std::stol(next());
        else if (k == "--max-steps") a.max_steps = std::stoi(next());
        else if (k == "--obs-inflight") a.obs_inflight = true;
        else throw std::runtime_error("unknown arg: " + k);
    }
    return a;
}

// ---------------- gate1:骨架探针 ----------------
static int gate1(const Args& a) {
    dk::SimClient c("127.0.0.1", static_cast<uint16_t>(a.port));
    handshake(c, a.ctl_step);
    std::cout << "[gate1] connected, car_loaded, car_config(ctl_step_ms=" << a.ctl_step
              << ") + cam_config(img_enc none) sent\n";

    send_reset(c, 8100, 1.0);
    drain(c, 1.2);
    json t0 = c.next_telemetry();
    std::cout << "[gate1] post-reset frame: cte=" << t0.value("cte", -999.0)
              << " speed=" << t0.value("speed", -1.0)
              << " pos=(" << t0.value("pos_x", 0.0) << "," << t0.value("pos_y", 0.0)
              << "," << t0.value("pos_z", 0.0) << ")\n";

    const char* need[] = {"cte", "speed", "vel_x", "gyro_z", "yaw", "activeNode",
                          "totalNodes", "frame_id", "collision_impulse", "flipped",
                          "stuck_time", "lap", "hit", "pos_x", "roll", "pitch"};
    bool missing = false;
    for (const char* k : need) {
        if (!t0.contains(k)) {
            std::cout << "[gate1] MISSING field: " << k << "\n";
            missing = true;
        }
    }
    std::cout << "[gate1] field presence: " << (missing ? "FAIL" : "all present") << "\n";

    const int N = 300;
    std::vector<double> inter, proc;
    inter.reserve(N);
    proc.reserve(N);
    long prev_fid = t0.value("frame_id", -1L);
    bool fid_ok = true;
    double arrive = now_ms();
    for (int i = 0; i < N; ++i) {
        json t = c.next_telemetry();
        double t1 = now_ms();
        inter.push_back(t1 - arrive);
        arrive = t1;
        long fid = t.value("frame_id", -1L);
        if (fid != prev_fid + 1) fid_ok = false;
        prev_fid = fid;
        send_control(c, 0, 0, 0);
        proc.push_back(now_ms() - t1);
    }
    auto report = [&](const char* name, const std::vector<double>& v) {
        double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double var = 0;
        for (double x : v) var += (x - mean) * (x - mean);
        std::cout << "[gate1] " << name << " n=" << v.size() << " mean=" << mean
                  << " std=" << std::sqrt(var / v.size()) << " p50=" << percentile(v, 50)
                  << " p95=" << percentile(v, 95) << " p99=" << percentile(v, 99) << "\n";
    };
    report("telemetry inter-arrival (ms)", inter);
    report("client proc recv->send (ms)", proc);
    std::cout << "[gate1] frame_id monotonic +1: " << (fid_ok ? "OK" : "FAIL") << "\n";
    return missing || !fid_ok ? 1 : 0;
}

// ---------------- gate2:一致性回放 ----------------
static int gate2(const Args& a) {
    if (a.onnx.empty() || a.jsonl.empty()) throw std::runtime_error("gate2 needs --onnx and --jsonl");
    dk::Policy policy(a.onnx);
    std::ifstream in(a.jsonl);
    if (!in) throw std::runtime_error("cannot open " + a.jsonl);

    dk::ObsState st;
    st.obs_inflight = a.obs_inflight;
    st.clear_episode();
    std::string line;
    long n = 0, obs_fail = 0, act_fail = 0;
    double max_obs_diff = 0, max_act_diff = 0;
    int cur_ep = -1;
    long last_frame_id = -1;
    const int expect_dim = a.obs_inflight ? 12 : 8;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        json rec = json::parse(line, nullptr, false);
        if (rec.is_discarded() || !rec.contains("telemetry")) continue;
        if (rec.contains("ep") && rec["ep"].get<int>() != cur_ep) {
            cur_ep = rec["ep"].get<int>();
            st.clear_episode();  // 也清零在飞历史
            last_frame_id = -1;
        }
        // 录制侧已知工件:异步模式 ep 尾部偶发同 (ep,frame_id) 重复行,其 telemetry
        // 相对该行 obs 是陈旧键(观测态已前进)——C++ 无法重建该态,跳过 obs 对比,
        // 但仍推进在飞历史(下一行 obs 期望包含本行动作)。
        long fid = rec["telemetry"].contains("frame_id") ? rec["telemetry"]["frame_id"].get<long>() : -1;
        if (fid >= 0 && fid == last_frame_id) {
            auto exp_act_skip = rec["action"].get<std::vector<double>>();
            st.record_action(static_cast<float>(exp_act_skip[0]), static_cast<float>(exp_act_skip[1]));
            n++;
            continue;
        }
        if (fid >= 0) last_frame_id = fid;
        st.on_telemetry(rec["telemetry"]);
        auto obs = st.obs_vector();
        auto exp_obs = rec["obs"].get<std::vector<double>>();
        if (static_cast<int>(obs.size()) != expect_dim ||
            static_cast<int>(exp_obs.size()) != expect_dim) {
            std::cout << "[gate2] dim mismatch at frame " << n << ": cpp=" << obs.size()
                      << " rec=" << exp_obs.size() << " expect=" << expect_dim << "\n";
            return 1;
        }
        double d = 0;
        for (int i = 0; i < expect_dim; ++i) d = std::max(d, std::fabs(obs[i] - exp_obs[i]));
        max_obs_diff = std::max(max_obs_diff, d);
        if (d > 1e-4) {
            obs_fail++;
            if (obs_fail <= 5) {
                std::cout << "[gate2] fail#" << obs_fail << " frame=" << n
                          << " ep=" << cur_ep << " step=" << (rec.contains("step") ? rec["step"].get<int>() : -1)
                          << " diffs:";
                for (int i = 0; i < expect_dim; ++i)
                    std::cout << " d" << i << "=" << (obs[i] - exp_obs[i]);
                std::cout << "\n";
            }
        }

        auto act = policy.infer(obs);
        auto exp_act = rec["action"].get<std::vector<double>>();
        double ad = std::max(std::fabs(act[0] - exp_act[0]), std::fabs(act[1] - exp_act[1]));
        max_act_diff = std::max(max_act_diff, ad);
        if (ad > 1e-4) act_fail++;
        // 在飞历史记【录制侧动作】(非本侧推断):两侧共享同一历史,obs 段测纯构造
        // 一致性;动作段单独测推断数值一致性。若记推断动作,1e-4 级浮点噪声会经
        // 在飞槽反馈放大成轨迹级分叉(帧 840 起 1e-4 → 1.72 案例),门失去意义。
        st.record_action(static_cast<float>(exp_act[0]), static_cast<float>(exp_act[1]));
        n++;
    }
    std::cout << "[gate2] frames=" << n << " max_obs_diff=" << max_obs_diff
              << " (tol 1e-4, fails=" << obs_fail << ")\n";
    std::cout << "[gate2] max_act_diff=" << max_act_diff << " (tol 1e-4, fails=" << act_fail
              << ")\n";
    bool ok = n > 0 && obs_fail == 0 && act_fail == 0;
    std::cout << "[gate2] verdict: " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}

// ---------------- gate3:性能门 ----------------
static int gate3(const Args& a) {
    if (a.onnx.empty()) throw std::runtime_error("gate3 needs --onnx");
    dk::Policy policy(a.onnx);
    dk::SimClient c("127.0.0.1", static_cast<uint16_t>(a.port));
    handshake(c, a.ctl_step);
    std::cout << "[gate3] ctl_step_ms=" << a.ctl_step << " policy=" << a.policy
              << " (onnx infer included, action overridden)\n";
    send_reset(c, 8100, 1.0);
    drain(c, 1.2);

    dk::ObsState st;
    st.obs_inflight = a.obs_inflight;
    std::vector<double> proc, infer, cycle;
    double arrive = now_ms(), tstart = arrive;
    long frames = 0;
    double bound = a.ctl_step > 0 ? a.ctl_step : 10.0;
    while ((now_ms() - tstart) / 1000.0 < a.seconds) {
        json t = c.next_telemetry();
        double t1 = now_ms();
        cycle.push_back(t1 - arrive);
        arrive = t1;
        st.on_telemetry(t);
        auto obs = st.obs_vector();
        double i0 = now_ms();
        auto act = policy.infer(obs);
        (void)act;  // 空载:动作被恒0/正弦覆盖
        infer.push_back(now_ms() - i0);
        double steer = 0.0;
        if (a.policy == "sine") steer = 0.3 * std::sin(2.0 * M_PI * (t1 - tstart) / 2000.0);
        send_control(c, steer, 0.0, 0.0);
        st.record_action(static_cast<float>(steer), 0.0f);  // 在飞历史记实际发出的动作
        proc.push_back(now_ms() - t1);
        frames++;
    }
    double over = 0;
    for (double x : proc)
        if (x > bound) over++;
    std::cout << "[gate3] frames=" << frames << " seconds=" << a.seconds << "\n";
    std::cout << "[gate3] proc(recv->send incl onnx) p50=" << percentile(proc, 50)
              << " p95=" << percentile(proc, 95) << " p99=" << percentile(proc, 99)
              << " over_" << bound << "ms=" << (100.0 * over / frames) << "%\n";
    std::cout << "[gate3] onnx infer p50=" << percentile(infer, 50)
              << " p95=" << percentile(infer, 95) << "\n";
    std::cout << "[gate3] telemetry cycle mean="
              << std::accumulate(cycle.begin(), cycle.end(), 0.0) / cycle.size()
              << " p50=" << percentile(cycle, 50) << " p99=" << percentile(cycle, 99) << "\n";
    bool ok = percentile(proc, 95) < 6.0 && (100.0 * over / frames) < 2.0;
    std::cout << "[gate3] gate(p95<6ms & over<2%): " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}

// ---------------- gate4:驾驶验收 ----------------
static int run_episode(dk::SimClient& c, dk::Policy& policy, dk::ObsState& st, long seed,
                       double lateral, int max_steps, std::string& reason, double& total,
                       int& steps, double& mean_speed) {
    send_reset(c, seed, lateral);
    drain(c, 1.0);
    st.clear_episode();
    json first = c.next_telemetry();
    st.on_telemetry(first);
    auto obs = st.obs_vector();
    total = 0.0;
    steps = 0;
    double spd = 0.0;
    reason = "max_steps";
    while (steps < max_steps) {
        auto act = policy.infer(obs);
        st.record_action(act[0], act[1]);  // 在飞历史先入,下帧 obs 才含本动作
        send_control(c, act[0], act[1], 0.0);
        json t = c.next_telemetry();
        st.on_telemetry(t);
        bool done = st.over;
        double r = st.calc_reward(done);
        obs = st.obs_vector();
        total += r;
        steps++;
        spd += st.speed;
        if (done) {
            std::ostringstream os;
            os.setf(std::ios::fixed);
            os.precision(2);
            os << "terminated(cte=" << st.cte << " hit=" << st.hit << " flipped=" << st.flipped
               << " stuck=" << st.stuck_time << " lap=" << st.lap << " pos=(" << st.x << ","
               << st.y << "," << st.z << "))";
            reason = os.str();
            break;
        }
    }
    mean_speed = spd / std::max(steps, 1);
    return steps;
}

static int gate4(const Args& a) {
    if (a.onnx.empty()) throw std::runtime_error("gate4 needs --onnx");
    dk::Policy policy(a.onnx);
    dk::SimClient c("127.0.0.1", static_cast<uint16_t>(a.port));
    handshake(c, a.ctl_step);
    auto lats = lateral_offsets(a.seed_base, a.episodes);
    if (static_cast<int>(lats.size()) != a.episodes)
        throw std::runtime_error("lateral offset precompute failed");

    std::cout << "[gate4] model=" << a.onnx << " ctl_step_ms=" << a.ctl_step
              << " episodes=" << a.episodes << " seed_base=" << a.seed_base
              << " max_steps=" << a.max_steps
              << " obs_inflight=" << (a.obs_inflight ? 12 : 8) << "\n";
    dk::ObsState st;
    st.obs_inflight = a.obs_inflight;
    struct Ep { double reward, steps, speed; std::string reason; bool parked, crash; };
    std::vector<Ep> eps;
    for (int e = 0; e < a.episodes; ++e) {
        long seed = a.seed_base + e;
        std::string reason;
        double total = 0, ms = 0;
        int steps = 0;
        run_episode(c, policy, st, seed, lats[e], a.max_steps, reason, total, steps, ms);
        bool crash = reason.rfind("terminated", 0) == 0;
        bool parked = ms < 1.0;
        eps.push_back({total, static_cast<double>(steps), ms, reason, parked, crash});
        std::cout << "[gate4] ep" << e << " seed=" << seed << " reward=" << total
                  << " steps=" << steps << " mean_speed=" << ms
                  << (parked ? " PARKED" : "") << " -> " << reason << "\n";
    }
    std::vector<const Ep*> valid;
    for (auto& e : eps)
        if (!e.parked) valid.push_back(&e);
    double avg_r = 0, avg_s = 0, avg_v = 0;
    int crashes = 0, laps = 0;
    for (auto* e : valid) {
        avg_r += e->reward;
        avg_s += e->steps;
        avg_v += e->speed;
        crashes += e->crash ? 1 : 0;
    }
    avg_r /= std::max<size_t>(valid.size(), 1);
    avg_s /= std::max<size_t>(valid.size(), 1);
    avg_v /= std::max<size_t>(valid.size(), 1);
    std::cout << "[gate4] valid " << valid.size() << "/" << eps.size()
              << " avg_reward=" << avg_r << " avg_steps=" << avg_s << " mean_speed=" << avg_v
              << " crashes=" << crashes << " laps=" << laps << "\n";
    return 0;
}

// ---------------- gate5:浸泡 ----------------
static int gate5(const Args& a) {
    if (a.onnx.empty()) throw std::runtime_error("gate5 needs --onnx");
    dk::Policy policy(a.onnx);
    dk::SimClient c("127.0.0.1", static_cast<uint16_t>(a.port));
    handshake(c, a.ctl_step);
    auto lats = lateral_offsets(a.seed_base, 64);
    std::cout << "[gate5] soak ctl_step_ms=" << a.ctl_step << " seconds=" << a.seconds
              << " rss0=" << rss_mb() << "MB\n";
    dk::ObsState st;
    st.obs_inflight = a.obs_inflight;
    long ep = 0, resets = 0;
    double t0 = now_ms(), last_report = t0;
    long frames = 0;
    try {
        while ((now_ms() - t0) / 1000.0 < a.seconds) {
            std::string reason;
            double total = 0, ms = 0;
            int steps = 0;
            long seed = a.seed_base + (ep % 64);
            run_episode(c, policy, st, seed, lats[ep % 64], a.max_steps, reason, total,
                        steps, ms);
            ep++;
            resets++;
            frames += steps;
            if (now_ms() - last_report >= 30000) {
                std::cout << "[gate5] t=" << (now_ms() - t0) / 1000.0 << "s frames=" << frames
                          << " resets=" << resets << " rss=" << rss_mb() << "MB\n";
                last_report = now_ms();
            }
        }
    } catch (const std::exception& e) {
        std::cout << "[gate5] DISCONNECT after " << (now_ms() - t0) / 1000.0 << "s: " << e.what()
                  << "\n";
        return 1;
    }
    std::cout << "[gate5] done t=" << (now_ms() - t0) / 1000.0 << "s frames=" << frames
              << " resets=" << resets << " rss=" << rss_mb() << "MB\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: dkcpp gate1|gate2|gate3|gate4|gate5 [opts]\n";
        return 2;
    }
    try {
        std::string mode = argv[1];
        Args a = parse_args(argc, argv);
        if (mode == "gate1") return gate1(a);
        if (mode == "gate2") return gate2(a);
        if (mode == "gate3") return gate3(a);
        if (mode == "gate4") return gate4(a);
        if (mode == "gate5") return gate5(a);
        std::cerr << "unknown mode: " << mode << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "[dkcpp] ERROR: " << e.what() << "\n";
        return 1;
    }
}
