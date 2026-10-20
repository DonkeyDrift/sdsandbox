// 状态观测构造:逐字段镜像 GymDonkeyCar fork 的
// gym_donkeycar/envs/donkey_sim.py —— on_telemetry 的字段更新顺序、
// observe() 的 8 维向量、calc_reward、determine_episode_over。
// 关键陷阱:on_telemetry 里 forward_vel 用【上一帧】的 roll/pitch/yaw 和 vel
// 计算(先算 forward_vel 再更新 vel/姿态),C++ 必须同样顺序,否则 obs 逐点不合。
#pragma once

#include <array>
#include <cmath>
#include <string>

#include <nlohmann/json.hpp>

namespace dk {

struct ObsState {
    // fork handler 状态
    double cte = 0.0;
    double speed = 0.0;
    double forward_vel = 0.0;
    double x = 0.0, y = 0.0, z = 0.0;
    double gyro_x = 0.0, gyro_y = 0.0, gyro_z = 0.0;
    double vel_x = 0.0, vel_y = 0.0, vel_z = 0.0;
    double roll = 0.0, pitch = 0.0, yaw = 0.0;
    std::string hit = "none";
    bool over = false;
    long steps_since_reset = 0;
    double collision_impulse = 0.0;
    bool flipped = false;
    double stuck_time = 0.0;
    int lap = 0;
    long frame_id = 0;
    int active_node = 0;
    int total_nodes = 1;
    // rl_eval 参数
    double max_cte = 2.5;
    double stuck_thresh = 2.5;

    // fork on_telemetry 的字段处理顺序(fork 行 715-795)。
    void on_telemetry(const nlohmann::json& m) {
        steps_since_reset++;

        if (m.contains("pos_x")) {
            x = m["pos_x"].get<double>();
            y = m["pos_y"].get<double>();
            z = m["pos_z"].get<double>();
        }
        if (m.contains("speed")) speed = m["speed"].get<double>();

        // forward_vel:上一帧 pitch/yaw/roll + 上一帧 vel(此时 vel/姿态还没更新)
        {
            const double e0 = pitch * M_PI / 180.0;  // pitch
            const double e1 = yaw * M_PI / 180.0;    // yaw
            const double e2 = roll * M_PI / 180.0;   // roll
            const double cx = std::cos(e0 * 0.5), sx = std::sin(e0 * 0.5);
            const double cy = std::cos(e1 * 0.5), sy = std::sin(e1 * 0.5);
            const double cz = std::cos(e2 * 0.5), sz = std::sin(e2 * 0.5);
            // euler_to_quat(CireNeikual):e = [pitch, yaw, roll]
            const double qx = sz * cx * cy - cz * sx * sy;
            const double qy = cz * sx * cy + sz * cx * sy;
            const double qz = cz * cx * sy - sz * sx * cy;
            const double qw = cz * cx * cy + sz * sx * sy;
            // rotate_vec(q, [0,0,1]) = forward
            auto cross = [](double ax, double ay, double az, double bx, double by, double bz,
                            double& ox, double& oy, double& oz) {
                ox = ay * bz - az * by;
                oy = az * bx - ax * bz;
                oz = ax * by - ay * bx;
            };
            double uvx, uvy, uvz, uuvx, uuvy, uuvz;
            cross(qx, qy, qz, 0.0, 0.0, 1.0, uvx, uvy, uvz);
            cross(qx, qy, qz, uvx, uvy, uvz, uuvx, uuvy, uuvz);
            const double s = 2.0 * qw;
            const double fx = 0.0 + uvx * s + uuvx * 2.0;
            const double fy = 0.0 + uvy * s + uuvy * 2.0;
            const double fz = 1.0 + uvz * s + uuvz * 2.0;  // v=[0,0,1],v[2]=1
            forward_vel = fx * vel_x + fy * vel_y + fz * vel_z;
        }

        if (m.contains("gyro_x")) {
            gyro_x = m["gyro_x"].get<double>();
            gyro_y = m["gyro_y"].get<double>();
            gyro_z = m["gyro_z"].get<double>();
        }
        if (m.contains("vel_x")) {  // vel 在 forward_vel 之后更新
            vel_x = m["vel_x"].get<double>();
            vel_y = m["vel_y"].get<double>();
            vel_z = m["vel_z"].get<double>();
        }
        if (m.contains("roll")) {  // 姿态在 forward_vel 之后更新
            roll = m["roll"].get<double>();
            pitch = m["pitch"].get<double>();
            yaw = m["yaw"].get<double>();
        }
        if (m.contains("cte")) cte = m["cte"].get<double>();

        if (m.contains("collision_impulse")) collision_impulse = m["collision_impulse"].get<double>();
        if (m.contains("flipped")) flipped = m["flipped"].get<bool>();
        if (m.contains("stuck_time")) stuck_time = m["stuck_time"].get<double>();
        if (m.contains("lap")) lap = m["lap"].get<int>();
        if (m.contains("frame_id")) frame_id = m["frame_id"].get<long>();
        if (m.contains("activeNode")) active_node = m["activeNode"].get<int>();
        if (m.contains("totalNodes")) total_nodes = m["totalNodes"].get<int>();

        // don't update hit once session over
        if (over) return;
        if (m.contains("hit")) hit = m["hit"].get<std::string>();
        determine_episode_over();
    }

    // fork determine_episode_over(fork 行 827-854,含出生宽限)
    void determine_episode_over() {
        if (std::fabs(cte) > max_cte) {
            if (steps_since_reset < 5 && std::fabs(cte) > 2.0 * max_cte) {
                // 出生宽限
            } else {
                over = true;
            }
        } else if (hit != "none") {
            over = true;
        }
        if (flipped) over = true;
        if (stuck_time > stuck_thresh) over = true;
    }

    // fork observe() 的 8 维状态向量(float32 截断前用 double 算,与 numpy 一致)
    std::array<float, 8> obs_vector() const {
        return {static_cast<float>(cte / 2.0),
                static_cast<float>(speed / 10.0),
                static_cast<float>(forward_vel / 10.0),
                static_cast<float>(yaw / 180.0),
                static_cast<float>(gyro_z / 90.0),
                static_cast<float>(static_cast<double>(active_node) /
                                    std::max(total_nodes, 1)),
                0.0f,
                0.0f};
    }

    // fork calc_reward(crash_penalty=0 的 rl_eval 口径)
    double calc_reward(bool done) const {
        if (done) return -1.0;
        if (cte > max_cte) return -1.0;
        if (hit != "none") return -2.0;
        if (forward_vel > 0.0) return (1.0 - std::fabs(cte) / max_cte) * forward_vel;
        return forward_vel;
    }

    // fork _clear_episode_state(reset 后调用)
    void clear_episode() {
        steps_since_reset = 0;
        hit = "none";
        cte = speed = forward_vel = 0.0;
        x = y = z = 0.0;
        gyro_x = gyro_y = gyro_z = 0.0;
        vel_x = vel_y = vel_z = 0.0;
        roll = pitch = yaw = 0.0;
        over = false;
        collision_impulse = 0.0;
        flipped = false;
        stuck_time = 0.0;
        lap = 0;
        frame_id = 0;
        active_node = 0;
        total_nodes = 1;
    }
};

}  // namespace dk
