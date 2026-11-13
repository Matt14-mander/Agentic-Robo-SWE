#pragma once
#include <algorithm>
namespace retrieval_fixture {
class IntegralController {
    double integral_ = 0.0;
public:
    double step(double error, double dt, double kp, double ki, double limit) {
        const double candidate = integral_ + error * dt;
        const double raw = kp * error + ki * candidate;
        const double bounded = std::clamp(raw, -limit, limit);
        if (raw == bounded || (raw > limit && error < 0) || (raw < -limit && error > 0))
            integral_ = candidate;
        return bounded;
    }
};
}
