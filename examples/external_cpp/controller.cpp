#include "controller.hpp"
#include <cmath>

double joint_error(double target, double actual) {
    // Deliberate public development defect: continuous joints need angle wrapping.
    return target - actual;
}
