#include "controller.hpp"
#include <iomanip>
#include <iostream>

int main() {
    double target, actual;
    std::cout << std::setprecision(17);
    while (std::cin >> target >> actual) {
        std::cout << joint_error(target, actual) << '\n';
    }
}
