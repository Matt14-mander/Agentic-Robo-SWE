"""一个**故意带 bug** 的 2-link 平面机械臂逆运动学示例。

预期 bug:
1. ``acos`` 输入未做 clip, 当目标点在工作空间边界时会 ``ValueError``。
2. ``theta1`` 计算用了错误的 ``+`` 而非 ``-`` (符号反了, 给出镜像解)。
3. 没有处理目标距离超出可达半径 (l1 + l2) 的情况, 直接溢出。
4. 关节角约束完全缺失 (真实机器人通常 +/- 170 度限位)。

用作 Agent 端到端测试: 修复建议应至少指出 1, 2 中的一项。
"""

from __future__ import annotations

import math


def inverse_kinematics_2link(x: float, y: float, l1: float = 1.0, l2: float = 1.0) -> tuple[float, float]:
    """2-link 机械臂逆运动学 (elbow-down 解)。

    Args:
        x, y: 末端目标位置。
        l1, l2: 两段连杆长度。
    Returns:
        (theta1, theta2) 关节角, 单位弧度。
    """
    d_sq = x * x + y * y
    cos_theta2 = (d_sq - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    # BUG 1: acos 输入未 clip, |cos_theta2| 可能 > 1
    theta2 = math.acos(cos_theta2)

    k1 = l1 + l2 * math.cos(theta2)
    k2 = l2 * math.sin(theta2)
    # BUG 2: 符号错误 —— 正确应为 atan2(y, x) - atan2(k2, k1)
    theta1 = math.atan2(y, x) + math.atan2(k2, k1)

    return theta1, theta2
