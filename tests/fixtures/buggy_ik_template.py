"""**含 bug 的 2-link 平面机械臂逆运动学** —— 这是测试/demo 的"题面"。

每次跑测试或 demo 时, 由 fixture / demo CLI 把本文件复制到一个 workspace 文件,
让 agent 在那个副本上修, 避免污染本 template。本文件**永远不应被修改**。

预期 bug:
1. ``acos`` 输入未 clip, 当 |cos_theta2| > 1 时抛 ``ValueError: math domain error``。
2. ``theta1`` 计算用了 ``+`` 而非 ``-``, 给出镜像解。
3. 没有处理目标距离超出可达半径 (l1 + l2) 的情况。
4. 关节角约束完全缺失 (真实机器人通常 +/- 170 度限位)。
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
