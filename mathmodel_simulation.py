import math

def calculate_interface_velocity(u0, epsilon, phi, beta, t):
    denominator_exp = epsilon + phi * (1 - epsilon)
    exponent = -beta * t / denominator_exp
    u_int = u0 * epsilon * math.exp(exponent)
    return u_int

import numpy as np
from scipy.integrate import solve_ivp

def solve_drug_concentration(t_span, c0, params, v_func, t_eval=None):
    """
    params: 包含K_L, k_D, theta, rho, beta, h, c_s的字典
    v_func: 函数，输入t返回当前体积v(t)
    """
    K_L = params['K_L']
    k_D = params['k_D']
    theta = params['theta']
    rho = params['rho']
    beta = params['beta']
    h = params['h']
    c_s = params['c_s']
    
    def ode_func(t, c):
        v = v_func(t)
        theta_rad = np.deg2rad(theta)  # 假设theta以度为单位
        factor = 4 * (k_D * np.tan(theta_rad) / (rho * np.cos(theta_rad))) * h**2
        term1 = -K_L * c
        term2_part1 = (beta * rho - c) / v
        term2_part2 = c_s - ((1 - beta)/beta) * c
        term2 = factor * term2_part1 * term2_part2
        dcdt = term1 + term2
        return dcdt
    
    sol = solve_ivp(ode_func, t_span, [c0], t_eval=t_eval, method='LSODA')
    return sol.t, sol.y[0]

def calculate_cumulative_release(d_s, P_w, c_SD_t, c_t, beta, rho, v_c0, m0):
    numerator = d_s * (P_w**2) * (c_SD_t + c_t)
    denominator = beta * rho * v_c0 + m0
    M_t = 1 - numerator / denominator
    return M_t

# 示例参数
u0 = 1e-6  # 初始速度
epsilon = 0.5
phi = 0.3
beta = 0.2
t = 10  # 时间点

# 计算界面速度
u_int = calculate_interface_velocity(u0, epsilon, phi, beta, t)
print(f"Interface Velocity at t={t}: {u_int:.2e} m/s")

# 药物浓度参数
params = {
    'K_L': 0.1,
    'k_D': 0.05,
    'theta': 30,  # 度
    'rho': 1000,
    'beta': 0.2,
    'h': 1e-4,
    'c_s': 500
}
v_func = lambda t: 0.001  # 假设体积恒定
t_span = [0, 10]
c0 = 0.0

# 求解浓度动态
time_points, c_values = solve_drug_concentration(t_span, c0, params, v_func, t_eval=np.linspace(0,10,100))

# 计算累积释放量（示例取最后一个时间点）
d_s = 0.01
P_w = 1e-3
c_SD_t = 100  # 假设皮肤药物浓度
v_c0 = 0.001
m0 = 0.5

M_t = calculate_cumulative_release(d_s, P_w, c_SD_t, c_values[-1], beta, params['rho'], v_c0, m0)
print(f"Cumulative Release at t={t}: {M_t:.4f}")



