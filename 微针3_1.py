import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize

# ---------------------------
# 参数设置（男性头皮特性）
# ---------------------------
layer_epidermis = 100e-6  # 表皮层厚度
layer_dermis = 1e-3       # 真皮层厚度
num_layers = 200          # 模拟层数
hair_follicle_density = 10 # 毛囊密度

# 扩散系数
D_epidermis = 1e-12       # 表皮层扩散系数
D_dermis = 5e-10          # 真皮层扩散系数
D_follicle = 2e-9         # 毛囊区域扩散系数

clearance_rate = 1e-1     # 清除率
C0 = 1.0                  # 初始药物浓度

# 初始针头长度和直径
needle_length_initial = 200e-6          
needle_diameter = 50e-6

simulation_time = 8 * 3600  # 模拟时间为8小时
dt = 0.1  # 时间步长

# 初始化空间网格
total_depth = layer_epidermis + layer_dermis
dx = total_depth / num_layers
x = np.linspace(0, total_depth, num_layers)

# 随机分布毛囊位置
np.random.seed(42)
follicle_positions = np.random.choice(num_layers, size=hair_follicle_density, replace=False)

# 初始化扩散系数矩阵
D = np.full(num_layers, D_dermis)
D[x <= layer_epidermis] = D_epidermis
D[follicle_positions] = D_follicle

# 初始化药物浓度
C = np.zeros(num_layers)
C[x <= needle_length_initial] = C0

def simulate_scalp_diffusion(C, D, dt, dx, num_steps, clearance_rate, num_layers):
    # 模拟头皮扩散过程
    for _ in range(num_steps):
        C_new = np.copy(C)
        for i in range(1, num_layers-1):
            flux = D[i] * dt / dx**2 * (C[i+1] - 2*C[i] + C[i-1])  # 扩散方程
            if np.isnan(flux) or np.isinf(flux):  # 检查flux是否有效
                return C_new
            C_new[i] = C[i] + flux - clearance_rate * C[i] * dt
            if np.isnan(C_new[i]) or np.isinf(C_new[i]):  # 检查新浓度值
                return C_new
        C = np.copy(C_new)
    return C

def objective(params):
    # 检查参数有效性
    if np.isnan(params).any() or (params <= 0).any():
        return np.inf

    # 分解参数
    needle_length, drug_release_time = params
    
    # 限制针长度
    if needle_length > layer_epidermis * 1.2:
        return np.inf

    # 计算药物释放步数
    num_steps = int(np.round(drug_release_time / dt))
    if num_steps <= 0:
        return np.inf

    # 模拟针头插入后的初始浓度分布
    needle_region = (x <= needle_length)
    C_initial = np.zeros(num_layers)
    C_initial[needle_region] = C0

    # 进行扩散模拟
    C_final = simulate_scalp_diffusion(C_initial, D, dt, dx, num_steps, clearance_rate, num_layers)

    # 检查计算结果
    if np.isnan(C_final).any() or np.isinf(C_final).any():
        return np.inf

    # 计算目标值：毛囊浓度和表皮残留
    follicle_concentration = np.mean(C_final[follicle_positions])
    epidermal_residue = np.mean(C_final[x <= layer_epidermis])
    return -follicle_concentration + 10 * np.clip(epidermal_residue, 0, None)

# 初始猜测和约束
initial_guess = [200e-6, 7*3600]
bounds = [(50e-6, 300e-6), (3600, 8 * 3600)]

# 运行优化
result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B', options={
    'ftol': 1e-6,
    'gtol': 1e-5,
    'maxfun': 150
})

# 提取优化结果
optimized_length, optimized_release_time = result.x

# 模拟优化后的药物分布
needle_region_opt = (x <= optimized_length)
C_initial_opt = np.zeros(num_layers)
C_initial_opt[needle_region_opt] = C0
num_steps_opt = int(optimized_release_time / dt)
C_final_opt = simulate_scalp_diffusion(C_initial_opt, D, dt, dx, num_steps_opt, clearance_rate, num_layers)

# 绘制浓度分布
plt.figure(figsize=(10, 6))
plt.plot(x, C_final_opt, label='Drug Concentration')
plt.axvline(layer_epidermis, color='r', linestyle='--', label='Epidermis-Dermis Junction')
plt.scatter(x[follicle_positions], C_final_opt[follicle_positions], c='green', label='Hair Follicles')
plt.xlabel('Depth from skin surface (m)')
plt.ylabel('Drug Concentration (mol/m³)')
plt.title(f'Optimized Microneedle: {optimized_length*1e6:.1f} μm, Release Time: {optimized_release_time/3600:.1f} h')
plt.legend()
plt.grid(True)
plt.show()

# 输出优化结果
print(f"Optimized Microneedle Length: {optimized_length*1e6:.1f} μm")
print(f"Optimized Drug Release Time: {optimized_release_time/3600:.1f} hours")
print(f"Target Follicle Concentration: {np.mean(C_final_opt[follicle_positions]):.3e} mol/m")
print(f"Epidermal Residue: {np.mean(C_final_opt[x <= layer_epidermis]):.3e} mol/m")