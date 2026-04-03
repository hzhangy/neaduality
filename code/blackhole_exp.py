import numpy as np
import matplotlib.pyplot as plt

def build_stable_3d(L, anchor_weight=2):
    """构建一个受4D强力锚定的稳定3D空间"""
    N = L**3
    A = np.zeros((N, N))
    def idx(x, y, z): return x*L*L + y*L + z
    
    # 基础3D骨架
    for x in range(L):
        for y in range(L):
            for z in range(L):
                u = idx(x, y, z)
                if x+1 < L: v = idx(x+1, y, z); A[u,v] = A[v,u] = 1
                if y+1 < L: v = idx(x, y+1, z); A[u,v] = A[v,u] = 1
                if z+1 < L: v = idx(x, y, z+1); A[u,v] = A[v,u] = 1
    
    # 添加4D锚点 (K4) 以锁定3维
    for _ in range(20):
        nodes = np.random.choice(N, 4, replace=False)
        for i in range(4):
            for j in range(i+1, 4):
                A[nodes[i], nodes[j]] = A[nodes[j], nodes[i]] = anchor_weight
    return A

def simulate_bh_collapse(A, intensity):
    """模拟黑洞坍缩：增加引力负载并触发带宽脱锁"""
    N = A.shape[0]
    side = int(round(N**(1/3)))
    center = N // 2
    
    A_bh = A.copy()
    # 1. 模拟引力增强：增加中心区域的权重
    for i in range(N):
        dist = np.linalg.norm(np.array(np.unravel_index(i, (side,side,side))) - side//2)
        if dist < 2:
            A_bh[center, i] = A_bh[i, center] = intensity
            
    # 2. 模拟带宽脱锁：当强度超过阈值，3D结构断裂，退化为2D全息映射
    # 逻辑：在高强度下，强制切断 z 轴方向的连接，只保留平面编织
    if intensity > 5:
        for i in range(N):
            x, y, z = np.unravel_index(i, (side,side,side))
            if z > side//2: # 模拟视界内的维度剥离
                neighbors = np.where(A_bh[i] > 0)[0]
                for n in neighbors:
                    # 如果连接跨越了深度，则切断
                    nx, ny, nz = np.unravel_index(n, (side,side,side))
                    if nz != z:
                        A_bh[i, n] = A_bh[n, i] = 0.1 # 极弱连接，模拟全息残留
    return A_bh

def compute_local_ds(A, t=1.0):
    """计算当前拓扑的平均谱维度"""
    deg = A.sum(axis=1)
    # 避免孤立点
    deg[deg == 0] = 1e-9
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    eigvals = np.linalg.eigvalsh(L)
    eigvals = eigvals[eigvals > 1e-10]
    
    # 使用中等尺度的 t 来测量维度
    t_vals = np.array([t*0.5, t, t*2.0])
    traces = [np.sum(np.exp(-tv * eigvals)) for tv in t_vals]
    ds = -2 * (np.log(traces[2]) - np.log(traces[0])) / (np.log(t_vals[2]) - np.log(t_vals[0]))
    return ds

# 主实验循环
L = 8
A_init = build_stable_3d(L)
intensities = np.linspace(1, 15, 10)
ds_history = []

print("Starting Black Hole Collapse Simulation...")
for m in intensities:
    A_curr = simulate_bh_collapse(A_init, m)
    ds = compute_local_ds(A_curr, t=2.0)
    ds_history.append(ds)
    print(f"Gravity Intensity: {m:.2f} | Spectral Dimension ds: {ds:.4f}")

plt.figure(figsize=(8,5))
plt.plot(intensities, ds_history, 'ro-', linewidth=2)
plt.axhline(y=3.0, color='g', linestyle='--', label='Normal Space (3D)')
plt.axhline(y=2.0, color='b', linestyle='--', label='Event Horizon (2D Surface)')
plt.xlabel('Gravity Intensity (Bandwidth Pressure)')
plt.ylabel('Effective Spectral Dimension ds')
plt.title('N.E.A. Black Hole: Dimensional Collapse (3D -> 2D)')
plt.legend()
plt.grid(True)
plt.savefig('figures/bh_collapse.png', dpi=150)
plt.show()