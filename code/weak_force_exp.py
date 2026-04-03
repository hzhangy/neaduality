import numpy as np
import matplotlib.pyplot as plt
import time

def build_1d_chain_dense(N):
    """
    直接构建稠密归一化拉普拉斯矩阵 L = I - D^{-1/2} A D^{-1/2}
    """
    # 构建邻接矩阵
    A = np.zeros((N, N))
    for i in range(N-1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    
    # 度矩阵
    degree = A.sum(axis=1)
    inv_sqrt_deg = 1.0 / np.sqrt(degree)
    D_inv_sqrt = np.diag(inv_sqrt_deg)
    
    # 归一化拉普拉斯
    L = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    return L

def compute_spectral_dimension_full(L_dense, t_min=1e-2, t_max=1e3, n_t=100):
    """
    使用全谱计算谱维度 d_s(t)
    """
    # 计算全部特征值
    eigvals = np.linalg.eigvalsh(L_dense)
    # 剔除可能的小负值（数值误差）
    eigvals = eigvals[eigvals > 1e-10]
    
    # t 范围
    t_range = np.logspace(np.log10(t_min), np.log10(t_max), n_t)
    
    # 热核迹
    trace = np.sum(np.exp(-t_range[:, None] * eigvals[None, :]), axis=1)
    
    # 对数微分
    log_t = np.log(t_range)
    log_trace = np.log(trace)
    d_log_t = np.gradient(log_t)
    d_log_trace = np.gradient(log_trace)
    d_s = -2 * d_log_trace / d_log_t
    
    return t_range, d_s, eigvals

def run_weak_force_experiment_full(N_list=[100, 200, 500]):
    results = {}
    for N in N_list:
        print(f"Building 1D chain with N={N}...")
        L_dense = build_1d_chain_dense(N)
        print(f"Computing full spectrum for N={N}...")
        start = time.time()
        t_range, d_s, eigvals = compute_spectral_dimension_full(L_dense)
        elapsed = time.time() - start
        
        # 取中段平台区平均值（避开两端）
        idx_mid = slice(len(d_s)//4, 3*len(d_s)//4)
        d_s_platform = np.mean(d_s[idx_mid])
        
        results[N] = (t_range, d_s, d_s_platform, elapsed, eigvals)
        print(f"N={N}, d_s_platform={d_s_platform:.4f}, time={elapsed:.2f}s")
    return results

def plot_results(results):
    plt.figure(figsize=(12, 5))
    
    # 谱维度曲线
    plt.subplot(1, 2, 1)
    for N, (t_range, d_s, d_s_platform, _, _) in results.items():
        plt.semilogx(t_range, d_s, label=f'N={N}, platform={d_s_platform:.3f}', linewidth=2)
    plt.axhline(y=1.0, color='k', linestyle='--', label='Theoretical d_s=1')
    plt.xlabel('Diffusion time t')
    plt.ylabel('Spectral dimension d_s(t)')
    plt.title('Spectral Dimension of 1D Chain (Full Spectrum)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 2)
    
    # 特征值分布
    plt.subplot(1, 2, 2)
    for N, (_, _, _, _, eigvals) in results.items():
        plt.plot(np.arange(1, len(eigvals)+1), eigvals, 'o-', markersize=2, label=f'N={N}')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue Spectrum')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weak_force_full_spectrum.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    results = run_weak_force_experiment_full(N_list=[100, 200, 500])
    plot_results(results)