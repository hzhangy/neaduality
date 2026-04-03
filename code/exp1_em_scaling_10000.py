import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time

def build_2d_weaver_sparse(N):
    side = int(np.sqrt(N))
    A = sp.lil_matrix((N, N))
    # 1. 弱力链
    for i in range(N-1):
        A[i, i+1] = 1.0; A[i+1, i] = 1.0
    # 2. 电磁缝合
    for i in range(N):
        j = i + side
        if j < N:
            A[i, j] = 1.0; A[j, i] = 1.0
    return A.tocsr()

def compute_ds_weyl(A):
    N = A.shape[0]
    deg = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg + 1e-9))
    L = sp.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # 取前 20% 的特征值进行拟合，这是 Weyl Law 最稳健的区间
    k_eig = max(50, int(N * 0.2))
    k_eig = min(k_eig, 2000) # 算力平衡点
    
    try:
        eigvals = eigsh(L, k=k_eig, which='SA', return_eigenvectors=False)
        eigvals = np.sort(eigvals)
        eigvals = eigvals[eigvals > 1e-8]
        
        cum_count = np.arange(1, len(eigvals) + 1)
        log_lambda = np.log(eigvals)
        log_cum = np.log(cum_count)
        
        # 拟合中间段
        idx = slice(len(eigvals)//10, 9*len(eigvals)//10)
        slope, _ = np.polyfit(log_lambda[idx], log_cum[idx], 1)
        return 2 * slope
    except:
        return np.nan

def main():
    # 扩大 N 范围，特别是大 N 端
    N_list = [100, 400, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000]
    ds_results = []
    
    print(f"{'N':<8} | {'side':<6} | {'d_s (Weyl)':<12} | {'Time (s)':<8}")
    print("-" * 40)
    
    for N in N_list:
        t0 = time.time()
        A = build_2d_weaver_sparse(N)
        ds = compute_ds_weyl(A)
        elapsed = time.time() - t0
        ds_results.append(ds)
        print(f"{N:<8} | {int(np.sqrt(N)):<6} | {ds:<12.4f} | {elapsed:<8.2f}")

    # FSS 拟合：ds(N) = ds_inf + a * (1/sqrt(N))
    inv_L = 1.0 / np.sqrt(N_list)
    p = np.polyfit(inv_L, ds_results, 1)
    ds_inf = p[1]
    
    print("-" * 40)
    print(f"外推极限 d_s(∞) = {ds_inf:.4f}")
    
    plt.plot(inv_L, ds_results, 'ro-', label='Measured d_s')
    plt.plot([0, inv_L[0]], [ds_inf, p[1] + p[0]*inv_L[0]], 'b--', label=f'Extrapolation (inf={ds_inf:.2f})')
    plt.axhline(2.0, color='g', linestyle=':', label='Theoretical d=2')
    plt.xlabel('1/L (Inverse Scale)')
    plt.ylabel('Spectral Dimension d_s')
    plt.title('Finite Size Scaling: EM Phase (Weaver Operator)')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/em_fss.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()