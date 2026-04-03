import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def compute_ds_high_res(A):
    N = A.shape[0]
    deg = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg + 1e-12))
    L = sp.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    try:
        # 针对低维探测，增加特征值捕捉量
        eigvals = eigsh(L, k=100, which='SA', return_eigenvectors=False)
        eigvals = np.sort(eigvals[eigvals > 1e-9])
        log_lambda = np.log(eigvals)
        log_cum = np.log(np.arange(1, len(eigvals) + 1))
        # 拟合最初始的斜率，看“第一声胎动”
        slope, _ = np.polyfit(log_lambda[:40], log_cum[:40], 1)
        return 2 * slope
    except: return 1.0

def main_detail():
    L = 12 # 稍微提高分辨率
    N = L**3
    chi_we = np.linspace(0, 1.0, 20) # 密集扫描 0 到 1
    ds_we = []

    print(">>> 运行代码 2：正在深度探测 0-1 (弱电合体) 阶段的胎动...")
    for c in chi_we:
        A = sp.lil_matrix((N, N))
        for i in range(N): A[i, (i+1)%N] = 1.0; A[(i+1)%N, i] = 1.0
        # 模拟编织强度的增加
        for i in range(N):
            j = (i + L) % N
            A[i, j] += c; A[j, i] += c
        
        ds = compute_ds_high_res(A.tocsr())
        ds_we.append(ds)
        print(f"编织强度 χ={c:.2f} | 维度 ds={ds:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(chi_we, ds_we, 'm-s', label='WE Phase (0 -> 1)')
    plt.axhline(1.0, color='gray', ls=':', label='1D Ground State')
    plt.xlabel('Weaving Strength'); plt.ylabel('Spectral Dimension ds')
    plt.title('GZ_01 Detail: The Birth of Surface from Chain')
    plt.legend()
    plt.savefig('figures/we_detail.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main_detail()