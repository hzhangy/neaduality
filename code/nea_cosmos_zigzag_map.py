import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def compute_ds(A):
    N = A.shape[0]
    deg = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg + 1e-12))
    L = sp.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    try:
        # 取 150 个特征值，拟合中段 Weyl Law
        eigvals = eigsh(L, k=150, which='SA', return_eigenvectors=False)
        eigvals = np.sort(eigvals[eigvals > 1e-8])
        log_lambda = np.log(eigvals)
        log_cum = np.log(np.arange(1, len(eigvals) + 1))
        idx = slice(len(eigvals)//4, 3*len(eigvals)//4)
        slope, _ = np.polyfit(log_lambda[idx], log_cum[idx], 1)
        return 2 * slope
    except: return 1.0

def main_map():
    L = 10
    N = L**3
    chi_range = np.linspace(0, 3, 20)
    ds_list = []
    
    print(">>> 运行代码 1：宇宙全史‘之字形’地图绘制中...")
    for chi in chi_range:
        A = sp.lil_matrix((N, N))
        # 0. 基础 1D (Weak)
        for i in range(N): A[i, (i+1)%N] = 1.0; A[(i+1)%N, i] = 1.0
        # 1. 织造 2D (EM)
        if chi > 0:
            w1 = min(1.0, chi)
            for i in range(N): j = (i + L)%N; A[i,j] += w1; A[j,i] += w1
        # 2. 拉伸 3D (Gravity)
        if chi > 1:
            w2 = min(1.0, chi-1)
            for i in range(N): k = (i + L*L)%N; A[i,k] += w2; A[k,i] += w2
        # 3. 锚定 4D (Strong)
        if chi > 2:
            w3 = (chi-2) * 10
            for i in range(0, N, 20): # 简化锚定点
                nodes = [i, (i+1)%N, (i+L)%N, (i+L*L)%N]
                for m in range(4):
                    for n in range(m+1, 4): A[nodes[m], nodes[n]] += w3; A[nodes[n], nodes[m]] += w3
        
        ds = compute_ds(A.tocsr())
        ds_list.append(ds)
        print(f"进度 χ={chi:.2f} | 维度 ds={ds:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(chi_range, ds_list, 'k-o', linewidth=2)
    plt.axvline(1, color='r', ls='--', label='GZ_01 (WE)')
    plt.axvline(2, color='b', ls='--', label='GZ_12 (EG)')
    plt.xlabel('Evolution Progress χ'); plt.ylabel('Spectral Dimension ds')
    plt.title('Universal Genesis Ladder (Zig-zag Map)')
    plt.legend()
    plt.savefig('figures/zigzag_map.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main_map()