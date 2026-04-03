"""
强力锚定（K4团块）的启发式模拟

参数可调：锚点数量、权重、簇检测灵敏度。
簇检测阈值基于特征值间隙的统计分布自动确定（均值+标准差）。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh
import time

def build_3d_grid(L):
    N = L**3
    A = lil_matrix((N, N), dtype=np.float32)
    def idx(x,y,z): return x*L*L + y*L + z
    for x in range(L):
        for y in range(L):
            for z in range(L):
                u = idx(x,y,z)
                if x+1 < L: v = idx(x+1,y,z); A[u,v]=A[v,u]=1.0
                if y+1 < L: v = idx(x,y+1,z); A[u,v]=A[v,u]=1.0
                if z+1 < L: v = idx(x,y,z+1); A[u,v]=A[v,u]=1.0
    return A.tocsr()

def add_K4_anchors(A, L, num_anchors, weight=3.0):
    """随机添加 K4 锚点，权重可调"""
    N = L**3
    A = A.tolil()
    candidates = []
    for x in range(L-1):
        for y in range(L-1):
            for z in range(L-1):
                v0 = x*L*L + y*L + z
                v1 = (x+1)*L*L + y*L + z
                v2 = x*L*L + (y+1)*L + z
                v3 = x*L*L + y*L + (z+1)
                candidates.append([v0, v1, v2, v3])
    np.random.shuffle(candidates)
    for i in range(min(num_anchors, len(candidates))):
        nodes = candidates[i]
        for a in range(4):
            for b in range(a+1, 4):
                A[nodes[a], nodes[b]] += weight
                A[nodes[b], nodes[a]] += weight
    return A.tocsr()

def compute_eigenvalues(A, k=30):
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg==0] = 1e-9
    D_inv_sqrt = diags(1.0/np.sqrt(deg), offsets=0, format='csr')
    I = csr_matrix(np.eye(A.shape[0], dtype=np.float32))
    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    eigvals = eigsh(L, k=k, which='SM', return_eigenvectors=False)
    return eigvals

def detect_clusters(eigvals, sensitivity=1.0):
    """自动检测簇：基于间隙的均值和标准差，阈值 = mean + sensitivity * std"""
    diffs = np.diff(eigvals)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    threshold = mean_diff + sensitivity * std_diff
    gaps = np.where(diffs > threshold)[0]
    n_clusters = len(gaps) + 1
    return n_clusters, gaps, diffs

def main(L=12, configs=None):
    if configs is None:
        configs = [
            (0, 0, "Pure 3D", 1.0),
            (100, 3.0, "100 K4 weight=3.0", 1.0),
            (200, 3.0, "200 K4 weight=3.0", 1.0),
            (300, 3.0, "300 K4 weight=3.0", 1.0),
            (400, 4.0, "400 K4 weight=4.0", 1.0),
        ]
    print(f"Building 3D grid L={L} (N={L**3})...")
    A = build_3d_grid(L)
    for num, w, label, sens in configs:
        if num == 0:
            A_curr = A
        else:
            A_curr = add_K4_anchors(A, L, num, w)
        print(f"Computing eigenvalues for {label}...")
        start = time.time()
        eig = compute_eigenvalues(A_curr, k=30)
        elapsed = time.time() - start
        n_clusters, gaps, diffs = detect_clusters(eig, sensitivity=sens)
        print(f"  Time: {elapsed:.2f}s, clusters: {n_clusters}")
        print(f"  First 10 eig: {eig[:10]}")
        if len(gaps) > 0:
            print(f"  Gaps at indices: {gaps}")
        plt.figure(figsize=(10,4))
        plt.plot(eig, 'o-', markersize=4)
        plt.title(f"{label} (clusters={n_clusters})")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"strong_L{L}_{num}_{w}.png", dpi=150)
        plt.show()

if __name__ == "__main__":
    # 可调整参数：L=12, 锚点数量、权重、灵敏度
    main(L=12)