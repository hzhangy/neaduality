import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def simulate_entanglement_shortcut(N=10000, use_random_fold=True):
    # 1. 建立 1D 链图（邻接矩阵）
    A = np.zeros((N, N))
    for i in range(N-1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    graph = csr_matrix(A)
    
    # 2. 将 1D 索引映射到 3D 空间
    side = int(N**(1/3))
    x0 = np.arange(N) % side
    y0 = (np.arange(N) // side) % side
    z0 = np.arange(N) // (side * side)
    
    if use_random_fold:
        perm = np.random.permutation(N)
        x = x0[perm]
        y = y0[perm]
        z = z0[perm]
    else:
        x, y, z = x0, y0, z0
    
    pos_3d = np.stack([x, y, z], axis=1)
    
    print(">>> N.E.A. 量子结算：正在扫描 3D 空间中的‘幽灵纠缠’...")
    print("-" * 60)
    
    sample_pairs = [100, 500, 1000, 2000, 5000]
    distances_3d = []
    logic_steps = []
    for i in sample_pairs:
        p1 = pos_3d[i]
        p2 = pos_3d[i+1]
        dist_3d = np.linalg.norm(p1 - p2)
        step_nea = 1
        distances_3d.append(dist_3d)
        logic_steps.append(step_nea)
        print(f"纠缠对索引 ({i}, {i+1}) | 3D 距离: {dist_3d:>6.2f} | 图距离: {step_nea}")
    
    random_pairs = np.random.randint(0, N, (1000, 2))
    rnd_dist_3d = []
    rnd_steps_nea = []
    for u, v in random_pairs:
        d3 = np.linalg.norm(pos_3d[u] - pos_3d[v])
        step = abs(u - v)
        rnd_dist_3d.append(d3)
        rnd_steps_nea.append(step)
    
    # 绘图并保存
    plt.figure(figsize=(10, 6))
    plt.scatter(rnd_dist_3d, rnd_steps_nea, alpha=0.2, c='gray', label='Normal Particles (Locality)')
    plt.scatter(distances_3d, logic_steps, c='red', s=100, edgecolors='black', label='Entangled Pairs (1D neighbors)')
    plt.yscale('log')
    plt.xlabel('Apparent 3D Distance')
    plt.ylabel('Graph Distance on 1D Chain')
    plt.title('Entanglement: 1D Topological Proximity vs 3D Illusory Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/entanglement.png', dpi=150)  # 添加保存命令
    plt.show()

if __name__ == "__main__":
    simulate_entanglement_shortcut(use_random_fold=True)