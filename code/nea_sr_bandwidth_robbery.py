import numpy as np
import matplotlib.pyplot as plt

def simulate_time_dilation(bandwidth_limit=1000, max_v=1000):
    """
    N.E.A. 狭义相对论实验：带宽抢占模拟
    bandwidth_limit: 节点的总处理能力 (c)
    """
    velocities = np.linspace(0, bandwidth_limit * 0.99, 50)
    internal_ticks = [] # 记录粒子的内部“演化时间”

    print(">>> N.E.A. 狭义相对论结算：正在测量‘带宽抢占’效应...")
    print("-" * 50)
    
    for v in velocities:
        # 1. 外部带宽消耗（移动占用的资源）
        ext_cost = v 
        
        # 2. 剩余带宽分配给内部演化（内禀时钟）
        # 根据 N.E.A. 能量守恒 (勾股定理形式的带宽划分):
        # B^2 = f_internal^2 + f_external^2
        remaining_sq = bandwidth_limit**2 - ext_cost**2
        f_internal = np.sqrt(max(0, remaining_sq))
        
        internal_ticks.append(f_internal)
        
        if v % 200 == 0 or v == velocities[-1]:
            print(f"速度 v = {v:>6.1f} | 内部时钟频率 f = {f_internal:>6.1f}")

    # 3. 理论洛伦兹对比 (Standard SR)
    # f' = f0 * sqrt(1 - v^2/c^2)
    theoretical_f = bandwidth_limit * np.sqrt(1 - (velocities/bandwidth_limit)**2)

    # 绘图：攻城略地的实证
    plt.figure(figsize=(10, 6))
    plt.plot(velocities, internal_ticks, 'ro', markersize=4, label='N.E.A. Bandwidth Allocation')
    plt.plot(velocities, theoretical_f, 'b-', alpha=0.6, label='Einstein Lorentz Factor (Theoretical)')
    
    plt.xlabel('Velocity v (External Bandwidth Cost)')
    plt.ylabel('Internal Clock Speed (Time Rate)')
    plt.title('Time Dilation as a Resource Scheduling Result')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 计算误差
    error = np.mean(np.abs(np.array(internal_ticks) - theoretical_f))
    print("-" * 50)
    print(f">>> 结算完成。N.E.A. 与爱因斯坦理论平均偏差: {error:.2e}")
    plt.savefig('figures/sr_bandwidth.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    simulate_time_dilation()