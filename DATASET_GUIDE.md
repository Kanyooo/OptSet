# UOBGen 数据集生成指南

## 0. 统一生成基元
- **Haar 正交矩阵 `haar_orthogonal(n)`**：从标准正态矩阵进行 QR 分解，调整对角符号，得到均匀分布在正交群上的矩阵。
- **几何谱 `geometric_spectrum(n, \sigma_{\min}, \sigma_{\max})`**：返回满足几何级数、条件数为 \(\kappa = \sigma_{\max}/\sigma_{\min}\) 的特征值向量，用于构造对称矩阵或 Hessian。
- **Toeplitz 相关矩阵 `toeplitz_corr(n, \rho)`**：构造 \(\Sigma_{jk} = \rho^{|j-k|}\)，控制自相关强度。
- **SNR 校准 `calibrate_sigma_for_snr(y, m, \mathrm{SNR})`**：求解噪声方差 \(\sigma^2\)，使得 \(\mathrm{SNR} = \|y\|_2^2 / (m \sigma^2)\)。
- **列归一化 `column_normalize(A)`**：对设计矩阵按列归一化，确保优化问题的尺度一致。
- **块状图像 `make_blocks_2d(H, W, B)`**：生成分块常值图像，支持 TV 去噪的 ground truth。
- **随机图 `simple_ER_graph(n, p)`**：生成对称邻接矩阵，并给出 [0,1] 权重，为 SDP MaxCut 提供 Laplacian。

## A 类：光滑问题
### A1 强凸病态二次规划 (A1_QP)
\[
\min_x \; \tfrac{1}{2} x^T Q x - b^T x,\quad Q \succ 0.
\]
- 生成：Haar 正交 \(U\)，几何谱构造 \(Q = U \operatorname{diag}(\sigma) U^T\)，\(b \sim \mathcal{N}(0,I)\)。
- 旋钮：维度 \(n\)，条件数 \(\kappa\)，可选稀疏化。
- 记录：\(x^* = Q^{-1}b\)，\(\|Q\|_2\) 与 \(\kappa(Q)\)。

### A2 L2 正则化逻辑回归 (A2_LogReg)
\[
\min_x \; \frac{1}{m} \sum_{i=1}^m \log\bigl(1 + e^{-y_i a_i^T x}\bigr) + \frac{\lambda}{2}\|x\|_2^2.
\]
- 生成：设计矩阵行向量 \(a_i \sim \mathcal{N}(0, \Sigma)\)，\(\Sigma\) 为 Toeplitz；稀疏真实解 \(x^*\)，标签 \(y\) 通过 Bernoulli(sigmoid)。
- 旋钮：样本规模 \((m,n)\)，相关系数 \(\rho\)，信噪比，正则 \(\lambda\)。
- 处理：列归一化，保存参考解和噪声水平。

### A3 Rosenbrock 链 (A3_Rosenbrock)
\[
 f(x) = \sum_{i=1}^{n-1} \bigl[100(x_{i+1}-x_i^2)^2 + (1-x_i)^2\bigr].
\]
- 生成：提供标准函数、推荐初值偏移尺度 \(\sigma\)。
- 旋钮：维度 \(n\)，初值尺度。

### A4 等式约束 QP (A4_ECQP)
\[
\min \tfrac{1}{2} x^T Q x - b^T x \quad \text{s.t.} \quad A x = d.
\]
- 生成：\(Q\) 同 A1；随机 \(x^\dagger\)，满秩 \(A\)，\(d = Ax^\dagger\)。
- 旋钮：维度 \((n,p)\)，\(\kappa(Q)\)，\(\kappa(AA^T)\)。
- 记录：KKT 可行性残差。

### A5 信赖域子问题 (A5_TRS)
\[
\min \tfrac{1}{2} x^T H x + g^T x \quad \text{s.t.} \; \|x\|_2 \le \Delta.
\]
- 生成：构造含指定负特征比例的对称矩阵 \(H\)；向量 \(g\) 与最小特征向量夹角控制在 \([60^\circ, 89^\circ]\)。
- 旋钮：负谱比例、半径 \(\Delta\)、夹角 \(\theta\)。

### A6 盒约束非凸 QP (A6_BoxQP)
- 目标：\( \min \tfrac{1}{2} x^T H x + c^T x \) s.t. \(l \le x \le u\)。
- 生成：控制负谱强度，设置上下界宽度与角点密度。
- 旋钮：负谱幅度、盒宽、角点密度。

## B 类：非光滑 / 复合
### B1 LASSO (B1_LASSO)
\[
\min \tfrac{1}{2}\|Ax - y\|_2^2 + \lambda\|x\|_1.
\]
- 生成：矩阵同 A2；稀疏 \(x^*\)，噪声按 SNR 校准，\(\lambda = \alpha \sigma \sqrt{2\log n}\)。
- 旋钮：稀疏度、\(\rho\)、SNR、\(\lambda\) 系数。

### B2 Elastic Net (B2_ElasticNet)
\[
\min \tfrac{1}{2}\|Ax - y\|_2^2 + \lambda_1\|x\|_1 + \tfrac{\lambda_2}{2}\|x\|_2^2.
\]
- 旋钮：\((\lambda_1, \lambda_2)\) 比例、\(\rho\)、SNR。

### B3 线性 SVM (B3_SVM)
\[
\min_x \frac{1}{m}\sum_{i=1}^m \max(0, 1 - y_i a_i^T x) + \frac{\gamma}{2}\|x\|_2^2.
\]
- 生成：两类高斯簇，均值距离控制重叠，列归一化。
- 旋钮：簇距离 \(\|\mu\|\)、\(\rho\)、正则 \(\gamma\)。

### B4 各向同性 TV 去噪 (B4_TV)
- 目标：\( \min_x \tfrac{1}{2}\|x - y\|_2^2 + \lambda \sum_p \sqrt{(D_x x)_p^2 + (D_y x)_p^2} \)。
- 生成：分块常值图像，添加高斯噪声，保存离散梯度算子。
- 旋钮：图像分辨率、块数、噪声方差、\(\lambda\)。

### B5 Group Lasso (B5_GroupLasso)
- 目标：\( \min \tfrac{1}{2}\|Ax - y\|_2^2 + \lambda \sum_k \|x_{G_k}\|_2 \)。
- 生成：构造组划分（均匀或长尾），组稀疏真解，噪声按 SNR。
- 旋钮：组大小分布、组间相关、\(\lambda\)。

### B6 非凸稀疏惩罚 Logistic (B6_NC_Sparse)
- 目标：逻辑损失加 SCAD/MCP 惩罚。
- 生成：与 A2 类似的特征与标签，参数 \(a\)、\(\lambda\) 控制惩罚强度。
- 旋钮：非凸参数 \(a\)、\(\lambda\)、SNR、\(\rho\)。

## C 类：变分不等式 / 补集
### C1 线性单调 VI (C1_VI)
- 条件：求 \(x\in K\) 使 \(\langle Qx + c, z - x \rangle \ge 0\)。
- 生成：\(Q\) 具有下界 \(\mu\)，条件数可控；可选盒约束或单纯形。
- 旋钮：强单调度 \(\mu\)、\(\kappa(Q)\)、集合类型。

### C2 线性互补问题 (C2_LCP)
- 求 \(z \ge 0, w = Mz + q \ge 0, z^T w = 0\)。
- 生成：\(M = S^TS + \delta I\) 保证 P-矩阵，控制 \(\delta\) 逼近奇异。
- 旋钮：条件数、\(\delta\)。

### C3 简单 MPCC (C3_MPCC)
- 目标：\(\min \tfrac{1}{2}\|x\|_2^2\) s.t. \(s = Ax + b, y\ge0, s\ge0, y^Ts=0\)。
- 旋钮：互补对数 \(p\)，偏移 \(b\) 规模控制退化程度。

## D 类：锥规划视角
### D1 SOCP 鲁棒回归 (D1_SOCP)
- 目标：\(\min t \) s.t. \(\|Ax - y\|_2 \le t, \|x\|_2 \le R\)。
- 生成：设计矩阵与噪声同 B1，保存半径与观测残差。
- 旋钮：半径 \(R\)、条件数、SNR。

### D2 Basis Pursuit (D2_BP)
- 目标：\(\min \|x\|_1\) s.t. \(Ax = y\), \(m<n\)。
- 生成：稀疏真解，列归一化设计矩阵，确保可行性。
- 旋钮：稀疏度、欠定比、相关结构。

### D3 MaxCut SDP 松弛 (D3_SDP)
- 目标：\(\min \langle L, X \rangle\) s.t. \(X \succeq 0, \operatorname{diag}(X)=1\)。
- 生成：Erdős–Rényi 图的 Laplacian，权重可选分布。
- 旋钮：图规模 \(n\)、边概率 \(p\)、权重分布。

## 套件规模与建议网格
| Problem | S | M | L | 主要网格 |
| --- | --- | --- | --- | --- |
| A1_QP | n=200 | n=2000 | n=50000 | \(\kappa=10,10^3,10^5\) |
| A2_LogReg | (5k,500) | (50k,5k) | (1M,50k) | \(\rho\in\{0,0.5,0.9\}, \lambda\in[10^{-4},10^{-1}]\) |
| A3_Rosenbrock | 100 | 1000 | 5000 | 初值尺度 \(0.1,1,5\) |
| A4_ECQP | (1k,50) | (5k,200) | (50k,500) | \(\kappa(Q)=10,10^3\) |
| A5_TRS | 200 | 2000 | 10000 | neg ratio=0.05/0.2, \(\theta=60^\circ,89^\circ\) |
| A6_BoxQP | 1k | 10k | 50k | 盒宽=1,10 |
| B1_LASSO | (2k,4k) | (50k,50k) | (500k,200k) | 稀疏度 1%,5%,10% |
| B2_ElasticNet | 同 B1 | 同 B1 | 同 B1 | \((\lambda_1,\lambda_2)\) 比例 |
| B3_SVM | (10k,1k) | (100k,10k) | (1M,50k) | \(\|\mu\|=0.5,1,2\) |
| B4_TV | 256^2 | 512^2 | 1024^2 | 块数 8,16,32 |
| B5_GroupLasso | (10k,5k) | (100k,20k) | (500k,50k) | 组大小均匀/长尾 |
| B6_NC_Sparse | (20k,2k) | (200k,10k) | (1M,50k) | \(a=3,3.7,5\) |
| C1_VI | 500 | 5000 | 50000 | \(\mu=0,10^{-3},10^{-1}\) |
| C2_LCP | 200 | 2000 | 10000 | \(\delta=10^{-1},10^{-3},10^{-5}\) |
| C3_MPCC | (500,200) | (2000,800) | (10000,3000) | \(\|b\|\) 尺度 |
| D1_SOCP | 同 B1 | 同 B1 | 同 B1 | \(R\) 与 SNR 网格 |
| D2_BP | (500,2k) | (5k,20k) | (20k,100k) | 稀疏度、欠定比 |
| D3_SDP | 40 | 100 | 200 | \(p=0.1,0.3,0.5\) |

## 校验与记录建议
- 每个实例保存 `meta.json`：包含问题 ID、名称、随机种子、维度、旋钮参数、诊断指标（条件数、谱范数、SNR 等）。
- `data.npz` 保存所有数值数组（矩阵、向量、图结构、梯度算子等）。
- `README.md` 简述数学模型、参数设置及参考解信息。
- 推荐记录：KKT 残差、约束违背、参考解质量、随机种子，确保可复现性。
- 校验：检查 SPD、条件数、SNR 偏差、互补残差、图 Laplacian 行和为零等，以便 `uobgen verify` 自动执行。
