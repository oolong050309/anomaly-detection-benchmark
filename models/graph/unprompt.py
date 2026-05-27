"""UNPrompt 包装（IJCAI 2025）。

底层实现来自 vendor 包 ``models/graph/_vendor/unprompt/``。

UNPrompt 原始流程：
  1. 在源图上预训练 GCN 编码器（GRACE 自监督）
  2. 固定 GCN，在源图上训练 prompt + projection
  3. 迁移到目标图：用同一 prompt + projection 推理，节点级"completion sim"作为分数

本包装类的简化策略（"in-graph" 模式）：
  在目标图本身上做 GRACE 自监督预训练 + prompt-tuning，然后用 completionsim 输出节点
  异常分数。这是无监督的——不依赖 ``y`` 标签训练，但可用 ``y`` 评估。
  这一简化与原论文的"零样本跨图迁移"略有不同，但能让我们在不持有源图预训练
  权重的情况下端到端跑通。

接口契约：
- 输入：PyG ``torch_geometric.data.Data``（``x`` 节点特征，``edge_index`` 边）
- ``fit(graph)``：在该图上训练 GCN + prompt
- ``decision_function(graph) -> np.ndarray``：返回每节点异常分数（越大越异常）
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import GraphDetector


def _fix_torch_seed(seed: int | None) -> None:
    if seed is None:
        return
    try:
        import torch
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass
    np.random.seed(int(seed))


def _select_device(prefer_cuda: bool = True) -> str:
    try:
        import torch
        if prefer_cuda and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _pyg_to_normalized_adj(graph, device: str):
    """从 PyG Data 构造 UNPrompt 期望的两个稀疏邻接矩阵：
    - adj_withloop : 加自环并做对称归一化（用于 GCN）
    - adj_woself   : 不含自环并做对称归一化（用于 prompt 阶段无邻居 / 有邻居对照）
    """
    import scipy.sparse as sp
    import torch

    n = int(graph.x.shape[0])
    ei = graph.edge_index.cpu().numpy()
    src, dst = ei[0], ei[1]
    data = np.ones_like(src, dtype=np.float32)
    adj = sp.coo_matrix((data, (src, dst)), shape=(n, n)).tocsr()
    # 对称化（确保无向）
    adj = (adj + adj.T)
    adj.data = np.minimum(adj.data, 1.0)

    # 复用 vendor utils 里的 normalize_adj
    from models.graph._vendor.unprompt.utils import (
        normalize_adj,
        sparse_mx_to_torch_sparse_tensor,
    )

    diag_present = (adj.diagonal() > 0).all()
    if diag_present:
        adj_withloop_won = adj
        adj_woself = adj - sp.eye(n, format="csr")
    else:
        adj_withloop_won = adj + sp.eye(n, format="csr")
        adj_woself = adj

    adj_withloop = sparse_mx_to_torch_sparse_tensor(
        normalize_adj(adj_withloop_won)
    ).to(device)
    adj_woself = sparse_mx_to_torch_sparse_tensor(
        normalize_adj(adj_woself)
    ).to(device)
    return adj_withloop, adj_woself


def _grace_loss(z1, z2, tau: float = 0.5):
    """简化的 GRACE 对比损失（节点级）。

    z1, z2 是同一个图两次随机扰动后的节点 embedding。
    损失 = -log(对应位置的相似度 / 全行总相似度)。
    """
    import torch
    import torch.nn.functional as F

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.matmul(z1, z2.t()) / tau
    sim_exp = sim.exp()
    pos = sim_exp.diag()
    loss = -torch.log(pos / sim_exp.sum(dim=1).clamp_min(1e-12))
    return loss.mean()


class UNPromptDetector(GraphDetector):
    """UNPrompt 图异常检测（简化为同图预训练 + prompt-tuning）。

    Parameters
    ----------
    contamination : float, default=0.1
    random_state : int | None, default=42
    embedding_dim : int, default=128
        GCN 隐藏维度。
    unifeat : int, default=8
        SVD 降维到的统一特征维度（论文：跨图特征对齐）。
    edge_drop_prob : float, default=0.2
        GRACE 的边丢弃比例。
    feat_drop_prob : float, default=0.3
        GRACE 的特征丢弃比例。
    pretrain_epochs : int, default=100
        GCN 预训练轮数。
    prompt_epochs : int, default=100
        Prompt-tuning 轮数。
    numprompts : int, default=10
    lr : float, default=1e-3
    """

    def __init__(
        self,
        contamination: float = 0.1,
        random_state: int | None = 42,
        embedding_dim: int = 128,
        unifeat: int = 8,
        edge_drop_prob: float = 0.2,
        feat_drop_prob: float = 0.3,
        pretrain_epochs: int = 100,
        prompt_epochs: int = 100,
        numprompts: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        tau: float = 0.5,
    ) -> None:
        super().__init__(contamination=contamination, random_state=random_state)
        self.embedding_dim = embedding_dim
        self.unifeat = unifeat
        self.edge_drop_prob = edge_drop_prob
        self.feat_drop_prob = feat_drop_prob
        self.pretrain_epochs = pretrain_epochs
        self.prompt_epochs = prompt_epochs
        self.numprompts = numprompts
        self.lr = lr
        self.weight_decay = weight_decay
        self.tau = tau
        self._scores: np.ndarray | None = None

    def _fit(self, graph, y=None, **kwargs):
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("[UNPromptDetector] torch 未安装") from e

        from models.graph._vendor.unprompt.model import (
            GPFplusAtt,
            Model,
            Projection,
            SimplePrompt,
        )
        from models.graph._vendor.unprompt.utils import (
            completionloss,
            completionsim,
            x_svd,
        )

        _fix_torch_seed(self.random_state)
        device = _select_device()

        # 1. 特征统一降维到 unifeat（UNPrompt 的关键预处理）
        feats = graph.x.float()
        if feats.shape[1] >= self.unifeat:
            feats = x_svd(feats, self.unifeat)
        else:
            # 输入维度小于 unifeat 时填充 0
            pad = torch.zeros(feats.shape[0], self.unifeat - feats.shape[1])
            feats = torch.cat([feats, pad], dim=1)
        bn = nn.BatchNorm1d(feats.shape[1], affine=False)
        feats = bn(feats).to(device)

        # 2. 构造两个邻接矩阵
        adj_withloop, adj_woself = _pyg_to_normalized_adj(graph, device)

        # 3. 预训练 GCN encoder（简化 GRACE：随机 drop edge / feature）
        model = Model(self.unifeat, self.embedding_dim, "prelu").to(device)
        opt_pre = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        for ep in range(self.pretrain_epochs):
            model.train()
            opt_pre.zero_grad()

            # 视图 1：随机丢部分特征
            f1 = feats * (
                torch.rand(feats.shape, device=device) > self.feat_drop_prob
            ).float()
            # 视图 2：另一个独立 mask
            f2 = feats * (
                torch.rand(feats.shape, device=device) > self.feat_drop_prob
            ).float()

            z1 = model(f1, adj_withloop)
            z2 = model(f2, adj_withloop)
            loss = _grace_loss(z1, z2, tau=self.tau)
            loss.backward()
            opt_pre.step()

        # 4. Prompt-tuning：固定 model，训练 prompt + projection
        model.eval()
        if self.numprompts < 2:
            prompts = SimplePrompt(self.unifeat).to(device)
        else:
            prompts = GPFplusAtt(self.unifeat, self.numprompts).to(device)
        proj = Projection(self.embedding_dim).to(device)

        # 用伪标签：UNPrompt 的 completionloss 需要 ano_label。
        # 在简化版里我们没有标签，所以用全 0 占位（让 loss 把所有节点都视作"正常"，
        # 学一个让正常节点 emb_nei ≈ emb_mlp 的 prompt）。
        ano_label = torch.zeros(feats.shape[0], device=device)

        opt_prompt = torch.optim.Adam(
            list(prompts.parameters()) + list(proj.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )

        for ep in range(self.prompt_epochs):
            prompts.train()
            proj.train()
            opt_prompt.zero_grad()

            modified = prompts.add(feats)
            emb_nei = proj(model(modified, adj_woself))
            emb_mlp = proj(model(modified, None))
            loss = completionloss(emb_nei, emb_mlp, ano_label)
            loss.backward()
            opt_prompt.step()

        # 5. 推理：completionsim 作为节点异常分数
        prompts.eval()
        proj.eval()
        with torch.no_grad():
            modified = prompts.add(feats)
            emb_nei = proj(model(modified, adj_woself))
            emb_mlp = proj(model(modified, None))
            sim = completionsim(emb_mlp, emb_nei)  # 越大表示越相似（越正常）

        # 异常分数 = 1 - 归一化相似度（越大越异常）
        sim_np = sim.cpu().numpy().astype(np.float64).ravel()
        # 归一化到 [0, 1]
        s_min, s_max = sim_np.min(), sim_np.max()
        if s_max - s_min > 1e-12:
            sim_norm = (sim_np - s_min) / (s_max - s_min)
        else:
            sim_norm = np.zeros_like(sim_np)
        self._scores = 1.0 - sim_norm

    def _decision_function(self, graph):
        if self._scores is None:
            raise RuntimeError(
                f"[{type(self).__name__}] scores not available; call fit() first"
            )
        return self._scores
