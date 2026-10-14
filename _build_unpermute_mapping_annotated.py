# ============================================================================
# _build_unpermute_mapping 逐行注释版
#
# 来源文件: megatron/core/transformer/moe/token_dispatcher.py (第 1028-1047 行)
# 当前分支: core_r0.13.0_moe_alltoallsdc (commit 25446c1b4)
# 所属类  : SDCMoEAlltoAllTokenDispatcher (基于 EP x ETP 联合 AlltoAll 的 MoE dispatcher)
#
# 调用链:
#   dispatch_preprocess()  L1153 -> self.reversed_local_input_permutation_mapping
#                                = self._build_unpermute_mapping(routing_map, moe_permute_fusion)
#   combine_postprocess()  L1313 -> unpermute(collapsed_tokens,
#                                self.reversed_local_input_permutation_mapping, ...)
#
# 背景知识:
#   MoE 前向要把 token 按路由结果重排(permute)成 expert-major 的 [M, H] 送给各 expert
#   计算, 算完再还原(unpermute)回 token-major 的 [N, H]。unpermute 需要一张"还原映射"
#   告诉它: [M, H] 的每一行应该回到 [N, H] 的哪个 token 位置。本函数就是在 dispatch
#   阶段根据 routing_map 预先把这张映射算好并缓存, combine 阶段直接使用。
# ============================================================================

def _build_unpermute_mapping(self, routing_map: torch.Tensor, fused: bool) -> torch.Tensor:
    """根据扩展前的路由关系，构建最终 unpermute 使用的索引。

    输入 `routing_map` 为 `[N, E]`，转置后的 `expert_to_token` 为 `[E, N]`。
    令 `M` 为其中 True 的数量，即 ETP 合并后的原始 token-expert 任务数；Dropless
    模式下通常有 `M = N * K`。

    非融合模式返回 `[M]` 的原始 token index，用于将 expert-major 的 `[M, H]`
    scatter-add 回 token-major 的 `[N, H]`。融合模式返回 `[E, N]` 的位置 mapping，
    被选中的位置记录其在 `[M, H]` 中的行号，未被选中的位置为 `-1`。
    """
    # N = token 数; routing_map 形状 [N, E], E = 全局 expert 数, True 表示该 token 路由到该 expert
    num_tokens = routing_map.shape[0]

    # bool() 确保是布尔掩码; .T 转置成 [E, N] 的 expert->token 视图(每个 expert 一行,
    # 列出选中了它的 token); .contiguous() 把转置后的非连续内存整理成连续内存,
    # 后续 reshape(-1) 展平/(-1 填充)都要求内存连续, 否则 view 会报错
    expert_to_token = routing_map.bool().T.contiguous()

    # ---------------- 融合模式(fused=True, 即 --moe-permute-fusion, 依赖 TE >= 2.1.0) ----------------
    if fused:
        # reshape(-1) 把 [E, N] 按行展开成一维(顺序: expert0 的 N 个位置, expert1 的 N 个位置...);
        # cumsum 做前缀和: 展平后第 i 个位置的累计值 = 从头到 i(含 i)共有多少个 True;
        # 减 1 把"第几个(从 1 数)"变成"从 0 开始的行号", 即该 token 在 [M, H] 中的目标行号。
        # 注意 False 位置的 cumsum 值会重复前一个 True 的计数, 但下一步会被 where 屏蔽掉
        positions = torch.cumsum(expert_to_token.reshape(-1), dim=0) - 1

        # reshape_as 把一维行号恢复成 [E, N] 布局;
        # torch.where(掩码, 行号, -1): 掩码为 True 的位置保留行号(0 ~ M-1),
        # False 的位置统一填 -1 占位 —— 最终返回 [E, N] 的位置映射(position mapping)
        return torch.where(expert_to_token, positions.reshape_as(expert_to_token), -1)

    # ---------------- 非融合模式(fused=False, 纯 PyTorch 实现) ----------------
    # arange 生成 [0, 1, ..., N-1] 即全部 token 的原始行号, 放在 routing_map 所在 device 上;
    # unsqueeze(0) 把 [N] 变成 [1, N], 为下面广播成 [E, N] 做准备
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0)

    # expand(E, -1) 把 [1, N] 广播成 [E, N]: 每个 expert 行都持有一份相同的完整 token 行号表;
    # expand 是视图操作, 不复制内存, 因此很轻量
    token_indices = token_indices.expand(routing_map.shape[1], -1)

    # masked_select 按 [E, N] 掩码把所有 True 位置对应的 token 行号抽取出来, 得到 [M] 一维索引;
    # 抽取顺序为 expert-major(expert0 选完再选 expert1), 与 permute() 生成
    # sorted_indices 的算法完全一致(见 moe_utils.py L311-314);
    # unpermute() 非融合分支用它做 scatter_add_(0, ...) 把 [M, H] 撒回 [N, H]
    return token_indices.masked_select(expert_to_token)


# ============================================================================
# 两种模式的输出对比(以 N=4, E=2, K=2 为例):
#   routing_map      = [[T, F],   # token0 -> expert0
#                       [F, T],   # token1 -> expert1
#                       [T, T],   # token2 -> expert0 + expert1
#                       [F, T]]   # token3 -> expert1
#   expert_to_token  = [[T, F, T, F],     # expert0 选中 token0, token2
#                       [F, T, T, T]]     # expert1 选中 token1, token2, token3
#   M = True 总数 = 5
#
#   fused=False 返回 [M]=[5]      : [0, 2, 1, 2, 3]
#     (expert0: token0, token2; expert1: token1, token2, token3 —— 按 expert-major 排列)
#     -> 消费方: unpermute() 非融合分支, output.scatter_add_(0, mapping, permuted_tokens)
#
#   fused=True  返回 [E, N]=[2,4]: [[ 0, -1,  1, -1],     # 行号 = 该 token 在 [M, H] 中的行
#                                    [-1,  2,  3,  4]]    # -1 表示该 expert 未选中此 token
#     -> 消费方: TE 的 fused_unpermute(permuted_tokens, mapping, merging_probs=probs,
#                restore_shape=...) 一次完成"乘 probs 加权 + scatter 还原"
# ============================================================================
