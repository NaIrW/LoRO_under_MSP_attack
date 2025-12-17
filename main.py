import torch
from transformers import AutoModelForCausalLM
import numpy as np


def generate_loro_strict(in_dim, out_dim, r=30, m=2, noise_mag=1, device="cpu"):
    """
    B = rand(...) * noise_mag
    A = rand(...) * noise_mag
    D = (B @ A).T
    """
    D_total = torch.zeros(out_dim, in_dim, device=device)
    
    # 模拟 m=2 的 Factor Multiplexing
    for k in range(m):
        # 使用 Gaussian (randn) 以符合论文理论假设
        B = torch.randn(in_dim, r, device=device) * noise_mag
        A = torch.randn(r, out_dim, device=device) * noise_mag
        
        # Random alpha (scalar mixing)
        alpha = torch.randn(1, device=device).item()
        
        # 累加
        component = (B @ A).T * alpha
        D_total += component
        
    return D_total

def run_strict_attack():
    # 配置
    base_id = "Qwen/Qwen2.5-3B-Instruct"
    
    layer_idx = 29
    proj_name = "up_proj"
    
    device = "cpu"
    
    print(f"Loading Base (Source of Truth): {base_id}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_id, 
        torch_dtype=torch.float16,
        device_map=device
    )
    base_layer = base_model.model.layers[layer_idx]
    base_linear = getattr(base_layer.mlp, proj_name)

    W_base = base_linear.weight.detach().to(torch.float32)

    print("\n[Simulation] Generating Large Random Delta (Sim < 0.4 Scenario)...")
    
    Delta_True = torch.randn_like(W_base) * 0.06
    
    # 构造 W_ft
    W_ft = W_base + Delta_True
    
    # 0. check W sim
    sim_w = torch.nn.functional.cosine_similarity(W_ft.flatten(), W_base.flatten(), dim=0).item()
    print(f"Sim of W_ft and W_base: {sim_w:.4f} (Expected to be low)")

    # 1. 计算 Signal Norm
    norm_delta = torch.norm(Delta_True).item()
    print(f"\n[Signal] Delta Norm: {norm_delta:.4f}")

    # 2. 生成 LoRO 噪声 (Strict Mode)
    # m=2, r=30, noise=1 (保持原设置，噪声幅度 1.0)
    Loro_R = 30
    Loro_M = 2
    Effective_Rank = Loro_R * Loro_M
    
    D_noise = generate_loro_strict(2048, 11008, r=Loro_R, m=Loro_M, noise_mag=1, device=device)
    norm_noise = torch.norm(D_noise).item()
    
    print(f"[Noise]  Mask D Norm:  {norm_noise:.4f}")
    print(f"[Ratio]  Noise/Signal: {norm_noise/norm_delta:.4f}")
    
    # 3. Obfuscation
    # W_obfus = W_ft + Noise = (W_base + Delta) + Noise
    # 攻击者拿到 W_obfus 和 W_base，计算 Diff = Delta + Noise
    W_obfus = W_ft + D_noise
    Diff_Attack = W_obfus - W_base # 这一步就是 Delta_True + D_noise
    
    # 4. SVD 攻击
    print("\n>> Running SVD Attack...")
    U, S, Vh = torch.linalg.svd(Diff_Attack, full_matrices=False)
    
    print("\n[Spectrum Analysis] Top 10 Singular Values:")
    print(S[:10].tolist())
    print(f"Value at rank {Effective_Rank-1}: {S[Effective_Rank-1]:.4f}")
    print(f"Value at rank {Effective_Rank}:   {S[Effective_Rank]:.4f}")
    print(f"Value at rank {Effective_Rank+1}: {S[Effective_Rank+1]:.4f}")
    
    # 剔除前 K 个奇异值 (即 LoRO 的低秩噪声)
    remove_rank = Effective_Rank
    S_attack = S.clone()
    S_attack[:remove_rank] = 0.0
    
    # 重构 Delta
    Delta_Rec = U @ torch.diag(S_attack) @ Vh
    
    # 5. 评估
    diff = torch.norm(Delta_True - Delta_Rec).item()
    rel_err = diff / norm_delta
    sim_delta = torch.nn.functional.cosine_similarity(Delta_True.flatten(), Delta_Rec.flatten(), dim=0).item()
    
    print("-" * 40)
    print("Attack Results (Random Full-Rank Delta):")
    print(f"Cosine Similarity (Recovered vs True): {sim_delta:.6f}")
    print(f"Relative Error:    {rel_err:.6f}")
    print("-" * 40)

if __name__ == "__main__":
    run_strict_attack()


""" res:

[Simulation] Generating Large Random Delta (Sim < 0.4 Scenario)...
Sim of W_ft and W_base: 0.4173 (Expected to be low)

[Signal] Delta Norm: 284.6178
[Noise]  Mask D Norm:  29180.6465
[Ratio]  Noise/Signal: 102.5257

>> Running SVD Attack...

[Spectrum Analysis] Top 10 Singular Values:
[5459.59521484375, 5418.390625, 5400.14892578125, 5318.54296875, 5249.0634765625, 5201.1982421875, 5100.7099609375, 5077.2412109375, 5049.83251953125, 5019.4716796875]
Value at rank 59: 1920.2369
Value at rank 60:   8.9490
Value at rank 61: 8.9107
----------------------------------------
Attack Results (Random Full-Rank Delta):
Cosine Similarity (Recovered vs True): 0.984973
Relative Error:    0.185986
----------------------------------------
"""
