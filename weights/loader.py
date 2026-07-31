import torch


def load_gpt2_weights(model):
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_sd = hf.state_dict()

    with torch.no_grad():
        model.token_emb.weight.copy_(hf_sd["transformer.wte.weight"])
        model.pos_emb.weight.copy_(hf_sd["transformer.wpe.weight"])
        model.final_norm.weight.copy_(hf_sd["transformer.ln_f.weight"])
        model.final_norm.bias.copy_(hf_sd["transformer.ln_f.bias"])

        for i, block in enumerate(model.blocks):
            # HF GPT-2 stores Q,K,V as one combined matrix and uses Conv1D (transposed weights)
            qkv_w = hf_sd[f"transformer.h.{i}.attn.c_attn.weight"]
            qkv_b = hf_sd[f"transformer.h.{i}.attn.c_attn.bias"]
            q_w, k_w, v_w = qkv_w.chunk(3, dim=1)
            q_b, k_b, v_b = qkv_b.chunk(3)

            block.mha.q_proj.weight.copy_(q_w.t())
            block.mha.q_proj.bias.copy_(q_b)
            block.mha.k_proj.weight.copy_(k_w.t())
            block.mha.k_proj.bias.copy_(k_b)
            block.mha.v_proj.weight.copy_(v_w.t())
            block.mha.v_proj.bias.copy_(v_b)

            out_w = hf_sd[f"transformer.h.{i}.attn.c_proj.weight"]
            block.mha.out_proj.weight.copy_(out_w.t())
            block.mha.out_proj.bias.copy_(hf_sd[f"transformer.h.{i}.attn.c_proj.bias"])

            block.norm1.weight.copy_(hf_sd[f"transformer.h.{i}.ln_1.weight"])
            block.norm1.bias.copy_(hf_sd[f"transformer.h.{i}.ln_1.bias"])
            block.norm2.weight.copy_(hf_sd[f"transformer.h.{i}.ln_2.weight"])
            block.norm2.bias.copy_(hf_sd[f"transformer.h.{i}.ln_2.bias"])

            fc_w = hf_sd[f"transformer.h.{i}.mlp.c_fc.weight"]
            block.ff.expand.weight.copy_(fc_w.t())
            block.ff.expand.bias.copy_(hf_sd[f"transformer.h.{i}.mlp.c_fc.bias"])

            proj_w = hf_sd[f"transformer.h.{i}.mlp.c_proj.weight"]
            block.ff.down_proj.weight.copy_(proj_w.t())
            block.ff.down_proj.bias.copy_(hf_sd[f"transformer.h.{i}.mlp.c_proj.bias"])

        model.lm_head.weight.copy_(model.token_emb.weight)

    print("Weights loaded!")
