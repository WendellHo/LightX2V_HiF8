from functools import partial

import torch

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import *
from lightx2v_platform.base.global_var import AI_DEVICE

from .triton_ops import fuse_scale_shift_kernel
from .utils import apply_wan_rope_with_chunk, apply_wan_rope_with_flashinfer, apply_wan_rope_with_torch, apply_wan_rope_with_torch_naive

torch_device_module = getattr(torch, AI_DEVICE)


def modulate(x, scale, shift):
    return x * (1 + scale.squeeze()) + shift.squeeze()


class WanTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        hif8_runtime = config.get("hif8_runtime", {}) if isinstance(config, dict) else {}
        self.hiband_runtime_enabled = bool(
            hif8_runtime.get("hiband_enabled", False)
        ) if isinstance(hif8_runtime, dict) else False
        self.hiband_runtime_group_act_scale_enabled = bool(
            hif8_runtime.get("hiband_runtime_group_act_scale_enabled", False)
        ) if isinstance(hif8_runtime, dict) else False
        self.hiband_group_source = (
            str(hif8_runtime.get("hiband_group_source", "htg")).lower()
            if isinstance(hif8_runtime, dict) else "htg"
        )
        self.hiband_fallback_global_act_scale = bool(
            hif8_runtime.get("hiband_fallback_global_act_scale", True)
        ) if isinstance(hif8_runtime, dict) else True
        self._hiband_tensor_context = {}
        self._current_htg_group_idx = 0
        self.task = config["task"]
        self.attention_type = config.get("attention_type", "flash_attn2")
        self.self_attn_1_type = config.get("self_attn_1_type", "flash_attn2")
        self.cross_attn_1_type = config.get("cross_attn_1_type", "flash_attn2")
        self.cross_attn_2_type = config.get("cross_attn_2_type", "flash_attn2")
        self.blocks_num = config["num_layers"]
        self.phases_num = 3
        self.has_post_adapter = False
        self.num_heads = config["num_heads"]
        self.head_dim = config["dim"] // config["num_heads"]
        self.window_size = config.get("window_size", (-1, -1))
        self.parallel_attention = None
        if self.config.get("modulate_type", "triton") == "triton":
            self.modulate_func = fuse_scale_shift_kernel
        else:
            self.modulate_func = modulate
        rope_funcs = {
            "flashinfer": apply_wan_rope_with_flashinfer,
            "torch": apply_wan_rope_with_torch,
            "torch_naive": apply_wan_rope_with_torch_naive,
        }
        rope_type = self.config.get("rope_type", "flashinfer")
        # Try to get rope function from registry first (for platform-specific implementations)
        if rope_type in ROPE_REGISTER:
            rope_class = ROPE_REGISTER[rope_type]
            self.rope_instance = rope_class()

            # Create a wrapper function that matches the expected signature
            def rope_wrapper(xq, xk, cos_sin_cache):
                return self.rope_instance.apply(xq, xk, cos_sin_cache)

            rope_func = rope_wrapper
        else:
            # Fallback to hardcoded functions
            rope_func = rope_funcs.get(rope_type, apply_wan_rope_with_torch)
        if self.config.get("rope_chunk", False):
            self.apply_rope_func = partial(apply_wan_rope_with_chunk, chunk_size=self.config.get("rope_chunk_size", 100), rope_func=rope_func)
        else:
            self.apply_rope_func = rope_func
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
            self.seq_p_fp8_comm = self.config["parallel"].get("seq_p_fp8_comm", False)
            self.seq_p_fp4_comm = self.config["parallel"].get("seq_p_fp4_comm", False)
            self.enable_head_parallel = self.config["parallel"].get("seq_p_head_parallel", False)
            self.seq_p_tensor_fusion = self.config["parallel"].get("seq_p_tensor_fusion", False)
        else:
            self.seq_p_group = None
            self.seq_p_fp8_comm = False
            self.seq_p_fp4_comm = False
            self.enable_head_parallel = False
            self.seq_p_tensor_fusion = False
        self.infer_func = self.infer_without_offload

        self.cos_sin = None

    @torch.no_grad()
    def reset_post_adapter_states(self):
        pass

    def reset_infer_states(self):
        self.self_attn_cu_seqlens_qkv = None
        self.cross_attn_cu_seqlens_q = None
        self.cross_attn_cu_seqlens_kv = None
        self.cross_attn_cu_seqlens_kv_img = None
        self._current_htg_group_idx = 0
        if self.has_post_adapter:
            self.reset_post_adapter_states()

    def _get_progress(self):
        infer_steps = getattr(self.scheduler, "infer_steps", 0)
        step_index = getattr(self.scheduler, "step_index", 0)
        if infer_steps is None or infer_steps <= 1:
            return 0.0
        return float(step_index) / float(max(infer_steps - 1, 1))

    def _get_htg_group_idx(self, boundaries):
        if boundaries is None:
            return 0
        if boundaries.numel() == 0:
            return 0
        progress = self._get_progress()
        idx = 0
        for b in boundaries.flatten():
            if progress > float(b.item()):
                idx += 1
            else:
                break
        return idx

    def _select_group_tensor(self, tensor, group_idx):
        if tensor is None:
            return None
        if tensor.dim() == 1:
            return tensor
        if tensor.shape[0] == 0:
            return None
        group_idx = max(0, min(group_idx, tensor.shape[0] - 1))
        return tensor[group_idx]

    def _get_phase_tensor(self, phase, name):
        if not hasattr(phase, name):
            return None
        tensor_holder = getattr(phase, name)
        return getattr(tensor_holder, "tensor", None)

    def _resolve_hiband_tensor(self, global_tensor, group_tensor=None, group_idx=None):
        if (
            self.hiband_runtime_enabled
            and self.hiband_runtime_group_act_scale_enabled
            and self.hiband_group_source == "htg"
            and group_tensor is not None
            and group_idx is not None
        ):
            selected = self._select_group_tensor(group_tensor, group_idx)
            if selected is not None:
                return selected
            if not self.hiband_fallback_global_act_scale:
                return None
        return global_tensor

    def _mm_apply_with_group_bias(self, mm_module, input_tensor, grouped_bias_tensor, group_idx):
        if self.hiband_runtime_enabled:
            hiband_scale_name = getattr(mm_module, "_hiband_scale_name", None)
            hiband_tensor = (
                getattr(self, "_hiband_tensor_context", {}).get(hiband_scale_name)
                if hiband_scale_name else None
            )
            if hiband_tensor is not None:
                mm_module.hiband_act_scale = hiband_tensor
            elif hasattr(mm_module, "hiband_act_scale"):
                mm_module.hiband_act_scale = None

        if grouped_bias_tensor is None:
            return mm_module.apply(input_tensor)

        grouped_bias = self._select_group_tensor(grouped_bias_tensor, group_idx)
        if grouped_bias is None:
            return mm_module.apply(input_tensor)

        original_bias = getattr(mm_module, "bias", None)
        try:
            mm_module.bias = grouped_bias
            return mm_module.apply(input_tensor)
        finally:
            if original_bias is not None:
                mm_module.bias = original_bias
            else:
                mm_module.bias = None

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        self.cos_sin = pre_infer_out.cos_sin
        self.reset_infer_states()
        x = self.infer_main_blocks(weights.blocks, pre_infer_out)
        return self.infer_non_blocks(weights, x, pre_infer_out.embed)

    def infer_main_blocks(self, blocks, pre_infer_out):
        x = self.infer_func(blocks, pre_infer_out.x, pre_infer_out)
        return x

    def infer_non_blocks(self, weights, x, e):
        if e.dim() == 2:
            modulation = weights.head_modulation.tensor  # 1, 2, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:  # For Diffustion forcing
            modulation = weights.head_modulation.tensor.unsqueeze(2)  # 1, 2, seq, dim
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
            e = [ei.squeeze(1) for ei in e]

        x = weights.norm.apply(x)

        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype)
        x.mul_(1 + e[1].squeeze()).add_(e[0].squeeze())
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.infer_dtype)

        x = weights.head.apply(x)

        if self.clean_cuda_cache:
            del e
            torch_device_module.empty_cache()
        return x

    def infer_without_offload(self, blocks, x, pre_infer_out):
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x = self.infer_block(blocks[block_idx], x, pre_infer_out)
        return x

    def infer_block(self, block, x, pre_infer_out):
        if hasattr(block.compute_phases[0], "before_proj") and block.compute_phases[0].before_proj.weight is not None:
            x = block.compute_phases[0].before_proj.apply(x) + pre_infer_out.x

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.pre_process(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )
        y_out = self.infer_self_attn(
            block.compute_phases[0],
            x,
            shift_msa,
            scale_msa,
        )
        x, attn_out = self.infer_cross_attn(
            block.compute_phases[1],
            x,
            pre_infer_out.context,
            y_out,
            gate_msa,
        )
        y = self.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
        x = self.post_process(x, y, c_gate_msa, pre_infer_out)
        if hasattr(block.compute_phases[2], "after_proj"):
            pre_infer_out.adapter_args["hints"].append(block.compute_phases[2].after_proj.apply(x))

        if self.has_post_adapter:
            x = self.infer_post_adapter(block.compute_phases[3], x, pre_infer_out)

        return x

    def pre_process(self, modulation, embed0):
        if embed0.dim() == 3 and embed0.shape[2] == 1:
            modulation = modulation.tensor.unsqueeze(2)
            embed0 = (modulation + embed0).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [ei.squeeze(1) for ei in embed0]
        else:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (modulation.tensor + embed0).chunk(6, dim=1)

        if self.clean_cuda_cache:
            del embed0
            torch_device_module.empty_cache()

        return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

    def infer_self_attn(self, phase, x, shift_msa, scale_msa):
        cos_sin = self.cos_sin
        use_htg_norm1 = (
            hasattr(phase, "htg_norm1_weight")
            and hasattr(phase, "htg_norm1_bias")
            and phase.htg_norm1_weight.tensor is not None
            and phase.htg_norm1_bias.tensor is not None
        )
        if use_htg_norm1:
            boundaries = (
                phase.htg_norm1_boundaries.tensor
                if hasattr(phase, "htg_norm1_boundaries")
                else None
            )
            group_idx = self._get_htg_group_idx(boundaries)
            htg_weight = self._select_group_tensor(phase.htg_norm1_weight.tensor, group_idx)
            htg_bias = self._select_group_tensor(phase.htg_norm1_bias.tensor, group_idx)
            norm1_weight = (1 + scale_msa.squeeze()) * htg_weight
            norm1_bias = (shift_msa.squeeze() - 1.0) * htg_weight + htg_bias
            norm1_out = phase.norm1.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm1_out = norm1_out.to(self.sensitive_layer_dtype)
            norm1_out.mul_(norm1_weight).add_(norm1_bias)
        elif hasattr(phase, "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa.squeeze()) * phase.smooth_norm1_weight.tensor
            norm1_bias = (shift_msa.squeeze() - 1.0) * phase.smooth_norm1_weight.tensor + phase.smooth_norm1_bias.tensor
            norm1_out = phase.norm1.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm1_out = norm1_out.to(self.sensitive_layer_dtype)
            norm1_out.mul_(norm1_weight).add_(norm1_bias)
        else:
            norm1_out = phase.norm1.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm1_out = norm1_out.to(self.sensitive_layer_dtype)
            norm1_out = self.modulate_func(norm1_out, scale=scale_msa, shift=shift_msa).squeeze()

        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.infer_dtype)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q_group_bias = None
        k_group_bias = None
        v_group_bias = None
        if self.hiband_runtime_enabled:
            self._hiband_tensor_context = {
                "self_attn_q": self._resolve_hiband_tensor(
                    self._get_phase_tensor(phase, "self_attn_q_hiband_act_scale"),
                    self._get_phase_tensor(phase, "self_attn_q_hiband_group_act_scales"),
                    group_idx if use_htg_norm1 else None,
                ),
                "self_attn_k": self._resolve_hiband_tensor(
                    self._get_phase_tensor(phase, "self_attn_k_hiband_act_scale"),
                    self._get_phase_tensor(phase, "self_attn_k_hiband_group_act_scales"),
                    group_idx if use_htg_norm1 else None,
                ),
                "self_attn_v": self._resolve_hiband_tensor(
                    self._get_phase_tensor(phase, "self_attn_v_hiband_act_scale"),
                    self._get_phase_tensor(phase, "self_attn_v_hiband_group_act_scales"),
                    group_idx if use_htg_norm1 else None,
                ),
                "self_attn_o": self._resolve_hiband_tensor(
                    self._get_phase_tensor(phase, "self_attn_o_hiband_act_scale"),
                    self._get_phase_tensor(phase, "self_attn_o_hiband_group_act_scales"),
                    group_idx if use_htg_norm1 else None,
                ),
            }
            phase.self_attn_q._hiband_scale_name = "self_attn_q"
            phase.self_attn_k._hiband_scale_name = "self_attn_k"
            phase.self_attn_v._hiband_scale_name = "self_attn_v"
            phase.self_attn_o._hiband_scale_name = "self_attn_o"
        if use_htg_norm1:
            if hasattr(phase, "self_attn_q_htg_group_bias"):
                q_group_bias = phase.self_attn_q_htg_group_bias.tensor
            if hasattr(phase, "self_attn_k_htg_group_bias"):
                k_group_bias = phase.self_attn_k_htg_group_bias.tensor
            if hasattr(phase, "self_attn_v_htg_group_bias"):
                v_group_bias = phase.self_attn_v_htg_group_bias.tensor
        self._current_htg_group_idx = group_idx if use_htg_norm1 else 0

        q_proj = self._mm_apply_with_group_bias(
            phase.self_attn_q,
            norm1_out,
            q_group_bias,
            group_idx if use_htg_norm1 else 0,
        )
        k_proj = self._mm_apply_with_group_bias(
            phase.self_attn_k,
            norm1_out,
            k_group_bias,
            group_idx if use_htg_norm1 else 0,
        )
        v_proj = self._mm_apply_with_group_bias(
            phase.self_attn_v,
            norm1_out,
            v_group_bias,
            group_idx if use_htg_norm1 else 0,
        )

        q = phase.self_attn_norm_q.apply(q_proj).view(s, n, d)
        k = phase.self_attn_norm_k.apply(k_proj).view(s, n, d)
        v = v_proj.view(s, n, d)
        q, k = self.apply_rope_func(q, k, cos_sin)
        img_qkv_len = q.shape[0]
        if self.self_attn_cu_seqlens_qkv is None:
            if self.self_attn_1_type in ["flash_attn2", "flash_attn3"]:
                self.self_attn_cu_seqlens_qkv = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)
            else:
                self.self_attn_cu_seqlens_qkv = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32)

        if self.clean_cuda_cache:
            del norm1_out, shift_msa, scale_msa
            torch_device_module.empty_cache()

        attn_running_args = {
            "block_idx": self.block_idx,
            "scheduler": self.scheduler,
        }

        if self.config["seq_parallel"]:
            attn_out = phase.self_attn_1_parallel.apply(
                q=q,
                k=k,
                v=v,
                slice_qkv_len=img_qkv_len,
                cu_seqlens_qkv=self.self_attn_cu_seqlens_qkv,
                attention_module=phase.self_attn_1,
                attention_type=self.self_attn_1_type,
                seq_p_group=self.seq_p_group,
                use_fp8_comm=self.seq_p_fp8_comm,
                use_fp4_comm=self.seq_p_fp4_comm,
                use_tensor_fusion=self.seq_p_tensor_fusion,
                enable_head_parallel=self.enable_head_parallel,
                **attn_running_args,
            )
        else:
            attn_out = phase.self_attn_1.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self.self_attn_cu_seqlens_qkv,
                cu_seqlens_kv=self.self_attn_cu_seqlens_qkv,
                max_seqlen_q=img_qkv_len,
                max_seqlen_kv=img_qkv_len,
                **attn_running_args,
            )

        self_attn_o_htg_input_shift = self._get_phase_tensor(phase, "self_attn_o_htg_input_shift")
        self_attn_o_htg_input_scale = self._get_phase_tensor(phase, "self_attn_o_htg_input_scale")
        if self_attn_o_htg_input_shift is not None and self_attn_o_htg_input_scale is not None:
            self_attn_o_boundaries = self._get_phase_tensor(phase, "self_attn_o_htg_group_boundaries")
            self_attn_o_group_idx = self._get_htg_group_idx(self_attn_o_boundaries)
            o_shift = self._select_group_tensor(self_attn_o_htg_input_shift, self_attn_o_group_idx)
            o_scale = self_attn_o_htg_input_scale
            attn_out = (attn_out - o_shift.to(attn_out.device, attn_out.dtype)) / o_scale.to(attn_out.device, attn_out.dtype)
            self_attn_o_group_bias = self._get_phase_tensor(phase, "self_attn_o_htg_group_bias")
            y = self._mm_apply_with_group_bias(
                phase.self_attn_o,
                attn_out,
                self_attn_o_group_bias,
                self_attn_o_group_idx,
            )
        else:
            y = self._mm_apply_with_group_bias(
                phase.self_attn_o,
                attn_out,
                None,
                group_idx if use_htg_norm1 else 0,
            )

        if self.clean_cuda_cache:
            del q, k, v, attn_out
            torch_device_module.empty_cache()

        return y

    def infer_cross_attn(self, phase, x, context, y_out, gate_msa):
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype) + y_out.to(self.sensitive_layer_dtype) * gate_msa.squeeze()
        else:
            x.add_(y_out * gate_msa.squeeze())

        cross_attn_boundaries = self._get_phase_tensor(phase, "cross_attn_htg_boundaries")
        if cross_attn_boundaries is not None:
            group_idx = self._get_htg_group_idx(cross_attn_boundaries)
        else:
            group_idx = self._current_htg_group_idx

        use_htg_cross_norm = (
            hasattr(phase, "cross_attn_htg_norm_weight")
            and hasattr(phase, "cross_attn_htg_norm_bias")
            and phase.cross_attn_htg_norm_weight.tensor is not None
            and phase.cross_attn_htg_norm_bias.tensor is not None
        )
        if use_htg_cross_norm:
            htg_weight = self._select_group_tensor(
                phase.cross_attn_htg_norm_weight.tensor, group_idx
            )
            htg_bias = self._select_group_tensor(
                phase.cross_attn_htg_norm_bias.tensor, group_idx
            )
            original_weight_diff = getattr(phase.norm3, "weight_diff", None)
            original_bias_diff = getattr(phase.norm3, "bias_diff", None)
            had_weight_diff = hasattr(phase.norm3, "weight_diff")
            had_bias_diff = hasattr(phase.norm3, "bias_diff")
            phase.norm3.weight_diff = htg_weight - phase.norm3.weight
            if phase.norm3.bias is None:
                phase.norm3.bias_diff = htg_bias
            else:
                phase.norm3.bias_diff = htg_bias - phase.norm3.bias
            try:
                norm3_out = phase.norm3.apply(x)
            finally:
                if had_weight_diff:
                    phase.norm3.weight_diff = original_weight_diff
                else:
                    delattr(phase.norm3, "weight_diff")
                if had_bias_diff:
                    phase.norm3.bias_diff = original_bias_diff
                elif hasattr(phase.norm3, "bias_diff"):
                    delattr(phase.norm3, "bias_diff")
        else:
            norm3_out = phase.norm3.apply(x)

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)
            if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
                context_img = context_img.to(self.infer_dtype)

        n, d = self.num_heads, self.head_dim
        q_group_bias = None
        if self.hiband_runtime_enabled:
            self._hiband_tensor_context = {
                "cross_attn_q": self._resolve_hiband_tensor(
                    self._get_phase_tensor(phase, "cross_attn_q_hiband_act_scale"),
                    self._get_phase_tensor(phase, "cross_attn_q_hiband_group_act_scales"),
                    group_idx,
                ),
                "cross_attn_k": self._resolve_hiband_tensor(
                    self._get_phase_tensor(phase, "cross_attn_k_hiband_act_scale"),
                    self._get_phase_tensor(phase, "cross_attn_k_hiband_group_act_scales"),
                    group_idx,
                ),
                "cross_attn_v": self._resolve_hiband_tensor(
                    self._get_phase_tensor(phase, "cross_attn_v_hiband_act_scale"),
                    self._get_phase_tensor(phase, "cross_attn_v_hiband_group_act_scales"),
                    group_idx,
                ),
                "cross_attn_o": self._resolve_hiband_tensor(
                    self._get_phase_tensor(phase, "cross_attn_o_hiband_act_scale"),
                    self._get_phase_tensor(phase, "cross_attn_o_hiband_group_act_scales"),
                    group_idx,
                ),
                "cross_attn_k_img": self._resolve_hiband_tensor(
                    self._get_phase_tensor(phase, "cross_attn_k_img_hiband_act_scale"),
                    self._get_phase_tensor(phase, "cross_attn_k_img_hiband_group_act_scales"),
                    group_idx,
                ),
                "cross_attn_v_img": self._resolve_hiband_tensor(
                    self._get_phase_tensor(phase, "cross_attn_v_img_hiband_act_scale"),
                    self._get_phase_tensor(phase, "cross_attn_v_img_hiband_group_act_scales"),
                    group_idx,
                ),
            }
            phase.cross_attn_q._hiband_scale_name = "cross_attn_q"
            phase.cross_attn_k._hiband_scale_name = "cross_attn_k"
            phase.cross_attn_v._hiband_scale_name = "cross_attn_v"
            phase.cross_attn_o._hiband_scale_name = "cross_attn_o"
            if hasattr(phase, "cross_attn_k_img"):
                phase.cross_attn_k_img._hiband_scale_name = "cross_attn_k_img"
            if hasattr(phase, "cross_attn_v_img"):
                phase.cross_attn_v_img._hiband_scale_name = "cross_attn_v_img"

        if use_htg_cross_norm and hasattr(phase, "cross_attn_q_htg_group_bias"):
            q_group_bias = phase.cross_attn_q_htg_group_bias.tensor

        q_proj = self._mm_apply_with_group_bias(
            phase.cross_attn_q,
            norm3_out,
            q_group_bias,
            group_idx,
        )
        k_proj = self._mm_apply_with_group_bias(phase.cross_attn_k, context, None, 0)
        v_proj = self._mm_apply_with_group_bias(phase.cross_attn_v, context, None, 0)
        q = phase.cross_attn_norm_q.apply(q_proj).view(-1, n, d)
        k = phase.cross_attn_norm_k.apply(k_proj).view(-1, n, d)
        v = v_proj.view(-1, n, d)

        if self.cross_attn_cu_seqlens_q is None:
            if self.cross_attn_1_type == "flash_attn2" or self.cross_attn_1_type == "flash_attn3":
                self.cross_attn_cu_seqlens_q = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True)
            else:
                self.cross_attn_cu_seqlens_q = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32)
        if self.cross_attn_cu_seqlens_kv is None:
            if self.cross_attn_1_type == "flash_attn2" or self.cross_attn_1_type == "flash_attn3":
                self.cross_attn_cu_seqlens_kv = torch.tensor([0, k.shape[0]]).cumsum(0, dtype=torch.int32).to(k.device, non_blocking=True)
            else:
                self.cross_attn_cu_seqlens_kv = torch.tensor([0, k.shape[0]]).cumsum(0, dtype=torch.int32)
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self.cross_attn_cu_seqlens_q,
            cu_seqlens_kv=self.cross_attn_cu_seqlens_kv,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True) and context_img is not None:
            k_img_proj = self._mm_apply_with_group_bias(phase.cross_attn_k_img, context_img, None, 0)
            v_img_proj = self._mm_apply_with_group_bias(phase.cross_attn_v_img, context_img, None, 0)
            k_img = phase.cross_attn_norm_k_img.apply(k_img_proj).view(-1, n, d)
            v_img = v_img_proj.view(-1, n, d)

            if self.cross_attn_cu_seqlens_kv_img is None:
                if self.cross_attn_2_type == "flash_attn2" or self.cross_attn_2_type == "flash_attn3":
                    self.cross_attn_cu_seqlens_kv_img = torch.tensor([0, k_img.shape[0]]).cumsum(0, dtype=torch.int32).to(k_img.device, non_blocking=True)
                else:
                    self.cross_attn_cu_seqlens_kv_img = torch.tensor([0, k_img.shape[0]]).cumsum(0, dtype=torch.int32)

            img_attn_out = phase.cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=self.cross_attn_cu_seqlens_q,
                cu_seqlens_kv=self.cross_attn_cu_seqlens_kv_img,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
            )
            attn_out.add_(img_attn_out)

            if self.clean_cuda_cache:
                del k_img, v_img, img_attn_out
                torch_device_module.empty_cache()

        cross_attn_o_htg_input_shift = self._get_phase_tensor(phase, "cross_attn_o_htg_input_shift")
        cross_attn_o_htg_input_scale = self._get_phase_tensor(phase, "cross_attn_o_htg_input_scale")
        if cross_attn_o_htg_input_shift is not None and cross_attn_o_htg_input_scale is not None:
            cross_attn_o_boundaries = self._get_phase_tensor(phase, "cross_attn_o_htg_group_boundaries")
            cross_attn_o_group_idx = self._get_htg_group_idx(cross_attn_o_boundaries)
            o_shift = self._select_group_tensor(cross_attn_o_htg_input_shift, cross_attn_o_group_idx)
            o_scale = cross_attn_o_htg_input_scale
            attn_out = (attn_out - o_shift.to(attn_out.device, attn_out.dtype)) / o_scale.to(attn_out.device, attn_out.dtype)
            cross_attn_o_group_bias = self._get_phase_tensor(phase, "cross_attn_o_htg_group_bias")
            attn_out = self._mm_apply_with_group_bias(phase.cross_attn_o, attn_out, cross_attn_o_group_bias, cross_attn_o_group_idx)
        else:
            attn_out = self._mm_apply_with_group_bias(phase.cross_attn_o, attn_out, None, 0)

        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context, context_img
            torch_device_module.empty_cache()
        return x, attn_out

    def infer_ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa):
        x.add_(attn_out)

        if self.clean_cuda_cache:
            del attn_out
            torch_device_module.empty_cache()

        use_htg_norm2 = (
            hasattr(phase, "htg_norm2_weight")
            and hasattr(phase, "htg_norm2_bias")
            and phase.htg_norm2_weight.tensor is not None
            and phase.htg_norm2_bias.tensor is not None
        )
        if use_htg_norm2:
            boundaries = (
                phase.htg_norm2_boundaries.tensor
                if hasattr(phase, "htg_norm2_boundaries")
                else None
            )
            group_idx = self._get_htg_group_idx(boundaries)
            htg_weight = self._select_group_tensor(phase.htg_norm2_weight.tensor, group_idx)
            htg_bias = self._select_group_tensor(phase.htg_norm2_bias.tensor, group_idx)
            norm2_weight = (1 + c_scale_msa.squeeze()) * htg_weight
            norm2_bias = (c_shift_msa.squeeze() - 1.0) * htg_weight + htg_bias
            norm2_out = phase.norm2.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm2_out = norm2_out.to(self.sensitive_layer_dtype)
            norm2_out.mul_(norm2_weight).add_(norm2_bias)
        elif hasattr(phase, "smooth_norm2_weight"):
            norm2_weight = (1 + c_scale_msa.squeeze()) * phase.smooth_norm2_weight.tensor
            norm2_bias = (c_shift_msa.squeeze() - 1.0) * phase.smooth_norm2_weight.tensor + phase.smooth_norm2_bias.tensor
            norm2_out = phase.norm2.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm2_out = norm2_out.to(self.sensitive_layer_dtype)
            norm2_out.mul_(norm2_weight).add_(norm2_bias)
        else:
            norm2_out = phase.norm2.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm2_out = norm2_out.to(self.sensitive_layer_dtype)
            norm2_out = self.modulate_func(norm2_out, scale=c_scale_msa, shift=c_shift_msa).squeeze()

        if self.sensitive_layer_dtype != self.infer_dtype:
            norm2_out = norm2_out.to(self.infer_dtype)

        ffn0_group_bias = None
        if self.hiband_runtime_enabled:
            self._hiband_tensor_context = {
                "ffn_0": self._resolve_hiband_tensor(
                    self._get_phase_tensor(phase, "ffn_0_hiband_act_scale"),
                    self._get_phase_tensor(phase, "ffn_0_hiband_group_act_scales"),
                    group_idx if use_htg_norm2 else None,
                ),
                "ffn_2": self._resolve_hiband_tensor(
                    self._get_phase_tensor(phase, "ffn_2_hiband_act_scale"),
                    self._get_phase_tensor(phase, "ffn_2_hiband_group_act_scales"),
                    group_idx if use_htg_norm2 else None,
                ),
            }
            phase.ffn_0._hiband_scale_name = "ffn_0"
            phase.ffn_2._hiband_scale_name = "ffn_2"
        if use_htg_norm2 and hasattr(phase, "ffn_0_htg_group_bias"):
            ffn0_group_bias = phase.ffn_0_htg_group_bias.tensor
        y = self._mm_apply_with_group_bias(
            phase.ffn_0,
            norm2_out,
            ffn0_group_bias,
            group_idx if use_htg_norm2 else 0,
        )
        if self.clean_cuda_cache:
            del norm2_out, x
            torch_device_module.empty_cache()
        y = torch.nn.functional.gelu(y, approximate="tanh")
        if self.clean_cuda_cache:
            torch_device_module.empty_cache()
        y = self._mm_apply_with_group_bias(
            phase.ffn_2,
            y,
            None,
            group_idx if use_htg_norm2 else 0,
        )

        return y

    def post_process(self, x, y, c_gate_msa, pre_infer_out=None):
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype) + y.to(self.sensitive_layer_dtype) * c_gate_msa.squeeze()
        else:
            x.add_(y * c_gate_msa.squeeze())

        if self.clean_cuda_cache:
            del y, c_gate_msa
            torch_device_module.empty_cache()
        return x
