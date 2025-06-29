from typing import List

import numpy as np
import torch

import backbones
import continual_lib


class Model(continual_lib.BaseContinualLearner):
    """LoRA adapters reinitialized at every task."""

    REQ_NON_AUG_INPUTS = False

    def __init__(
        self,
        args,
        backbone,
        head,
        loss,
        device,
        scale: float = 1,
        rank: int = 5,
        backbone_block_idcs: List[int] = [0, 1, 2, 3, 4, 5],
        head_block_idcs: List[int] = [0, 1, 2, 3, 4, 5],
        kv_only: bool = True,
        tune_logit_scale: bool = False,
        **kwargs,
    ):
        super().__init__(args, backbone, head, loss, device)

        self.scale = scale
        self.rank = rank
        self.backbone_block_idcs = backbone_block_idcs
        self.head_block_idcs = head_block_idcs
        self.kv_only = kv_only

        for _, w in self.backbone.named_parameters():
            w.requires_grad = False
        for n, w in self.head.named_parameters():
            if tune_logit_scale and "logit_scale" in n:
                w.requires_grad = True
            else:
                w.requires_grad = False

        self.stored_adapters = []
        self.adapter_dict = {}
        self.to_optimize = []

        self._attach_adapters()

    # ------------------------------------------------------------------
    def _attach_adapters(self):
        self.adapter_dict = {}
        self.to_optimize = []

        backbone_adapter_blocks = None
        if not self.freeze_features:
            backbone_adapter_blocks = backbones.backbone_separators[
                self.args.experiment.backbone.name
            ]
            if len(self.backbone_block_idcs):
                backbone_adapter_blocks = [
                    backbone_adapter_blocks[i] for i in self.backbone_block_idcs
                ]

        head_adapter_blocks = None
        if not self.freeze_head and self.args.experiment.backbone.head == "default":
            head_adapter_blocks = backbones.head_separators[
                self.args.experiment.backbone.name
            ]
            if len(self.head_block_idcs):
                head_adapter_blocks = [
                    head_adapter_blocks[i] for i in self.head_block_idcs
                ]

        blocks = [backbone_adapter_blocks, head_adapter_blocks]
        modules = [self.backbone, self.head]

        for module_blocks, module in zip(blocks, modules):
            if module_blocks is None:
                continue
            for layer, weight in module.module.named_parameters():
                mod = module.module
                if len(module_blocks) and not any(x + "." in layer for x in module_blocks):
                    continue
                if "bias" in layer:
                    continue
                mod_layer = ".".join(layer.split(".")[:-1])
                for name in mod_layer.split("."):
                    mod = mod._modules[name]

                if isinstance(mod, torch.nn.Conv2d) and "patch_embed" not in mod_layer:
                    self.adapter_dict[mod_layer] = LoRA_Conv2d(
                        mod,
                        self.rank,
                        self.scale,
                        *weight.shape[:3],
                        name=mod_layer,
                    )
                    mod.forward = self.adapter_dict[mod_layer].forward
                    self.to_optimize.extend(self.adapter_dict[mod_layer].to_optimize)

                if isinstance(mod, torch.nn.Linear) and "head" not in mod_layer and "fc" not in mod_layer:
                    self.adapter_dict[mod_layer] = LoRA_Linear(
                        mod,
                        self.rank,
                        self.scale,
                        mod.out_features,
                        mod.in_features,
                        name=mod_layer,
                    )
                    mod.forward = self.adapter_dict[mod_layer].forward
                    self.to_optimize.extend(self.adapter_dict[mod_layer].to_optimize)

                if (
                    isinstance(mod, torch.nn.Linear)
                    and "qkv" in mod_layer
                    and self.args.experiment.backbone.name in backbones.timm_models
                ):
                    self.adapter_dict[mod_layer] = LoRA_QKVlinear(
                        mod,
                        self.rank,
                        self.scale,
                        mod.in_features,
                        mod.in_features,
                        name=mod_layer,
                        kv_only=self.kv_only,
                    )
                    mod.forward = self.adapter_dict[mod_layer].forward
                    self.to_optimize.extend(self.adapter_dict[mod_layer].to_optimize)

                if (
                    mod_layer.endswith("attn")
                    and self.args.experiment.backbone.name in backbones.clip_models + backbones.openclip_models
                ):
                    self.adapter_dict[mod_layer] = LoRA_MHA(
                        mod,
                        self.rank,
                        self.scale,
                        name=mod_layer,
                        kv_only=self.kv_only,
                    )
                    mod.forward = self.adapter_dict[mod_layer].forward
                    self.to_optimize.extend(self.adapter_dict[mod_layer].to_optimize)

        if not self.freeze_head and head_adapter_blocks is None:
            self.to_optimize.append({"params": self.head.parameters()})

        if hasattr(self.head.module, "text_encoder") and hasattr(self.head.module.text_encoder, "logit_scale"):
            if any(p.requires_grad for p in self.head.module.text_encoder.logit_scale.parameters()):
                self.to_optimize.append({"params": self.head.module.text_encoder.logit_scale})

    def _detach_adapters(self):
        for adapter in self.adapter_dict.values():
            mod = adapter.base_module
            mod.forward = mod.__class__.forward
        self.adapter_dict = {}
        self.to_optimize = []

    # ------------------------------------------------------------------
    def prepare_for_training(self, experiment=None, **kwargs):
        super().prepare_for_training(experiment, **kwargs)
        self.attach_average_adapter()

    def end_task(self, experiment=None, **kwargs):
        if self.adapter_dict:
            self.stored_adapters.append({name: a.state_dict() for name, a in self.adapter_dict.items()})
        self._detach_adapters()
        super().end_task(experiment, **kwargs)

    # ------------------------------------------------------------------
    def observe(self, images, targets, **kwargs):
        self.opt.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = self.forward(images=images.cuda(), **kwargs)
            logit_scale = getattr(self.head.module.text_encoder, "logit_scale", 1.0)
            temp = 1.0 / logit_scale.exp()
            loss = self.loss(targets=targets.cuda(), temperature=temp, **outputs, **kwargs)
        self.gradient_update(loss)
        return loss.item()

    @property
    def checkpoint(self):
        return {"self": self.state_dict(), "stored_adapters": self.stored_adapters}

    def load_from_checkpoint(self, state_dict):
        self.stored_adapters = state_dict.get("stored_adapters", [])
        self.load_state_dict(state_dict["self"])

    # ------------------------------------------------------------------
    def _average_adapter_state(self):
        """Return averaged state dict over all stored adapters."""
        if not self.stored_adapters:
            return None
        avg_state = {}
        num = len(self.stored_adapters)
        for name in self.stored_adapters[0]:
            avg_state[name] = {}
            for param_name in self.stored_adapters[0][name]:
                vals = [ad[name][param_name] for ad in self.stored_adapters]
                stacked = torch.stack(vals, dim=0)
                avg_state[name][param_name] = stacked.mean(dim=0)
        return avg_state

    def attach_average_adapter(self):
        """Attach new adapters initialized with the average of stored adapters."""
        avg_state = self._average_adapter_state()
        self._attach_adapters()
        if avg_state:
            for name, sd in avg_state.items():
                if name in self.adapter_dict:
                    self.adapter_dict[name].load_state_dict(sd)
        return avg_state

    def max_logits_over_stored_adapters(self, images, **kwargs):
        """Return logits by selecting the maximum over all stored adapters."""
        if not self.stored_adapters:
            with torch.no_grad(), torch.cuda.amp.autocast():
                return self.forward(images=images.cuda(), **kwargs)["logits"]
        logits_list = []
        for adapter_state in self.stored_adapters:
            self._attach_adapters()
            for name, sd in adapter_state.items():
                if name in self.adapter_dict:
                    self.adapter_dict[name].load_state_dict(sd)
            with torch.no_grad(), torch.cuda.amp.autocast():
                logits = self.forward(images=images.cuda(), **kwargs)["logits"]
            logits_list.append(logits)
            self._detach_adapters()
        stacked = torch.stack(logits_list, dim=0)
        return stacked.max(dim=0).values


class LoRA_Conv2d(torch.nn.Module):
    def __init__(self, base_module, rank, scale, out_channels, in_channels, kernel_size, name=None):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.lora_A = torch.nn.Parameter(
            self.base_module.weight.new_zeros((self.rank * kernel_size, in_channels * kernel_size))
        )
        self.lora_B = torch.nn.Parameter(
            self.base_module.weight.new_zeros((out_channels * kernel_size, self.rank * kernel_size))
        )
        self.to_optimize = [{"params": self.lora_A}, {"params": self.lora_B}]
        self.scale = scale
        self.assigned_name = name
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(self.rank))
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, *input):
        device = input[0].device
        lora_A = self.lora_A.to(device)
        lora_B = self.lora_B.to(device)
        weight = self.scale * (lora_B @ lora_A).view(self.base_module.weight.shape)
        base_weight = self.base_module.weight.to(device)
        base_bias = self.base_module.bias.to(device) if self.base_module.bias is not None else None
        return torch.nn.functional.conv2d(
            *input,
            base_weight + weight,
            base_bias,
            self.base_module.stride,
            self.base_module.padding,
            self.base_module.dilation,
            self.base_module.groups,
        )


class LoRA_Linear(torch.nn.Module):
    def __init__(self, base_module, rank, scale, out_features, in_features, name=None):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.lora_A = torch.nn.Parameter(self.base_module.weight.new_zeros((self.rank, in_features)))
        self.lora_B = torch.nn.Parameter(self.base_module.weight.new_zeros((out_features, self.rank)))
        self.to_optimize = [{"params": self.lora_A}, {"params": self.lora_B}]
        self.scale = scale
        self.assigned_name = name
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(self.rank))
        torch.nn.init.zeros_(self.lora_B)

    def forward(self, *input, **kwargs):
        device = input[0].device
        lora_A = self.lora_A.to(device)
        lora_B = self.lora_B.to(device)
        weight = self.scale * (lora_B @ lora_A)
        base_weight = self.base_module.weight.to(device)
        base_bias = self.base_module.bias.to(device) if self.base_module.bias is not None else None
        return torch.nn.functional.linear(*input, base_weight + weight, base_bias)


class LoRA_QKVlinear(torch.nn.Module):
    def __init__(self, base_module, rank, scale, out_features, in_features, name=None, kv_only: bool = True):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.kv_only = kv_only
        mul = 2 if kv_only else 3
        self.lora_A = torch.nn.ParameterList(
            [torch.nn.Parameter(self.base_module.weight.new_zeros((self.rank, in_features))) for _ in range(mul)]
        )
        self.lora_B = torch.nn.ParameterList(
            [torch.nn.Parameter(self.base_module.weight.new_zeros((out_features, self.rank))) for _ in range(mul)]
        )
        self.to_optimize = [{"params": self.lora_A}, {"params": self.lora_B}]
        self.scale = scale
        self.assigned_name = name
        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(2):
            torch.nn.init.kaiming_uniform_(self.lora_A[i], a=np.sqrt(self.rank))
            torch.nn.init.zeros_(self.lora_B[i])

    def forward(self, *input, **kwargs):
        weight = torch.cat([self.scale * B @ A for A, B in zip(self.lora_A, self.lora_B)], dim=0)
        if self.kv_only:
            zeros = torch.zeros_like(self.base_module.weight)[: -weight.shape[0]]
            weight = torch.cat([zeros, weight], dim=0)
        return torch.nn.functional.linear(*input, self.base_module.weight + weight, self.base_module.bias)


class LoRA_MHA(torch.nn.Module):
    def __init__(self, base_module, rank, scale, name: str = None, kv_only: bool = False):
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.kv_only = kv_only
        mul = 2 if self.kv_only else 3
        self.mul = mul
        _, in_features = self.base_module.in_proj_weight.size()
        self.lora_A = torch.nn.ParameterList(
            [torch.nn.Parameter(self.base_module.in_proj_weight.new_zeros((self.rank, in_features))) for _ in range(mul)]
        )
        self.lora_B = torch.nn.ParameterList(
            [torch.nn.Parameter(self.base_module.in_proj_weight.new_zeros((in_features, self.rank))) for _ in range(mul)]
        )
        self.to_optimize = [{"params": self.lora_A}, {"params": self.lora_B}]
        self.scale = scale
        self.assigned_name = name
        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(self.mul):
            torch.nn.init.kaiming_uniform_(self.lora_A[i], a=np.sqrt(self.rank))
            torch.nn.init.zeros_(self.lora_B[i])

    def forward(self, *input, **kwargs):
        device = input[0].device
        lora_A = [param.to(device) for param in self.lora_A]
        lora_B = [param.to(device) for param in self.lora_B]
        in_proj_weight = torch.cat([self.scale * B @ A for A, B in zip(lora_A, lora_B)], dim=0)
        if self.kv_only:
            zeros = torch.zeros_like(self.base_module.in_proj_weight)[: -in_proj_weight.shape[0]]
            in_proj_weight = torch.cat([zeros, in_proj_weight], dim=0)
        base_module_weights = self.base_module.in_proj_weight.to(device)
        base_module_bias = self.base_module.in_proj_bias.to(device)
        out_proj_weight = self.base_module.out_proj.weight.to(device)
        out_proj_bias = self.base_module.out_proj.bias.to(device)
        return torch.nn.functional.multi_head_attention_forward(
            *input,
            self.base_module.embed_dim,
            self.base_module.num_heads,
            base_module_weights + in_proj_weight,
            base_module_bias,
            self.base_module.bias_k,
            self.base_module.bias_v,
            self.base_module.add_zero_attn,
            self.base_module.dropout,
            out_proj_weight,
            out_proj_bias,
            training=self.training,
            **kwargs,
        )

