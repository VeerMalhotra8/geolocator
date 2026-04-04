"""
Phase 2: GeoLocator — Full geolocation classification model.

Architecture (from GEOGUESSR_ARCHITECTURE.md):
  - Backbone: StreetCLIP (CLIP ViT-L/14 @ 336px) with DoRA adapters
  - Geo head: Linear(768→2048)→GELU→Dropout→Linear(2048→num_cells)
  - Auxiliary heads: scene(6), climate(5), driving(3), region(16)
  - Loads DoRA weights from Phase 1.5 contrastive pretraining checkpoint

Usage:
    model = GeoLocator(num_cells=8970)
    model.load_contrastive_checkpoint("GEOPROJECT/checkpoints/contrastive_best.pt")
    outputs = model(images)  # dict: geo, scene, climate, driving, region, embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel


class GeoLocator(nn.Module):
    """Full geolocation model: StreetCLIP + DoRA + classification heads."""

    def __init__(
        self,
        num_cells: int,
        num_scene: int = 6,
        num_climate: int = 5,
        num_driving: int = 3,
        num_region: int = 16,  # 15 UN regions + 1 unknown for unmapped country codes
        dora_rank: int = 16,
        dora_alpha: int = 32,
        dora_layers: tuple = (16, 17, 18, 19, 20, 21, 22, 23),
        unfreeze_layers: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_cells = num_cells
        self.unfreeze_layers = unfreeze_layers

        # Load StreetCLIP (CLIP ViT-L/14 @ 336px, geo-adapted)
        clip = CLIPModel.from_pretrained("geolocal/StreetCLIP")
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection  # 1024 → 768 (frozen, shared with Phase 1.5)

        # ViT-L/14 has 24 encoder layers (0-23)
        num_encoder_layers = len(self.vision_model.encoder.layers)
        if not (0 <= unfreeze_layers <= num_encoder_layers):
            raise ValueError(
                f"unfreeze_layers must be 0-{num_encoder_layers}, got {unfreeze_layers}"
            )

        # Freeze entire backbone first
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.visual_projection.parameters():
            param.requires_grad = False

        # Determine DoRA layers, excluding any that will be fully unfrozen
        effective_dora_layers = dora_layers
        unfreeze_start = num_encoder_layers
        if unfreeze_layers > 0:
            unfreeze_start = num_encoder_layers - unfreeze_layers
            effective_dora_layers = tuple(l for l in dora_layers if l < unfreeze_start)
            print(f"  Partial unfreeze: layers {unfreeze_start}-{num_encoder_layers-1} fully trainable")
            print(f"  DoRA on layers: {effective_dora_layers if effective_dora_layers else 'none'}")
            if not effective_dora_layers:
                print(f"  WARNING: All DoRA layers are fully unfrozen. "
                      f"Contrastive checkpoint DoRA weights will NOT be loaded.")
        self._has_dora = bool(effective_dora_layers)

        # Apply DoRA first (PEFT re-freezes everything, so unfreeze AFTER)
        if effective_dora_layers:
            dora_config = LoraConfig(
                r=dora_rank,
                lora_alpha=dora_alpha,
                target_modules=["q_proj", "v_proj"],
                use_dora=True,
                layers_to_transform=list(effective_dora_layers),
            )
            self.vision_model = get_peft_model(self.vision_model, dora_config)

        # Now unfreeze last N encoder layers (after PEFT wrapping)
        if unfreeze_layers > 0:
            # PEFT wraps the model: vision_model.base_model.model.encoder.layers
            if self._has_dora:
                encoder_layers = self.vision_model.base_model.model.encoder.layers
            else:
                encoder_layers = self.vision_model.encoder.layers
            for i in range(unfreeze_start, num_encoder_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True

        # Embedding dimension from CLIP ViT-L/14
        embed_dim = 768

        # Main geo classification head
        self.geo_head = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, num_cells),
        )

        # Auxiliary heads (multi-task regularization + explainability)
        self.scene_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, num_scene)
        )
        self.climate_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, num_climate)
        )
        self.driving_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, num_driving)
        )
        self.region_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, num_region)
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: (B, 3, 336, 336) tensor, CLIP-normalized

        Returns:
            dict with keys:
                geo: (B, num_cells) logits
                scene: (B, 6) logits
                climate: (B, 5) logits
                driving: (B, 3) logits
                region: (B, 16) logits
                embedding: (B, 768) L2-normalized embedding
        """
        # Vision encoder → pooler output (post-LayerNorm CLS token)
        vision_out = self.vision_model(pixel_values=images)
        pooled = vision_out.pooler_output  # (B, 1024)

        # Project to shared embedding space
        embedding = self.visual_projection(pooled)  # (B, 768)
        embedding = F.normalize(embedding, dim=-1)

        return {
            "geo": self.geo_head(embedding),
            "scene": self.scene_head(embedding),
            "climate": self.climate_head(embedding),
            "driving": self.driving_head(embedding),
            "region": self.region_head(embedding),
            "embedding": embedding,
        }

    def load_contrastive_checkpoint(self, path: str):
        """Load DoRA adapter weights from Phase 1.5 contrastive pretraining.

        Only loads the DoRA/LoRA parameters into the vision_model,
        leaving classification heads randomly initialized.
        """
        if not self._has_dora:
            print(f"Skipping contrastive checkpoint load — no DoRA adapters in model "
                  f"(all DoRA layers fully unfrozen with unfreeze_layers={self.unfreeze_layers})")
            return

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        dora_state = ckpt["dora_state_dict"]

        # Phase 1.5 saved keys like "vision_model.base_model.model.encoder..."
        # Our model has the same structure, so direct load works
        missing, unexpected = self.load_state_dict(dora_state, strict=False)

        # All classification heads should be missing (expected)
        loaded = len(dora_state) - len(unexpected)
        print(f"Loaded {loaded} DoRA parameters from {path}")
        print(f"  Epoch: {ckpt['epoch']+1}, Contrastive loss: {ckpt['loss']:.4f}")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys (may be from different model structure)")

    def get_parameter_groups(
        self,
        lr_dora: float = 1e-4,
        lr_heads: float = 5e-4,
        lr_backbone: float = 2e-5,
    ):
        """Get parameter groups with differential learning rates.

        DoRA adapters: lower LR (fine-tuning pretrained representations)
        Classification heads: higher LR (training from scratch)
        Unfrozen backbone layers: lowest LR (careful fine-tuning of pretrained weights)
        """
        dora_params = []
        head_params = []
        backbone_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name or "dora_" in name or "magnitude" in name:
                dora_params.append(param)
            elif ("vision_model" in name or "visual_projection" in name
                  or "base_model" in name or "encoder.layers" in name):
                # Unfrozen backbone layers (only present when unfreeze_layers > 0)
                # PEFT wraps names as base_model.model.encoder.layers.*
                backbone_params.append(param)
            else:
                head_params.append(param)

        groups = []
        if dora_params:
            groups.append({"params": dora_params, "lr": lr_dora})
        if head_params:
            groups.append({"params": head_params, "lr": lr_heads})
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})

        return groups

    def trainable_summary(self) -> str:
        """Print trainable parameter summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dora = sum(
            p.numel() for n, p in self.named_parameters()
            if p.requires_grad and ("lora_" in n or "dora_" in n or "magnitude" in n)
        )
        backbone = sum(
            p.numel() for n, p in self.named_parameters()
            if p.requires_grad
            and ("vision_model" in n or "visual_projection" in n
                 or "base_model" in n or "encoder.layers" in n)
            and "lora_" not in n and "dora_" not in n and "magnitude" not in n
        )
        heads = trainable - dora - backbone
        lines = [
            f"Parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.2f}%)",
            f"  DoRA adapters: {dora:,}",
        ]
        if backbone > 0:
            lines.append(f"  Unfrozen backbone: {backbone:,}")
        lines.append(f"  Classification heads: {heads:,}")
        return "\n".join(lines)
