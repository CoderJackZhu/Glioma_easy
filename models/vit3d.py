# -*- coding:utf-8 -*-
# @PROJECT_NAME :Glioma_easy
# @FileName     :vit3d.py
# @Time         :2023/7/21 22:21
# @Author       :Jack Zhu

import torch
import torch.nn as nn
import torchvision

class VisionTransformer_stage(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed_3d, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer_stage, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1  # num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # 默认参数
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # token/patch的个数

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # parameter构建可训练参数，第一个1是batch size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # 位置编码的大小和加入分类token之后的大小相同
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # 构建等差序列，dropout率是递增的
        #         self.blocks = nn.Sequential(*[
        #             Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                   drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #                   norm_layer=norm_layer, act_layer=act_layer)
        #             for i in range(depth)
        #         ])
        self.stage1 = nn.Sequential(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[0], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[1], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[2], norm_layer=norm_layer, act_layer=act_layer))

        self.stage2 = nn.Sequential(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[3], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[4], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[5], norm_layer=norm_layer, act_layer=act_layer))

        self.stage3 = nn.Sequential(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[6], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[7], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[8], norm_layer=norm_layer, act_layer=act_layer))

        self.stage4 = nn.Sequential(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[9], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[10], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[11], norm_layer=norm_layer, act_layer=act_layer))

        self.norm = norm_layer(embed_dim)

        self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # 把cls_token复制batch_size份
        x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]

        x = self.pos_drop(x + self.pos_embed)
        # x = self.blocks(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.norm(x)

        x = self.pre_logits(x[:, 0])
        x = self.head(x)  # 执行这里

        return x


class PatchEmbed_3d(nn.Module):
    """
    3D Volume to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=1, embed_dim=4096, norm_layer=None):
        # embed_dim = 16*16*16 = 4096 token在flatten之后的长度
        super().__init__()
        img_size = (img_size, img_size, img_size)
        patch_size = (patch_size, patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        # 一共有多少个token(patche)  (224/16)*(224/16)*(224/16) = 14*14*14 = 2744

        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, P = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 输入图片的大小必须是固定的

        # x 大小224*224*224 经过k=16,s=16,c=4096的卷积核之后大小为 14*14*14*4096
        # flatten: [B, C, H, W, P] -> [B, C, HWP]   [B, 4096, 14, 14, 14] -> [B, 4096, 2744]
        # 对于Transfoemer模块，要求输入的是token序列，即 [num_token,token_dim] = [2744,4096]
        # transpose: [B, C, HWP] -> [B, HWP, C]   [B, 4096, 2744] -> [B, 2744, 4096]

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


if __name__ == "__main__":
    num_classes = 4
    vit_base_patch16_224_3d = VisionTransformer_stage(img_size=128,
                                                      in_c=128,
                                                      patch_size=16,
                                                      embed_dim=16 * 16 * 16,
                                                      depth=12,
                                                      num_heads=4,
                                                      num_classes=num_classes,
                                                      embed_layer=PatchEmbed_3d)
    x = torch.randn(1, 4, 128, 128, 128)
    X = vit_base_patch16_224_3d(x)
    print(X.shape)
    # torch.Size([1, 3])

