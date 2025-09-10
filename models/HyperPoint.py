import torch
import torch.nn as nn
import numpy as np
from .build import MODELS
from utils.logger import *
import random
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from models.transformer import Group, Encoder, TransformerEncoder, TransformerDecoder
from models.CrossModal import TextTransformer as TextEncoder
from models.CrossModal import VisionTransformer as ImageEncoder
from timm.models.layers import trunc_normal_
import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix
import torch.nn.functional as F

def contrastive_loss(x0, x1, tau, hyp_c):
    # x0 and x1 - positive pair
    # tau - temperature
    # hyp_c - hyperbolic curvature, "0" enables sphere mode

    if hyp_c == 0:
        dist_f = lambda x, y: x @ y.t()
    else:
        dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)  #

    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    logits00 = dist_f(x0, x0) / tau - eye_mask
    logits01 = dist_f(x0, x1) / tau
    logits = torch.cat([logits01, logits00], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    loss = F.cross_entropy(logits, target)
    stats = {
        "logits/min": logits01.min().item(),
        "logits/mean": logits01.mean().item(),
        "logits/max": logits01.max().item(),
        "logits/acc": (logits01.argmax(-1) == target).float().mean().item(),
    }
    return loss, stats


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Transformer')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.group_size = config.group_size
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        self.downrate = config.transformer_config.downrate
        self.gcn_k = config.transformer_config.gcn_k
        self.ad_layers = config.transformer_config.ad_layers
        self.mask_type = config.transformer_config.mask_type
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.img_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.text_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.img_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.text_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.to_poincare_point = hypnn.ToPoincare(c=0.1, ball_dim=self.trans_dim, riemannian=False, clip_r=2.3)
        self.to_poincare_img = hypnn.ToPoincare(c=0.1, ball_dim=self.trans_dim, riemannian=False, clip_r=2.3)
        self.to_poincare_text = hypnn.ToPoincare(c=0.1, ball_dim=self.trans_dim, riemannian=False, clip_r=2.3)

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.img_token, std=.02)
        trunc_normal_(self.text_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.img_pos, std=.02)
        trunc_normal_(self.text_pos, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, points, neighborhood, center, noaug=False):

        bool_masked_pos_block = self._mask_center_block(center)
        bool_masked_pos_rand = self._mask_center_rand(center)


        group_input_tokens = self.encoder(neighborhood)  # B G C

        cls_token = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        batch_size, seq_len, C = group_input_tokens.size()
        x_vis_block = group_input_tokens[~bool_masked_pos_block].reshape(batch_size, -1, C)
        x_vis_rand = group_input_tokens[~bool_masked_pos_rand].reshape(batch_size, -1, C)
        # add pos embedding
        masked_center_block = center[~bool_masked_pos_block].reshape(batch_size, -1, 3)
        masked_center_rand = center[~bool_masked_pos_rand].reshape(batch_size, -1, 3)
        pos_block = self.pos_embed(masked_center_block)
        pos_rand = self.pos_embed(masked_center_rand)

        if eval:
            x_vis = x_vis_block
            pos = pos_block
        else:
            x_vis = torch.cat([x_vis_block, x_vis_rand], dim=0)
            pos = torch.cat([pos_block, pos_rand], dim=0)
        img_token = self.img_token.expand(group_input_tokens.size(0), -1, -1)
        img_pos = self.img_pos.expand(group_input_tokens.size(0), -1, -1)
        text_token = self.text_token.expand(group_input_tokens.size(0), -1, -1)
        text_pos = self.text_pos.expand(group_input_tokens.size(0), -1, -1)

        x = torch.cat((cls_token, img_token, text_token, x_vis), dim=1)
        pos = torch.cat((cls_pos, img_pos, text_pos, pos), dim=1)

        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)

        point_feature = self.to_poincare_point(x[:, 0].unsqueeze(1)).squeeze(1)  
        img_feature = self.to_poincare_img(x[:, 1].unsqueeze(1)).squeeze(1) 
        text_feature = self.to_poincare_text(x[:, 2].unsqueeze(1)).squeeze(1)  

        # return point_feature, img_feature, text_feature, x[:, 3:],
        return x[:, 0], x[:, 1], x[:, 2], x[:, 3:], bool_masked_pos_rand

class ContrastiveHead(nn.Module):

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, similarity, select):
        B = similarity.size(0)
        losses = torch.zeros(B).to(similarity.device)
        for i in range(B):
            pos = torch.masked_select(similarity[i], select[i] == 1)
            neg = torch.masked_select(similarity[i], select[i] == 0)
            pos = torch.mean(pos, dim=0, keepdim=True)
            logits = torch.cat((pos, neg)).reshape(1, -1)
            label = torch.zeros(1, dtype=torch.long).to(similarity.device)
            losses[i] = self.criterion(logits/self.temperature, label)
        loss = losses.mean()
        return loss


@MODELS.register_module()
class HyperPoint(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[HyperPoint] ', logger='HyperPoint')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[HyperPoint] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='HyperPoint')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

        # 添加双曲投影层
        self.to_poincare_point = hypnn.ToPoincare(c=0.1, ball_dim=self.trans_dim, riemannian=False, clip_r=2.3)
        self.to_poincare_img = hypnn.ToPoincare(c=0.1, ball_dim=self.trans_dim, riemannian=False, clip_r=2.3)
        self.to_poincare_text = hypnn.ToPoincare(c=0.1, ball_dim=self.trans_dim, riemannian=False, clip_r=2.3)

        # cross model contrastive
        self.csc_loss = torch.nn.SmoothL1Loss()
        self.csc_img = True if config.img_encoder else False
        self.csc_text = True if config.text_encoder else False

        if self.csc_img:
            self.img_encoder = ImageEncoder(config)
            for p in self.img_encoder.parameters():
                p.requires_grad = False
            self.img_proj = hypnn.HypLinear(self.trans_dim, self.img_encoder.output_dim,c=0.1)
            self.img_proj.apply(self._init_weights)

        if self.csc_text:
            self.text_encoder = TextEncoder(config)
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_proj = hypnn.HypLinear(self.trans_dim, self.text_encoder.embed_dim,c=0.1)

            self.text_proj.apply(self._init_weights)

        # single modal contrastive
        self.smc = config.self_contrastive
        if self.smc:
            self.cls_proj = nn.Sequential(
                nn.Linear(self.trans_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128)
            )
            self.cls_proj.apply(self._init_weights)
            self.contrastive_head = ContrastiveHead(temperature=0.1)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.02, 0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pts, img, text, **kwargs):

        losses = {}
        neighborhood, center = self.group_divider(pts)
        cls_token, img_token, text_token, x_vis, mask= self.MAE_encoder(pts, neighborhood, center)

        cls_token=self.to_poincare_point(cls_token)
        img_token=self.to_poincare_img(img_token)
        text_token=self.to_poincare_text(text_token)

        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        losses['mdm'] = self.loss_func(rebuild_points, gt_points)

        if self.csc_img:
            img_feature = self.img_encoder(img)
            img_token = self.img_proj(img_token)
            loss, stats = contrastive_loss(img_feature, img_token,tau=0.1,hyp_c=0.1)
            losses['csc_img']=loss.mean()

        if self.csc_text:
            text_feature = self.text_encoder(text)
            text_token = self.text_proj(text_token)
            loss, stats=contrastive_loss(text_feature, text_token,tau=0.1,hyp_c=0.1)
            losses['csc_text'] = loss.mean()

        if self.smc:
            cls_proj = self.cls_proj(cls_token)
            cls_proj = nn.functional.normalize(cls_proj, dim=1)
            similarity = torch.matmul(cls_proj, cls_proj.permute(1, 0))

            select = torch.zeros([B, B], dtype=torch.uint8).to(similarity.device)
            for i in range(B):
                for j in range(B):
                    if text[i] == text[j]:
                        select[i, j] = 1
            loss, stats=contrastive_loss(similarity, select,tau=0.1,hyp_c=0.1)
            losses['smc'] = loss.mean()
        loss = sum(losses.values())
        return loss
