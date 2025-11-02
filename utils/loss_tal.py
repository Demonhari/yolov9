import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import xywh2xyxy
from utils.metrics import bbox_iou
from utils.tal.anchor_generator import dist2bbox, make_anchors, bbox2dist
from utils.tal.assigner import TaskAlignedAssigner
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(),
                                                       reduction="none") * weight).sum()
        return loss


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
        
        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        loss_iou = 1.0 - iou

        loss_iou *= bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum

        # dfl loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1),
                                     reduction="none").view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)


class ComputeLoss:
    # Compute losses
    def __init__(self, model, use_dfl=True):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.assigner = TaskAlignedAssigner(topk=int(os.getenv('YOLOM', 10)),
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = None 
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    def _infer_na_nbins(self, ch, nc):
        # Solve ch = na * (4*nbins + nc)
        # Try reasonable ranges: na in [1..9], nbins in [8..64]
        for na in range(1, 13):  # a bit generous
            rem = ch - na * nc
            if rem <= 0: 
                continue
            if rem % (4 * na) == 0:
                nbins = rem // (4 * na)
                if 8 <= nbins <= 64:
                    return na, nbins
        # Fallback: assume anchor-free (na=1) and try to back out nbins
        nbins = (ch - nc) // 4 if (ch - nc) % 4 == 0 else None
        if nbins is not None and 8 <= nbins <= 64:
            return 1, nbins
        raise RuntimeError(f"Cannot infer (na, nbins) from channels={ch}, nc={nc}.")

    def bbox_decode(self, anchor_points, pred_dist):
        b, a, c = pred_dist.shape  # (B, HW, 4*nbins)
        if self.use_dfl:
            nbins = c // 4
            # lazily (re)build proj if needed
            if (self.proj is None or
                self.proj.numel() != nbins or
                self.proj.device != pred_dist.device):
                self.proj = torch.arange(nbins, device=pred_dist.device).float()
            pred_dist = pred_dist.view(b, a, 4, nbins).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, p, targets, img=None, epoch=0):
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        # p can be (pred, aux) tuple; keep original behavior
        feats = p[1] if isinstance(p, tuple) else p

        # --- Normalize outputs and build pred_distri / pred_scores robustly ---
        # Layout 1: tuple-of-lists => ([pd_l3,pd_l4,pd_l5], [ps_l3,ps_l4,ps_l5])
        if (isinstance(feats, (list, tuple))
            and len(feats) == 2
            and isinstance(feats[0], (list, tuple))
            and isinstance(feats[1], (list, tuple))
            and torch.is_tensor(feats[0][0])
            and torch.is_tensor(feats[1][0])):

            pd_src_levels = list(feats[0])  # original tensors for anchors
            ps_src_levels = list(feats[1])
            b = pd_src_levels[0].shape[0]

            pd_list, ps_list = [], []
            na_per_level, hw_per_level, nbins_per_level = [], [], []

            for pd_lvl, ps_lvl in zip(pd_src_levels, ps_src_levels):
                # pd_lvl: (B, 4*nbins*na, H, W)
                # ps_lvl: (B, nc*na,      H, W)
                _, chd, h, w = pd_lvl.shape
                _, chc, _, _ = ps_lvl.shape

                na = chc // self.nc
                assert na > 0 and chc == na * self.nc, f"[L1] cls channels {chc} not divisible by nc={self.nc}"
                nbins = chd // (4 * na)
                assert 4 * nbins * na == chd, f"[L1] dist channels {chd} not divisible by 4*na={4*na}"

                # (B, HW*na, ...)
                dist = pd_lvl.view(b, na, 4*nbins, h*w).permute(0, 3, 1, 2).contiguous().view(b, h*w*na, 4*nbins)
                cls  = ps_lvl.view(b, na, self.nc, h*w).permute(0, 3, 1, 2).contiguous().view(b, h*w*na, self.nc)

                pd_list.append(dist)
                ps_list.append(cls)

                na_per_level.append(na)
                hw_per_level.append(h * w)
                nbins_per_level.append(nbins)

            pred_distri = torch.cat(pd_list, dim=1)  # (B, sum(HW*na), 4*nbins_levelwise)
            pred_scores = torch.cat(ps_list, dim=1)  # (B, sum(HW*na), nc)

            # Use the original dist tensors to derive H,W per level later
            feats_for_anchors = pd_src_levels
            first_feat = feats_for_anchors[0]

        # Layout 2: list-of-pairs per level => [(pd_l3, ps_l3), (pd_l4, ps_l4), (pd_l5, ps_l5)]
        elif (isinstance(feats, (list, tuple))
            and len(feats) > 0
            and isinstance(feats[0], (list, tuple))
            and torch.is_tensor(feats[0][0])):
            pd_list, ps_list = [], []
            na_per_level, hw_per_level, nbins_per_level = [], [], []

            b = feats[0][0].shape[0]
            for (pd_lvl, ps_lvl) in feats:
                # pd_lvl: (B, 4*nbins*na, H, W)
                # ps_lvl: (B, nc*na,      H, W)
                _, chd, h, w = pd_lvl.shape
                _, chc, _, _ = ps_lvl.shape

                na = chc // self.nc
                assert na > 0 and chc == na * self.nc, f"cls channels {chc} not divisible by nc={self.nc}"

                nbins = chd // (4 * na)
                assert nbins * 4 * na == chd, f"dist channels {chd} not divisible by 4*na={4*na}"

                # reshape to (B, HW*na, …)
                dist = pd_lvl.view(b, na, 4*nbins, h*w).permute(0, 3, 1, 2).contiguous().view(b, h*w*na, 4*nbins)
                cls  = ps_lvl.view(b, na, self.nc, h*w).permute(0, 3, 1, 2).contiguous().view(b, h*w*na, self.nc)

                pd_list.append(dist)
                ps_list.append(cls)

                na_per_level.append(na)
                hw_per_level.append(h * w)
                nbins_per_level.append(nbins)

            pred_distri = torch.cat(pd_list, dim=1)   # (B, sum(HW*na), 4*nbins_levelwise)
            pred_scores = torch.cat(ps_list, dim=1)   # (B, sum(HW*na), nc)

            # Use the original per-level dist tensors for anchor shapes
            feats_for_anchors = [x[0] for x in feats]
            first_feat = feats_for_anchors[0]


        # Layout 3: old style list of fused tensors (channels include dist+cls)
        else:
            # ----- Adaptive Case B: each level is a single tensor with anchors packed in channels -----
            b = feats[0].shape[0]
            pd_list, ps_list = []
            na_per_level, hw_per_level, nbins_per_level = [], [], []

            for xi in feats:
                # xi: (B, C, H, W)
                _, ch, h, w = xi.shape
                na, nbins = self._infer_na_nbins(ch, self.nc)   # <- parse anchors & bins
                c_dist = 4 * nbins * na
                c_cls  = self.nc * na
                assert c_dist + c_cls == ch, "channel split mismatch"

                # split channels
                dist = xi[:, :c_dist, :, :]   # (B, 4*nbins*na, H, W)
                cls  = xi[:, c_dist:, :, :]   # (B, nc*na,      H, W)

                # reshape to (B, HW*na, …)
                dist = dist.view(b, na, 4*nbins, h*w).permute(0, 3, 1, 2).contiguous().view(b, h*w*na, 4*nbins)
                cls  = cls.view(b, na, self.nc, h*w).permute(0, 3, 1, 2).contiguous().view(b, h*w*na, self.nc)

                pd_list.append(dist)
                ps_list.append(cls)

                na_per_level.append(na)
                hw_per_level.append(h * w)
                nbins_per_level.append(nbins)

            pred_distri = torch.cat(pd_list, dim=1)   # (B, sum(HW*na), 4*nbins_levelwise)  nbins may vary per level (often same)
            pred_scores = torch.cat(ps_list, dim=1)   # (B, sum(HW*na), nc)

            feats_for_anchors = feats                 # we still use the raw feature maps for H,W extraction
            first_feat = feats[0]


        # From here on, the shapes are unified
        dtype = pred_scores.dtype
        batch_size, grid_size = pred_scores.shape[:2]
        imgsz = torch.tensor(first_feat.shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # (h,w)
        anchor_points, stride_tensor = make_anchors(feats_for_anchors, self.stride, 0.5)
        
        B = pred_scores.shape[0]

        ap_slices, st_slices = [], []
        offset = 0

        # Helper to slice one level depending on dims
        def _slice_level(ap, st, ofs, hw):
            if ap.dim() == 2:
                # (A,2) and (A,1)
                ap_lvl = ap[ofs:ofs+hw, :]          # (HW, 2)
                st_lvl = st[ofs:ofs+hw, :]          # (HW, 1)
            elif ap.dim() == 3:
                # (B, A, 2) and (B, A, 1)
                ap_lvl = ap[:, ofs:ofs+hw, :]       # (B, HW, 2)
                st_lvl = st[:, ofs:ofs+hw, :]       # (B, HW, 1)
            else:
                raise RuntimeError(f"Unexpected anchor_points dim={ap.dim()}")
            return ap_lvl, st_lvl

        for lvl, feat in enumerate(feats_for_anchors):
            h, w = feat.shape[2], feat.shape[3]
            hw = h * w

            ap_lvl, st_lvl = _slice_level(anchor_points, stride_tensor, offset, hw)
            offset += hw

            na = na_per_level[lvl] if 'na_per_level' in locals() else 1  # Case A may not set it
            if ap_lvl.dim() == 2:
                # (HW,*) -> (HW*na,*)
                if na > 1:
                    ap_lvl = ap_lvl.repeat_interleave(na, dim=0)
                    st_lvl = st_lvl.repeat_interleave(na, dim=0)
                ap_slices.append(ap_lvl)  # (HW*na, 2)
                st_slices.append(st_lvl)  # (HW*na, 1)
            else:
                # (B, HW,*) -> (B, HW*na,*)
                if na > 1:
                    ap_lvl = ap_lvl.repeat(1, na, 1)
                    st_lvl = st_lvl.repeat(1, na, 1)
                ap_slices.append(ap_lvl)  # (B, HW*na, 2)
                st_slices.append(st_lvl)  # (B, HW*na, 1)

        # Concatenate and ensure (B, sum(HW*na), *)
        if ap_slices[0].dim() == 2:
            # stack to batch
            anchor_points = torch.cat(ap_slices, dim=0).unsqueeze(0).expand(B, -1, -1)
            stride_tensor = torch.cat(st_slices, dim=0).unsqueeze(0).expand(B, -1, -1)
        else:
            anchor_points = torch.cat(ap_slices, dim=1)
            stride_tensor = torch.cat(st_slices, dim=1)               # (B, sum(HW*na), 1)

        # targets
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Ensure BboxLoss.reg_max matches current distribution bins (nbins - 1)
        if self.use_dfl:
            # infer nbins from concatenated pred_distri (it’s levelwise-consistent within each row)
            nbins = pred_distri.shape[-1] // 4
            if self.proj is None or self.proj.numel() != nbins or self.proj.device != pred_distri.device:
                self.proj = torch.arange(nbins, device=pred_distri.device).float()
            if self.bbox_loss.reg_max != nbins - 1:
                self.bbox_loss.reg_max = nbins - 1


        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        ap_assign = anchor_points if anchor_points.dim() == 2 else anchor_points[0]  # (A, 2)
        st_assign = stride_tensor if stride_tensor.dim() == 2 else stride_tensor[0]  # (A, 1)

        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),                                  # (B, A, nc)
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),    # (B, A, 4) in pixels
            ap_assign * st_assign,                                           # (A, 2) in pixels
            gt_labels,                                                       # (B, N, 1)
            gt_bboxes,                                                       # (B, N, 4) xyxy in pixels
            mask_gt                                                          # (B, N, 1)
        )

        target_bboxes /= stride_tensor
        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.BCEcls(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # bbox loss
        if fg_mask.sum():
            loss[0], loss[2], iou = self.bbox_loss(pred_distri,
                                                   pred_bboxes,
                                                   anchor_points,
                                                   target_bboxes,
                                                   target_scores,
                                                   target_scores_sum,
                                                   fg_mask)

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
