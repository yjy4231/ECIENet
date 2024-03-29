import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet as focal_loss
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss

class Hierarchical_Task_Learning:
    def __init__(self,epoch0_loss,stat_epoch_nums=5):
        self.index2term = [*epoch0_loss.keys()]
        self.term2index = {term:self.index2term.index(term) for term in self.index2term}
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses=[]
        self.loss_graph = {'seg_loss':[],
                           'size2d_loss':[], 
                           'offset2d_loss':[],
                           'offset3d_loss':['size2d_loss','offset2d_loss'],
                           'size3d_loss':['size2d_loss','offset2d_loss'], 
                           'heading_loss':['size2d_loss','offset2d_loss'], 
                           'depth_loss':['size2d_loss','size3d_loss','offset2d_loss']}

    def compute_weight(self,current_loss,epoch):
        T=140
        #compute initial weights
        loss_weights = {}
        eval_loss_input = torch.cat([_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)
        for term in self.loss_graph:
            if len(self.loss_graph[term])==0:
                loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device)
            else:
                loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device) 
        #update losses list
        if len(self.past_losses)==self.stat_epoch_nums:
            past_loss = torch.cat(self.past_losses)
            mean_diff = (past_loss[:-2]-past_loss[2:]).mean(0)
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff
            c_weights = 1-(mean_diff/self.init_diff).relu().unsqueeze(0)
            
            time_value = min(((epoch-5)/(T-5)),1.0)
            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic])!=0:
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][self.term2index[pre_topic]]      
                    loss_weights[current_topic] = time_value**(1-control_weight)
                    if loss_weights[current_topic] != loss_weights[current_topic]:
                        for pre_topic in self.loss_graph[current_topic]:
                            print('NAN===============', time_value, control_weight, c_weights[0][self.term2index[pre_topic]], pre_topic, self.term2index[pre_topic])
            #pop first list
            self.past_losses.pop(0)
        self.past_losses.append(eval_loss_input)

        return loss_weights
    def update_e0(self,eval_loss):
        self.epoch0_loss = torch.cat([_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)

class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            if len(weights.shape) == len(loss.shape) - 1:
                weights = weights.unsqueeze(-1)
            assert len(weights.shape) == len(loss.shape)
            loss = loss * weights
        loss = torch.mean(loss)
        return loss

class DIDLoss(nn.Module):
    def __init__(self,epoch):
        super().__init__()
        self.stat = {}
        self.epoch = epoch
        self.smooth_l1_loss = WeightedSmoothL1Loss()

    def forward(self, preds, targets):

        if targets['mask_2d'].sum() == 0:
            bbox2d_loss = 0
            bbox3d_loss = 0
            self.stat['offset2d_loss'] = 0
            self.stat['size2d_loss'] = 0
            self.stat['depth_loss'] = 0
            self.stat['offset3d_loss'] = 0
            self.stat['size3d_loss'] = 0
            self.stat['heading_loss'] = 0
        else:
            bbox2d_loss = self.compute_bbox2d_loss(preds, targets)
            bbox3d_loss = self.compute_bbox3d_loss(preds, targets)

        seg_loss = self.compute_segmentation_loss(preds, targets)
        mean_loss = seg_loss + bbox2d_loss + bbox3d_loss
        return float(mean_loss), self.stat

    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss = focal_loss(input['heatmap'], target['heatmap'])
        self.stat['seg_loss'] = loss
        return loss

    def compute_bbox2d_loss(self, input, target):
        # compute size2d loss
        size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
        size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
        size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
        # compute offset2d loss
        offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
        offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
        offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')

        loss = offset2d_loss + size2d_loss   
        self.stat['offset2d_loss'] = offset2d_loss
        self.stat['size2d_loss'] = size2d_loss
        return loss


    def compute_bbox3d_loss(self, input, target, mask_type = 'mask_2d'):
        vis_depth = input['vis_depth'][input['train_tag']]
        att_depth = input['att_depth'][input['train_tag']]
        vis_depth_target = extract_target_from_tensor(target['vis_depth'], target[mask_type])
        att_offset_target = extract_target_from_tensor(target['att_depth'], target[mask_type])
        depth_mask_target = extract_target_from_tensor(target['depth_mask'], target[mask_type])
        ins_depth_target = extract_target_from_tensor(target['depth'], target[mask_type])

        vis_depth_uncer = input['vis_depth_uncer'][input['train_tag']]
        att_depth_uncer = input['att_depth_uncer'][input['train_tag']]
        vis_depth_loss = laplacian_aleatoric_uncertainty_loss(vis_depth[depth_mask_target],
                                                              vis_depth_target[depth_mask_target],
                                                              vis_depth_uncer[depth_mask_target])
        att_depth_loss = laplacian_aleatoric_uncertainty_loss(att_depth[depth_mask_target],
                                                              att_offset_target[depth_mask_target],
                                                              att_depth_uncer[depth_mask_target])

        ins_depth = input['ins_depth'][input['train_tag']]
        ins_depth_uncer = input['ins_depth_uncer'][input['train_tag']]
        ins_depth_loss = laplacian_aleatoric_uncertainty_loss(ins_depth.view(-1, 7*7),
                                                                ins_depth_target.repeat(1, 7*7),
                                                                ins_depth_uncer.view(-1, 7*7))
        #####################################################################################
        K = input['ins_depth'].shape[0]
        merge_prob = (-(0.5 * input['ins_depth_uncer']).exp()).exp()
        merge_depth = (torch.sum((input['ins_depth']*merge_prob).view(K,-1), dim=-1) /
                    (torch.sum(merge_prob.view(K,-1), dim=-1) + 1e-8))
        merge_depth = merge_depth.unsqueeze(1)
        merge_depth = merge_depth[input['train_tag']]
        ins_depth_target = extract_target_from_tensor(target['depth'], target[mask_type])
        theta = 0.35
        d_weight = (torch.abs(torch.abs(merge_depth - ins_depth_target) - theta) * -1).exp().clone().detach()
        ins_depth_loss1 = self.smooth_l1_loss(merge_depth, ins_depth_target, d_weight)
        #####################################################################################
        depth_loss = vis_depth_loss + att_depth_loss + ins_depth_loss + ins_depth_loss1

        # compute offset3d loss
        offset3d_input = input['offset_3d'][input['train_tag']]
        offset3d_target = extract_target_from_tensor(target['offset_3d'], target[mask_type])
        offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')


        # compute size3d loss
        size3d_input = input['size_3d'][input['train_tag']]
        size3d_target = extract_target_from_tensor(target['size_3d'], target[mask_type])
        size3d_loss = F.l1_loss(size3d_input, size3d_target, reduction='mean')
        ############################################################################################################
        size3d_loss2 = self.smooth_l1_loss(size3d_input[:,-1].view(-1,1), size3d_target[:,-1].view(-1,1), d_weight)
        size3d_loss = size3d_loss + size3d_loss2
        ############################################################################################################
        # compute heading loss
        heading_loss = compute_heading_loss(input['heading'][input['train_tag']] ,
                                            target[mask_type],  ## NOTE
                                            target['heading_bin'],
                                            target['heading_res'])

        loss = depth_loss + offset3d_loss + size3d_loss + heading_loss 

        if depth_loss != depth_loss:
            print('badNAN----------------depth_loss', depth_loss)
            print(vis_depth_loss, att_depth_loss, ins_depth_loss, depth_mask_target.sum())
        if offset3d_loss != offset3d_loss:
            print('badNAN----------------offset3d_loss', offset3d_loss)
        if size3d_loss != size3d_loss:
            print('badNAN----------------size3d_loss', size3d_loss)
        if heading_loss != heading_loss:
            print('badNAN----------------heading_loss', heading_loss)

        self.stat['depth_loss'] = depth_loss
        self.stat['offset3d_loss'] = offset3d_loss
        self.stat['size3d_loss'] = size3d_loss
        self.stat['heading_loss'] = heading_loss
        
        return loss


### ======================  auxiliary functions  =======================

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask]

#compute heading loss two stage style  

def compute_heading_loss(input, mask, target_cls, target_reg):
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    target_cls = target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')
    
    # regression loss
    input_reg = input[:, 12:24]
    target_reg = target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss    
