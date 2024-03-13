import torch
import torch.nn as nn
import torch.jit

from WSTTA.utils import create_pseudo_labels

class WSTTA(nn.Module):
    def __init__(self, args, model, optimizer):
        super().__init__();
        self.model = model;
        self.mom_lb = args.mom_lb; # lower bound of momentum
        self.mom = args.mom_init; # momentum to decay
        self.omega = args.omega; # decay factor
        self.alpha = args.alpha;
        self.psd_thr = args.psd_thr; # prob threshold to create pseudo labels
        self.optimizer = optimizer;
        

    def forward_only(self, x):
        # forward
        self.model.eval();
        with torch.no_grad(): # no_grad to fix the inheritance of nn.Module
            preds = self.model(x);
        return preds
    
    def forward_then_adapt(self, x):
        preds = self.forward_only(x);
        
         # decay momemtum
        self.mom = (self.mom * self.omega);
        self._unfreeze_net();
        
        # adapt
        loss = 0;
        if len(preds[0]['instances']) == 0:
            losses = self.model.adapt(x);
        else:
            x_with_psd = create_pseudo_labels(inputs=x, preds=preds, psd_thr=self.psd_thr);
            if len(x_with_psd[0]['instances']) == 0:
                losses = self.model.adapt(x_with_psd);
            else:
                losses = self.model.adapt(x_with_psd);
                loss = losses['loss_cls'] + losses['loss_rpn_cls'];            
        
        loss += (self.alpha * losses['loss_img']);
        
        if hasattr(loss, "backward"):
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        self.model.eval();
        
        return preds;
    
    def _unfreeze_net(self):
        self._unfreeze_backbone();
        self._unfreeze_rpn();
        self._unfreeze_roi();
        
    def _unfreeze_backbone(self):
        for m in self.model.backbone.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train();
                m.momentum = self.mom + self.mom_lb;
    
    def _unfreeze_rpn(self):
        for m in self.model.proposal_generator.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train();
                m.momentum = self.mom + self.mom_lb;
                
    def _unfreeze_roi(self):
        for m in self.model.roi_heads.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train();
                m.momentum = self.mom + self.mom_lb;