import torch
import torch.nn as nn
import torchvision.models as torch_models
from collections import OrderedDict


class ManCoTrainNet(nn.Module):
    def __init__(self, classifier, num_voxels, num_classes=50):
        """
        :param classifier:
        :param num_voxels:
        """

        super(ManCoTrainNet, self).__init__()
        self.shared_layer = nn.Sequential(OrderedDict([
            ('conv1', classifier.conv1),
            ('bn1', classifier.bn1),
            ('relu', classifier.relu),
            ('maxpool', classifier.maxpool),
            ('layer1', classifier.layer1),
            ('layer2', classifier.layer2),
            ('layer3', classifier.layer3),
            ('layer4', classifier.layer4),
            ('avgpool', classifier.avgpool)
        ]))

        self.neural_head = nn.Sequential(OrderedDict([
            ("neural_flatten", nn.Flatten()),
            ("neural_fc", nn.Linear(512, num_voxels)),
        ]))

        # self.classification_head = classifier.fc
        self.classification_head = nn.Linear(512 * 1, num_classes)

        self.num_voxels = num_voxels

    def _classification_head(self, x):
        classification_out = torch.clone(x)
        classification_out = torch.flatten(classification_out, 1)
        classification_final = self.classification_head(classification_out)
        return classification_final

    def forward(self, x): # , idx2select): # , classification_only=False):
        shared_out = self.shared_layer(x)

        classification_final = self._classification_head(shared_out)
        neural_final = self.neural_head(shared_out)

        return neural_final, classification_final  # , neural_man_orig_space, neural_man_decorr


class AttackManNet(nn.Module):
    def __init__(self, man_net):
        super(AttackManNet, self).__init__()
        self.module = man_net

    def forward(self, x):
        _, clf_out = self.module(x) # , idx2select=None, classification_only=True)

        return clf_out


class ManLoss(nn.Module):
    def __init__(self, man_stats, NNR_dim, NNR_rad, safe_print, gpu, abl=None, abl_num_dim=None, decorr=False):
        super(ManLoss, self).__init__()
        # self.alphas = alphas
        self.clf_CE = nn.CrossEntropyLoss()  # classification
        self.max_op = nn.ReLU()  # for radius
        # self.criterion2 = nn.MSELoss()
        
        if abl == "dim":
            basis1 = man_stats["basis"][0].clone()
            basis1 = basis1.unsqueeze(0).repeat(man_stats["basis"].shape[0], 1, 1)
            self.register_buffer("basis", basis1)

            safe_print(f"!!!WARNING: Using the first basis for all categories!!!")
            safe_print(f"Sanity check: \n"
                       f"\t1stbasis: {basis1[0, :3, :3]}\n"
                       f"\t2ndbasis: {basis1[1, :3, :3]}\n"
                       f"\tbut the actual 2nd basis: {man_stats['basis'][1, :3, :3]}\n")
        
        elif abl == "dim-rand":
            safe_print(f"!!!WARNING: Using random basis for all categories!!! abl_num_dim: {abl_num_dim}")

            rand_U = torch.randn(man_stats["basis"].shape[-1], abl_num_dim)
            print(f"gpu: {gpu}, rand_U ({rand_U.size()}): {rand_U[0, :3]}", flush=True)

            U, S, Vh = torch.linalg.svd(rand_U, full_matrices=False)
            basis1_r = U @ U.T

            basis1_r = basis1_r.unsqueeze(0).repeat(man_stats["basis"].shape[0], 1, 1)
            self.register_buffer("basis", basis1_r)
            safe_print(f"Sanity check: \n"
                       f"\t1stbasis: {basis1_r[0, :3, :3]}\n"
                       f"\tbut the actual 1st basis: {man_stats['basis'][0, :3, :3]}\n")

        else:
            self.register_buffer("basis", man_stats["basis"])

        if abl == "rad":
            center1 = man_stats["center"][0].clone()
            center1 = center1.unsqueeze(0).repeat(man_stats["center"].shape[0], 1)
            self.register_buffer("center", center1)

            rad1 = man_stats["rad"][0].clone()
            rad1 = rad1.repeat(man_stats["rad"].shape[0])
            self.register_buffer("rad", rad1)

            safe_print(f"!!!WARNING: Using the first center/radius for all categories!!!")
            safe_print(f"Sanity check: \n"
                       f"\t1stcenter: {center1[0, :3]}\n"
                       f"\t2ndcenter: {center1[1, :3]}\n"
                       f"\tbut the actual 2nd center: {man_stats['center'][1, :3]}\n\n"
                       f"\t1strad: {rad1[0]}\n"
                       f"\t2ndrad: {rad1[1]}\n"
                       f"\tbut the actual 2nd rad: {man_stats['rad'][1]}\n")
        else:
            self.register_buffer("center", man_stats["center"])
            self.register_buffer("rad", man_stats["rad"])
        
        # NNR
        self.NNR_dim: bool = NNR_dim
        self.NNR_rad: float = NNR_rad

        self.decorr = decorr
        # Decorrelated space stats (force hard radius and dimension
        if self.decorr:
            raise NotImplementedError("Decorrelation not implemented yet")

    def _calc_per_category(self, uniq_cats, cat_idx, data, device):
        """
        idx2select: tensor of category indices for each sample in the current batch
        data: tensor of data to be aggregated per category, should be the same length as idx2select
        """
        cat_sums = torch.zeros(uniq_cats.shape[0], device=device)
        cat_cnts = torch.zeros_like(cat_sums)  # can match device type

        cat_sums.scatter_add_(0, cat_idx, data) # summing up the data for each category
        cat_cnts.scatter_add_(0, cat_idx, torch.ones_like(data)) # counting the number of samples for each category
        return cat_sums, cat_cnts

    def _calc_batch_categor_means(self, uniq_cats, cat_idx, neural_outs, device):
        #### calc category-wise RMS on data.
        category_sums = torch.zeros(uniq_cats.size()[0], neural_outs.size()[1], device=device)
        category_sums.scatter_add_(0, cat_idx.unsqueeze(-1).expand_as(neural_outs), neural_outs)  # cat_idx became 2D: [[0, 0, ..., 0], [1, 1, ..., 1], ...]

        category_cnts = torch.zeros(uniq_cats.size()[0], 1, device=device)
        category_cnts.scatter_add_(0, cat_idx.unsqueeze(-1), torch.ones_like(neural_outs))
        category_means = category_sums / category_cnts

        return category_means
    

    def _neural_rad_loss(self, gt_cls_lb, uniq_cats, cat_idx, neural_outs, device):
        #2. orig space man center-radius loss, need to follow the RMS in calc_radius
        centers = torch.index_select(self.center, 0, gt_cls_lb)
        #### centering neural outputs using the orig manifold centers!!!
        neural_outs_Y = neural_outs - centers 
        rms = torch.sum(neural_outs_Y**2, dim=1)
        #### calculate the rms for each category separately
        rms_sum_per_cat, rms_cnt_per_cat = self._calc_per_category(uniq_cats, cat_idx, rms, device)
        #### mean for each category and take the square root: RMS: R_M = np.sqrt(np.mean(ds_sq_sum))
        rms_per_cat = torch.sqrt(rms_sum_per_cat / rms_cnt_per_cat)
        
        rads4current_batch = torch.index_select(self.rad, 0, uniq_cats)
        loss2 = self.max_op(rms_per_cat - rads4current_batch).mean()

        return loss2


    def _NNR_rad_loss(self, neural_outs, curr_batch_category_means, uniq_cats, cat_idx, device):
        """
        Diff from neural man: the center needs to be estimated from the data itself, so measuring the spread of the current minibatch.
        """
        #TODO: enable using the center from the original manifold

        neural_outs_centered = neural_outs - curr_batch_category_means[cat_idx]  # centering
        neural_outs_sq_sum = torch.sum(neural_outs_centered**2, dim=1)
        
        #### calculate the rms for each category separately
        rms_sum_per_cat, rms_cnt_per_cat = self._calc_per_category(uniq_cats, cat_idx, neural_outs_sq_sum, device)
        #### mean for each category and take the square root: RMS: R_M = np.sqrt(np.mean(ds_sq_sum))
        rms_per_cat = torch.sqrt(rms_sum_per_cat / rms_cnt_per_cat)
        loss2 = self.max_op(rms_per_cat - self.NNR_rad).mean()
        return loss2

    def _neural_dim_loss(self, gt_cls_lb, uniq_cats, cat_idx, neural_outs, device):
        #3.  orig space man dimension loss
        UUT = torch.index_select(self.basis, 0, gt_cls_lb)  # batch * 1084 * 1084
        
        #### centering neural outputs using its own, actual meaning centering!!!
        # centers = torch.index_select(self.center, 0, gt_cls_lb)
        # neural_outs_centered = neural_outs - centers
        neural_outs_centered = neural_outs - torch.mean(neural_outs, dim=0) 
        neural_outs_centered_expanded = neural_outs_centered.unsqueeze(2)  # batch * voxels * 1
        neural_recon = torch.bmm(UUT, neural_outs_centered_expanded).squeeze(2)
        recon_err = torch.linalg.norm(neural_outs_centered - neural_recon, dim=1)
        vaf = 1 - recon_err / torch.linalg.norm(neural_outs_centered, dim=1)
        #### calculate the vaf for each category separately
        vaf_sum_per_cat, vaf_cnt_per_cat = self._calc_per_category(uniq_cats, cat_idx, vaf, device)
        #### mean for each category
        vaf_per_cat = vaf_sum_per_cat / vaf_cnt_per_cat

        loss3 = vaf_per_cat.mean()  # average across categories
        return loss3
    
    def _NNR_dim_loss(self, neural_outs, curr_batch_category_means, uniq_cats, cat_idx, device):
        neural_outs_centered = neural_outs - curr_batch_category_means[cat_idx] # centering
        
        category_nuc_norms = torch.zeros(uniq_cats.size()[0], device=device)
        for i, cat in enumerate(uniq_cats):
            mask = cat_idx == cat
            if mask.sum() < 2:
                continue
            category_nuc_norms[i] = torch.norm(neural_outs_centered[mask], p='nuc')

        loss3 = category_nuc_norms.mean()
        return loss3

    def forward(self, outputs, targets, neural_outs, device):

        gt_cls_lb = targets
        uniq_cats, cat_idx = gt_cls_lb.unique(return_inverse=True) # uniq_cats: sorted unique categories, cat_idx: indices of each category in the original tensor at each element
        
        #1. classification
        loss1 = self.clf_CE(outputs, targets)
        
        curr_batch_category_means = None
        #2. radius
        if self.NNR_rad is not None:
            curr_batch_category_means = self._calc_batch_categor_means(uniq_cats, cat_idx, neural_outs, device)
            loss2 = self._NNR_rad_loss(neural_outs, curr_batch_category_means, uniq_cats, cat_idx, device)
        else:
            loss2 = self._neural_rad_loss(gt_cls_lb, uniq_cats, cat_idx, neural_outs, device)
        
        #3. dimension
        if self.NNR_dim:
            if curr_batch_category_means is None:
                curr_batch_category_means = self._calc_batch_categor_means(uniq_cats, cat_idx, neural_outs, device)
            loss3 = self._NNR_dim_loss(neural_outs, curr_batch_category_means, uniq_cats, cat_idx, device)
        else:
            loss3 = self._neural_dim_loss(gt_cls_lb, uniq_cats, cat_idx, neural_outs, device)

        return loss1, loss2, loss3  # , loss4, loss5
    


def instantiate_ROI_man_model(model_f, classifier_arch:str, device, attack_wrapper=True):
 
    checkpoint = torch.load(model_f, map_location=device)

    num_voxels = checkpoint["state_dict"]['neural_head.neural_fc.weight'].size()[0]
    num_classes = checkpoint["state_dict"]['classification_head.weight'].size()[0]
    classifier = torch_models.__dict__[classifier_arch]()

    # instantiate a model
    man_model = ManCoTrainNet(classifier, num_voxels, num_classes=num_classes)
    man_model.load_state_dict(checkpoint['state_dict'])

    if attack_wrapper:
        man_model = AttackManNet(man_model)

    return man_model


