from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from torch.optim import AdamW
from nnunetv2.utilities.collate_outputs import collate_outputs
from torch import distributed as dist
from typing import Union, Tuple, List
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, save_json, maybe_mkdir_p
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.training.dataloading.utils import unpack_dataset
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
import shutil
from tqdm import tqdm

from InstanceSegFrac.nets.ContrastiveLearning.contrastiveNets.ContrastiveAttetion import TwoClassOnePosMaskedLoss, MultiScaleGlobalContrastiveLoss, VoxelWiseContrastiveLoss
from InstanceSegFrac.nets.ContrastiveLearning.contrastiveNets.ContrastiveAttentionMergedTraining import ContrastivePairAttetionMergedTraining

class nnUNetTrainer_pre_contrastiveAttentionMergedTraining(nnUNetTrainerNoDeepSupervision):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = True
        self.num_epochs = 1000
        self.print_to_log_file("Using VoCo pretrained nnUNet backbone weights !!!!!!!")
        self.loss_dc_and_ce = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)
        self.voxel_Contrast_Loss = VoxelWiseContrastiveLoss()
        self.global_contrast_loss = MultiScaleGlobalContrastiveLoss(
                                    base_loss_fn=TwoClassOnePosMaskedLoss(),
                                    enable_deep_supervision=True,
                                    )
        self.batch_size = 4
        self.debug = False
        self.k_memory = 100
        self.use_momentum_update = True
        
    # @staticmethod
    def build_network_architecture(self, plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = False) -> nn.Module:
        # model = get_network_from_plans(plans_manager, dataset_json, configuration_manager,
        #                        num_input_channels, deep_supervision=enable_deep_supervision)

        deep_supervision = enable_deep_supervision
        num_stages = len(configuration_manager.conv_kernel_sizes)

        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        model = ContrastivePairAttetionMergedTraining(
            input_channels= 1,
            n_stages = 6,
            features_per_stage = (32, 64, 128, 256, 320, 320),
            conv_op = nn.Conv3d,
            kernel_sizes = 3,
            strides = (1, 2, 2, 2, 2, 2),
            n_conv_per_stage = (2, 2, 2, 2, 2, 2),
            num_classes = 4,
            n_conv_per_stage_decoder = (2, 2, 2, 2, 2),
            conv_bias = True,
            norm_op = nn.InstanceNorm3d,
            norm_op_kwargs = {'eps': 1e-5, 'affine': True},
            dropout_op = None,
            dropout_op_kwargs = None,
            nonlin = nn.LeakyReLU,
            nonlin_kwargs = {'inplace': True},
            deep_supervision = deep_supervision,
            projection_dim= 256,
            multi_scale_contrastive = True,
            K_memory= 100,
            use_momentum_update= True,
            batch_size= 4
            )

        return model
    

    def initialize_memory_bank(self, network, dataloader, K, device='cuda'):

        memory_bank = []
        dataloader_iter = iter(dataloader)
        network = network.to(device).eval()
        with torch.no_grad():
            for i in tqdm(range(K), desc='Building Memory Bank'):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)
                data = batch['data'] if isinstance(batch, dict) else batch
                _, _, global_feats = network(data.to(device))
                # global_feats: List of [B, F], length=N, or [B, F]

                if isinstance(global_feats, list):
                    # global_feats: List of [B, F], length N
                    feats = torch.stack(global_feats, dim=1)  # [B, N, F]
                    feats = feats.reshape(-1, feats.shape[-1])  # [B*N, F] (patch优先)
                else:
                    feats = global_feats  # [B, F]
                memory_bank.append(feats)
        memory_bank = torch.cat(memory_bank, dim=0)  # [K*B*N, F] 或 [K*B, F]
        return memory_bank

    
    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()
        
        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(self.enable_deep_supervision)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_pair_dataloaders()
        
        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        print("self.network.prior_memory_bank.shape = ",self.network.prior_memory_bank.shape)
        memory_bank_init = self.initialize_memory_bank(self.network, self.dataloader_train, K=self.k_memory, device=self.device)
        
        print(memory_bank_init.shape)
        print(memory_bank_init)

        with torch.no_grad():
            self.network.prior_memory_bank.copy_(memory_bank_init.detach())
            print("self.network.prior_memory_bank.shape = ",self.network.prior_memory_bank.shape)

    def move_to_device(self, x, device):
        if isinstance(x, (list, tuple)):
            return [t.to(device, non_blocking=True) for t in x]
        return x.to(device, non_blocking=True)

    def split_channels(self, target_list):
        num_channels = target_list[0].shape[1]
        out_lists = [[] for _ in range(num_channels)]
        for t in target_list:
            splitted = torch.split(t, split_size_or_sections=1, dim=1)
            for i, ch in enumerate(splitted):
                out_lists[i].append(ch)
        return out_lists

    def train_step(self, batch: dict) -> dict:

        half = self.batch_size // 2
        pair_data = batch['data']
        pair_target = batch['target']

        pos_data = pair_data[:half]       # [B, C, D, H, W]
        neg_data = pair_data[half:]       # [B, C, D, H, W]

        if self.enable_deep_supervision:

            pos_target = [t[:half] for t in pair_target]     # List[Tensor of shape [B, C, ...]]
            neg_target = [t[half:] for t in pair_target]     # List[Tensor of shape [B, C, ...]]

            _, _, _, _, target_merged_pos, _ = self.split_channels(pos_target)
            _, _, _, _, target_merged_neg, _ = self.split_channels(neg_target)

        else:
            # pair_target is Tensor，shape: [2B, C, D, H, W]

            pos_target = pair_target[:half]     # [B, C, D, H, W]
            neg_target = pair_target[half:]

            _, _, _, _, target_merged_pos, _ = \
                torch.split(pos_target, split_size_or_sections=1, dim=1)

            _, _, _, _, target_merged_neg, _ = \
                torch.split(neg_target, split_size_or_sections=1, dim=1)
            
        pos_data = self.move_to_device(pos_data, self.device)        
        neg_data = self.move_to_device(neg_data, self.device)

        target_merged_pos = self.move_to_device(target_merged_pos, self.device)
        target_merged_neg = self.move_to_device(target_merged_neg, self.device)

        self.optimizer.zero_grad(set_to_none=True)

        seg_merged_logits_neg, _, global_feat_neg = self.network(neg_data)   # proj_neg: (P, F)
        seg_merged_logits_pos, _, global_feat_pos = self.network(pos_data)   # proj_pos: (P, F)

        if self.enable_deep_supervision:

            l_dc_and_ce_neg = self.loss_dc_and_ce(seg_merged_logits_neg, target_merged_neg[0])
            l_dc_and_ce_pos = self.loss_dc_and_ce(seg_merged_logits_pos, target_merged_pos[0])
            l_dc_and_ce = 0.5 *(l_dc_and_ce_neg + l_dc_and_ce_pos)

        else:
            l_dc_and_ce_neg = self.loss_dc_and_ce(seg_merged_logits_neg, target_merged_neg)
            l_dc_and_ce_pos = self.loss_dc_and_ce(seg_merged_logits_pos, target_merged_pos)

            l_dc_and_ce = 0.5 * (l_dc_and_ce_neg + l_dc_and_ce_pos)

        loss_global = self.global_contrast_loss(global_feat_pos, global_feat_neg)

        l = l_dc_and_ce + loss_global

        self.print_to_log_file(f"loss_total = {l:.4f}; l_dc_and_ce = {l_dc_and_ce:.4f};loss_global = {loss_global:.4f}")
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}
        

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        # print(data.shape, target.shape)

        if self.enable_deep_supervision:
            _, _, _, _, target_merged, _ = self.split_channels(target)

        else:
            # pair_target is Tensor，shape: [2B, C, D, H, W]
            _, _, _, _,target_merged, _ = \
                torch.split(target, split_size_or_sections=1, dim=1)
        
        data = data.to(self.device, non_blocking=True)
        target_merged = self.move_to_device(target_merged, self.device)
        seg_logits, _, _ = self.network(data)
        del data
        if self.enable_deep_supervision:
            l_dc_and_ce = self.loss_dc_and_ce(seg_logits, target_merged[0])

        else:
            l_dc_and_ce = self.loss_dc_and_ce(seg_logits, target_merged)

        l = l_dc_and_ce
            
        self.print_to_log_file(f"validation:loss_total = {l:.4f}; l_dc_and_ce = {l_dc_and_ce:.4f}")
        return {'loss': l.detach().cpu().numpy(), 'dc_and_ce_loss': l_dc_and_ce.detach().cpu().numpy(), 'l_infoNCE':l_dc_and_ce.detach().cpu().numpy()}
        

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        dc_and_ce_loss_total = np.mean(outputs_collated['dc_and_ce_loss'])
        self.print_to_log_file('dc_and_ce_loss_total',dc_and_ce_loss_total)
        
        nz = [v for v in outputs_collated['l_infoNCE'] if v != 0]
        infoNCE_loss_total = np.mean(nz) if nz else 0.0
        print(infoNCE_loss_total)
        self.print_to_log_file('infoNCE_loss_total',infoNCE_loss_total)
        if self.is_ddp:
            world_size = dist.get_world_size()

            dc_and_ces = [None for _ in range(world_size)]
            dist.all_gather_object(dc_and_ces, dc_and_ce_loss_total)
            dc_and_ce_loss_total = np.mean(dc_and_ces)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])
            self.print_to_log_file('loss_here',loss_here)

        # Log metrics
        self.logger.log('mean_fg_dice', dc_and_ce_loss_total, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('l_infoNCE', infoNCE_loss_total, self.current_epoch)

    def configure_optimizers(self):

        optimizer = AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay, eps=1e-5)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.0)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler

    def set_deep_supervision_enabled(self, enabled: bool):
        pass


    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()