import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from MAE.util.misc import get_mae_model
from .networks import MamerFuse_net, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss


LOGGER = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        
        self.name = name
        self.config = config
        self.iteration = 0
        if config.MODE == 2:
            if config.MODELPATH is not None:
                self.gen_weights_path = config.MODELPATH
        if config.MODE == 1:
            if config.RESUMEPATH is not None:
                self.gen_weights_path = os.path.join(config.RESUMEPATH + '_gen.pth')
                self.dis_weights_path = os.path.join(config.RESUMEPATH + '_dis.pth')

    def load(self):
        
        if os.path.exists(self.gen_weights_path):     
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'], strict=False)
            self.iteration = data['iteration']
            LOGGER.info('Generator Loading from %s ' % self.gen_weights_path)
        else:
            raise ValueError("Generator is None") 
            
        # load discriminator only when training
        if self.config.MODE == 1: 
            if os.path.exists(self.dis_weights_path):
                if torch.cuda.is_available():
                    data = torch.load(self.dis_weights_path)
                else:
                    data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

                self.discriminator.load_state_dict(data['discriminator'])
                LOGGER.info('Discriminator Loading from %s ' % self.dis_weights_path)
            else:
                raise ValueError("discriminator is None")    
            
    def save(self, path, name):
        LOGGER.info('saving model at %s' % path)

        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, os.path.join(path, name + '_gen.pth'))
        LOGGER.info('generator is saved')
        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, os.path.join(path, 'discriminator', name + '_dis.pth'))
        LOGGER.info('discriminator is saved')


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        generator = MamerFuse_net()
        # LOGGER.info(generator)
        if config.MAE_MODELPATH is not None:
            LOGGER.info("MAE Loading checkpoint: {} ...".format(config.MAE_MODELPATH))
            MAE_model = get_mae_model('mae_vit_base_patch16', mask_decoder=True)
            checkpoint = torch.load(config.MAE_MODELPATH, map_location='cpu')
            MAE_model.load_state_dict(checkpoint['model'])
            self.add_module('MAE_model', MAE_model)

        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.MAE_model.requires_grad_(False).eval()

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

        #### learning rate decay
        # self.gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.gen_optimizer, last_epoch=-1,
        #                                                           milestones=[20000, 40000, 60000, 80000, 120000],
        #                                                           gamma=self.config.LR_Decay)
        # self.dis_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.dis_optimizer, last_epoch=-1,
        #                                                           milestones=[20000, 40000, 60000, 80000, 120000],
        #                                                           gamma=self.config.LR_Decay)
        self.gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.gen_optimizer,
            mode='max',             
            factor=self.config.LR_Decay,              
            patience=5,              
            verbose=True             
        )

        self.dis_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.dis_optimizer,
            mode='max',
            factor=self.config.LR_Decay,
            patience=5,
            verbose=True
        )

        self.scaler = torch.cuda.amp.GradScaler()

    def process(self, images, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs

        outputs_img = self(images, masks)

        gen_loss = 0
        dis_loss = 0

        with torch.cuda.amp.autocast():
            # discriminator loss
            dis_input_real = images
            dis_input_fake = outputs_img.detach()

            dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
            dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]

            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2

            gen_input_fake = outputs_img

            gen_fake, _ = self.discriminator(gen_input_fake)
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_loss += gen_gan_loss

            gen_l1_loss = self.l1_loss(outputs_img, images) * self.config.L1_LOSS_WEIGHT

            gen_loss += gen_l1_loss

            # generator perceptual loss
            gen_content_loss = self.perceptual_loss(outputs_img, images)
            gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
            gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs_img * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # create logs
        logs = [
            ("gLoss", round(gen_loss.item(), 4)),
            ("dLoss", round(dis_loss.item(), 4)),
            ("advLoss", round(gen_gan_loss.item(), 4)),
            ("l1Loss", round(gen_l1_loss.item(), 4)),
            ("percepLoss", round(gen_content_loss.item(), 4)),
            ("styleLoss", round(gen_style_loss.item(), 4)),
        ]

        return outputs_img, gen_loss, dis_loss, logs, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss

    def forward(self, images, masks):
        images_masked = (images * (1 - masks).float()) + masks
        inputs = images_masked
        self.MAE_model.eval()
        with torch.no_grad():
            mae_feats, scores = self.MAE_model.forward_return_feature(images, masks)
        outputs_img = self.generator(inputs, masks, mae_feats, scores)

        return outputs_img

    def backward(self, gen_loss=None, dis_loss=None):
        self.scaler.scale(dis_loss).backward(retain_graph=True)

        self.scaler.scale(gen_loss).backward()

        self.scaler.step(self.dis_optimizer)

        self.scaler.step(self.gen_optimizer)
        self.scaler.update()


    def backward_joint(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()

