import torch

import os
INPUT_PATH = os.environ['INPUT_PATH']

from ClipModel.clip import clip

import torchvision.transforms as transforms


class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='cosine', device=torch.device('cuda')):
        super(DirectionLoss, self).__init__()

        self.device = device
        self.clip, clip_preprocess = clip.load(f'{INPUT_PATH}/ViT-B-32.pt', device=self.device)
        self.clip_preprocess = clip_preprocess
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0],
                                                                   std=[2.0, 2.0,
                                                                        2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                             clip_preprocess.transforms[:2] +  # to match ClipModel input scale assumptions
                                             clip_preprocess.transforms[4:])

        self.loss_type = loss_type

        self.loss_func = {
            'mse': torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae': torch.nn.L1Loss
        }[loss_type]()

    # Calculate losses between directions in ClipModel space
    # ----------------------------------------------------------------------------
    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
    # ----------------------------------------------------------------------------

    # Preprocess and encode images
    # ----------------------------------------------------------------------------
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.clip.encode_image(images)
    # ----------------------------------------------------------------------------

    # Get only features w/o preprocessing
    # ----------------------------------------------------------------------------
    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features
    # ----------------------------------------------------------------------------

    # Calculate direction in ClipModel space between source and target images
    # ----------------------------------------------------------------------------
    def compute_image_direction(self, source_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        source_features = self.get_image_features(source_img)
        target_features = self.get_image_features(target_img)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    def direction_loss(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
    # ----------------------------------------------------------------------------

    # Directional loss between two directions
    # ----------------------------------------------------------------------------
    def loss(self,
             src_img,
             pred_img,
             target_img) -> torch.Tensor:

        # Calculate the prediction direction
        src_encoding = self.get_image_features(src_img)
        pred_encoding = self.get_image_features(pred_img)

        pred_direction = (pred_encoding - src_encoding)
        pred_direction /= (pred_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)

        # Calculate the target direction
        with torch.no_grad():
            target_encoding = self.get_image_features(target_img)

            true_direction = (target_encoding - src_encoding)
            true_direction /= true_direction.norm(dim=-1, keepdim=True)

        return self.direction_loss(pred_direction, true_direction)
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    def loss2(self,
              pred_img,
              target_img):

        pred_encoding = self.get_image_features(pred_img)
        target_encoding = self.get_image_features(target_img)

        loss = torch.nn.MSELoss(reduction='sum')(pred_encoding, target_encoding)

        return loss
    # ----------------------------------------------------------------------------
