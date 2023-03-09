import torch
import os
import pickle
import numpy as np

from tqdm import tqdm

from data_utils import save_batch, calc_fid, delete_and_create_dir


class FineTuner(object):
    def __init__(self,
                 model,
                 clip,
                 dataset,
                 device='cpu',
                 n_iters=50,
                 lr=2e-6):
        """
        Fine-tuner that update the provided model
        """
        super(FineTuner, self).__init__()

        # Base settings
        self.model = model
        self.clip = clip
        self.dataset_iterator = dataset

        # Specific settings
        self.lr = lr
        self.device = device
        self.n_iters = n_iters

        self.fid_stats = {}

    # ----------------------------------------------------------------------------
    def fine_tune(self):

        # Make the model trainable
        self.model.change_regime('train')
        self._conf_opt()

        # FINE-TUNING PHASE
        print('Fine tune')

        # STEP 0. Initial estimation
        self._estim_and_save('runs', is_first=True)

        for it in tqdm(range(self.n_iters)):

            # STEP 1. Sample batch of images
            images = next(self.dataset_iterator)[0].to(self.device)
            images = images.to(torch.float32) / 127.5 - 1

            # STEP 2. Sample random schedule to noise the images
            t_steps = self.model.net.round_sigma(80).cuda() #self.model.get_random_from_schedule(images)).cuda()
            noised_images = self.model.noising_images(images, t_steps)

            # STEP 3. Predict the real image
            pred_images = self.model.single_step(noised_images, t_steps)

            # STEP 4. Loss calculation and updates the model
            loss_clip = (2 - self.clip.loss(noised_images, pred_images, images)) / 2
            loss_clip = -torch.log(loss_clip)

            self.optim_ft.zero_grad()
            loss_clip.backward()
            self.optim_ft.step()

            self._estim_and_save('runs')

            print(f"{t_steps}, CLIP {round(loss_clip.item(), 3)}")
    # ----------------------------------------------------------------------------

    ########################################################################
    #
    # UTILS FUNCTIONS
    #
    ########################################################################

    # ----------------------------------------------------------------------------
    def _conf_opt(self):
        print(f"Setting optimizer with lr={self.lr}")

        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        self.optim_ft = torch.optim.Adam(params_to_update,
                                         weight_decay=0,
                                         lr=self.lr)
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    @torch.no_grad()
    def _estim_and_save(self, path='runs', max_samples=50000, is_first=False):
        """
        Generation
        :param path:
        :param is_final:
        :return:
        """
        print('Estimation started')

        path_samples = f'{path}/samples'
        delete_and_create_dir(path_samples)

        final_samples = []
        x0s_samples = [[] for _ in range(self.model.num_steps)]
        n_generated = 0

        # Generation of max_samples objects
        #######################
        while True:
            # Batch generation
            final, x0s = self.model.sample_batch_from_noise()

            final_samples.append(final)
            for i, x0 in enumerate(x0s):
                x0s_samples[i].append(x0)

            n_generated += len(final)
            if n_generated >= max_samples:
                break
        #######################

        # FID calculation for all estimations and final objects
        #######################
        # FID for all estimations
        for step, x0s_step in enumerate(x0s_samples):
            iter_ = 0
            for batch_x0s in x0s_step:
                for x0 in batch_x0s:
                    if iter_ <= (max_samples - 1):
                        save_batch(x0.unsqueeze(0), f'{path_samples}/{iter_}.png')
                        iter_ += 1

            fid = calc_fid(path_samples, 'ffhq', max_samples, len(final), is_first)
            is_first = False

            # Update the fid stats
            try:
                self.fid_stats[step].append(fid)
            except KeyError:
                self.fid_stats[step] = []
                self.fid_stats[step].append(fid)

            delete_and_create_dir(path_samples)

        # FID for the final samples
        iter_ = 0
        for batch_x0s in final_samples:
            for x0 in batch_x0s:
                if iter_ <= (max_samples - 1):
                    save_batch(x0.unsqueeze(0), f'{path_samples}/{iter_}.png')
                    iter_ += 1

        fid = calc_fid(path_samples, 'ffhq', max_samples, len(final), is_first)

        # Update the fid stats
        try:
            self.fid_stats['final'].append(fid)
        except KeyError:
            self.fid_stats['final'] = []
            self.fid_stats['final'].append(fid)

        delete_and_create_dir(path_samples)
        #######################

        with open(f'{path}/fid_stats.pickle', 'wb') as handle:
            pickle.dump(self.fid_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Estimation ended')
    # ----------------------------------------------------------------------------
    @torch.no_grad()
    def generate_image_grid(
            self,
            gridw=8, gridh=8, device=torch.device('cuda'),
            num_steps=40, sigma_min=0.002, sigma_max=80, rho=7,
            S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    ):
        batch_size = gridw * gridh
        #torch.manual_seed(seed)

        # Pick latents and labels.
        print(f'Generating {batch_size} images...')
        latents = torch.randn([batch_size, self.model.net.img_channels, self.model.net.img_resolution, self.model.net.img_resolution], device=device)

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.model.net.sigma_min)
        sigma_max = min(sigma_max, self.model.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.model.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))),
                                            unit='step'):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.model.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.model.net(x_hat, t_hat, None).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.model.net(x_next, t_next, None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next
