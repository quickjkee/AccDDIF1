import torch
import numpy as np

from tqdm import tqdm


class DiffModel(torch.nn.Module):

    def __init__(self,
                 net,
                 num_steps,
                 device='cpu'):
        """
        Model to fune-tune.
        It should contain:
        - noising of real images
        - sampling from noise
        - schedule of timestep

        :param net: Neural network predicting noise or x0
        :param num_steps: Number of steps for sampling
        """
        super(DiffModel, self).__init__()

        # Base configurations
        self.device = device
        self.net = net.to(device)
        self.num_steps = num_steps

        # Specific configurations
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 0.5
        self._conf_schedule()

    # Prediction of x0 based on the provided time step and noised image
    # ----------------------------------------------------------------------------
    def single_step(self, noised_images, sigma):
        """
        :param noised_images: [b_size, C, W, H], in [-1, 1] range
        :param sigma: [b_size, 1, 1, 1]
        :return:
        """
        x0_pred = self.net(noised_images, sigma, None, None)

        return x0_pred
    # ----------------------------------------------------------------------------

    # Noising clear image
    # ----------------------------------------------------------------------------
    def noising_images(self, images, sigma):
        noise = torch.randn_like(images) * sigma

        return images + noise
    # ----------------------------------------------------------------------------

    # ----------------------------------------------------------------------------
    @torch.no_grad()
    def sample_batch_from_noise(
            self,
            b_size=64,
            seed=0,
            S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    ):
        #torch.manual_seed(seed)
        net = self.net

        # Pick latents and labels.
        print(f'Generating {b_size} images...')
        latents = torch.randn([b_size, net.img_channels, net.img_resolution, net.img_resolution],
                              device=self.device)
        class_labels = None

        # Calculation of time steps based on the provided number of the steps
        t_steps = torch.tensor(self.t_steps).to(self.device)

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        interm_x0s = []
        for i, (t_cur, t_next) in tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))),
                                            unit='step'):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / self.num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            interm_x0s.append(denoised.cpu())
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
                interm_x0s.append(denoised.cpu())
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next, interm_x0s
    # ----------------------------------------------------------------------------

    # Random time step from the predefined schedule
    # ----------------------------------------------------------------------------
    def get_random_from_schedule(self, images):
        """
        The same timestep for each image in batch
        :return: (Tensor), [b_size]
        """
        ones = torch.ones([images.shape[0], 1, 1, 1], device=images.device)

        for i in range(len(images)):
            random_t = np.random.choice(self.t_steps_to_sample, 1)[0]
            ones[i] *= random_t

        #sigma = ones * random_t

        return ones
    # ----------------------------------------------------------------------------

    # Random time step
    # ----------------------------------------------------------------------------
    def get_random_time(self, images):
        """
        Different timestep for each image in batch
        :return: (Tensor), [b_size]
        """
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        return sigma
    # ----------------------------------------------------------------------------

    ########################################################################
    #
    # UTILS FUNCTIONS
    #
    ########################################################################

    # Calculation of schedule based on the number of steps
    # ----------------------------------------------------------------------------
    def _conf_schedule(self, sigma_min=0.002, sigma_max=80, rho=7):
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(self.num_steps,
                                    dtype=torch.float64)

        t_steps = (sigma_max ** (1 / rho) + step_indices / (self.num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.net.round_sigma(t_steps),
                             torch.zeros_like(t_steps[:1])])  # t_N = 0

        self.t_steps = t_steps.cpu().numpy()
        print(self.t_steps)
        self.t_steps_to_sample = self.t_steps[self.t_steps > 0.1]
        #self.t_steps_to_sample = self.t_steps_to_sample[self.t_steps_to_sample < 40]
    # ----------------------------------------------------------------------------

    # Change regime from train to eval and vice versa
    # ----------------------------------------------------------------------------
    def change_regime(self, regime='train'):
        if regime == 'train':
            self.net.train().requires_grad_(True)
        elif regime == 'test':
            self.net.eval().requires_grad_(False)
        else:
            print('Unknown regime, train and eval are only available')
    # ----------------------------------------------------------------------------
