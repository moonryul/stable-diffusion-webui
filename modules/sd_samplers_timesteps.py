import torch
import inspect
import sys
from modules import devices, sd_samplers_common, sd_samplers_timesteps_impl
from modules.sd_samplers_cfg_denoiser import CFGDenoiser
from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback

from modules.shared import opts
import modules.shared as shared

samplers_timesteps = [
    ('DDIM', sd_samplers_timesteps_impl.ddim, ['ddim'], {}),
    ('PLMS', sd_samplers_timesteps_impl.plms, ['plms'], {}),
    ('UniPC', sd_samplers_timesteps_impl.unipc, ['unipc'], {}),
]


samplers_data_timesteps = [
    sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: CompVisSampler(funcname, model), aliases, options)
    for label, funcname, aliases, options in samplers_timesteps
]


class CompVisTimestepsDenoiser(torch.nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_model = model

    def forward(self, input, timesteps, **kwargs):
        return self.inner_model.apply_model(input, timesteps, **kwargs)


class CompVisTimestepsVDenoiser(torch.nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_model = model

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return self.inner_model.sqrt_alphas_cumprod[t.to(torch.int), None, None, None] * v + self.inner_model.sqrt_one_minus_alphas_cumprod[t.to(torch.int), None, None, None] * x_t

    def forward(self, input, timesteps, **kwargs):
        model_output = self.inner_model.apply_model(input, timesteps, **kwargs)
        e_t = self.predict_eps_from_z_and_v(input, timesteps, model_output)
        return e_t


class CFGDenoiserTimesteps(CFGDenoiser):

    def __init__(self, sampler):
        super().__init__(sampler)

        self.alphas = shared.sd_model.alphas_cumprod
        self.mask_before_denoising = True

    def get_pred_x0(self, x_in, x_out, sigma):
        ts = sigma.to(dtype=int)

        a_t = self.alphas[ts][:, None, None, None]
        sqrt_one_minus_at = (1 - a_t).sqrt()

        pred_x0 = (x_in - sqrt_one_minus_at * x_out) / a_t.sqrt()

        return pred_x0

    @property
    def inner_model(self):
        if self.model_wrap is None:
            denoiser = CompVisTimestepsVDenoiser if shared.sd_model.parameterization == "v" else CompVisTimestepsDenoiser
            self.model_wrap = denoiser(shared.sd_model)

        return self.model_wrap


class CompVisSampler(sd_samplers_common.Sampler):
    def __init__(self, funcname, sd_model):
        super().__init__(funcname)

        self.eta_option_field = 'eta_ddim'
        self.eta_infotext_field = 'Eta DDIM'
        self.eta_default = 0.0

        self.model_wrap_cfg = CFGDenoiserTimesteps(self)

    def get_timesteps(self, p, steps):
        discard_next_to_last_sigma = self.config is not None and self.config.options.get('discard_next_to_last_sigma', False)
        if opts.always_discard_next_to_last_sigma and not discard_next_to_last_sigma:
            discard_next_to_last_sigma = True
            p.extra_generation_params["Discard penultimate sigma"] = True

        steps += 1 if discard_next_to_last_sigma else 0

        timesteps = torch.clip(torch.asarray(list(range(0, 1000, 1000 // steps)), device=devices.device) + 1, 0, 999)

        return timesteps
    #MJ:  samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning,image_conditioning=self.image_conditioning)
    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        
        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps) #MJ: t_enc = steps * denoising_strength = 20 *0.2 = 4, for example

        timesteps = self.get_timesteps(p, steps) #MJ: timesteps = 
        timesteps_sched = timesteps[:t_enc] #MJ: timesteps[0:4] = [ 1, 51, 101, 151];  timesteps[20]  implying random noise

        alphas_cumprod = shared.sd_model.alphas_cumprod
        sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[timesteps[t_enc]]) #MJ: timesteps[t_enc]= timesteps[4]=201
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[timesteps[t_enc]])

        xi = x * sqrt_alpha_cumprod + noise * sqrt_one_minus_alpha_cumprod #MJ: Get the blurred version of the initial image:xi = x_{init} = x_{start}
        #MJ: xi= self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise, index = time step t;
        # refer: x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t

        if opts.img2img_extra_noise > 0:
            p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
            extra_noise_params = ExtraNoiseParams(noise, x, xi)
            extra_noise_callback(extra_noise_params)
            noise = extra_noise_params.noise
            xi += noise * opts.img2img_extra_noise * sqrt_alpha_cumprod

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters #MJ: self.func = <function ddim ..>

        if 'timesteps' in parameters:
            extra_params_kwargs['timesteps'] = timesteps_sched
        if 'is_img2img' in parameters:
            extra_params_kwargs['is_img2img'] = True

        self.model_wrap_cfg.init_latent = x
        self.last_latent = x  #MJ: it modifies self, but self.last_latent is not used here
        self.sampler_extra_args = {
            'cond': conditioning, #MJ: len=1
            'image_cond': image_conditioning, #MJ: shape= (1,5,1,1) = dummy image conditioning
            'uncond': unconditional_conditioning, #MJ: len =1
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }
        #MJ: def launch_sampling(self, steps, func): func=ddim
        samples = self.launch_sampling(t_enc + 1, lambda: self.func(self.model_wrap_cfg, xi, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))

        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        return samples
    #MJ: sample() vs sample_img2img()
    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        steps = steps or p.steps
        timesteps = self.get_timesteps(p, steps)

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        if 'timesteps' in parameters:
            extra_params_kwargs['timesteps'] = timesteps

        self.last_latent = x
        self.sampler_extra_args = {
            'cond': conditioning,
            'image_cond': image_conditioning,
            'uncond': unconditional_conditioning,
            'cond_scale': p.cfg_scale,
            's_min_uncond': self.s_min_uncond
        }
        #MJ: def launch_sampling(self, steps, func):
        samples = self.launch_sampling(steps, lambda: self.func(self.model_wrap_cfg, x, extra_args=self.sampler_extra_args, disable=False, callback=self.callback_state, **extra_params_kwargs))

        if self.model_wrap_cfg.padded_cond_uncond:
            p.extra_generation_params["Pad conds"] = True

        return samples


sys.modules['modules.sd_samplers_compvis'] = sys.modules[__name__]
VanillaStableDiffusionSampler = CompVisSampler  # temp. compatibility with older extensions
