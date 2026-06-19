from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import math

from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipelineOutput, StableVideoDiffusionPipeline
from PIL import Image
import cv2

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class NormalCrafterPipeline(StableVideoDiffusionPipeline):

    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance, scale=1, image_size=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.video_processor.pil_to_numpy(image) # (0, 255) -> (0, 1)
            image = self.video_processor.numpy_to_pt(image) # (n, h, w, c) -> (n, c, h, w)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            pixel_values = image
            B, C, H, W = pixel_values.shape
            patches = [pixel_values]
            # patches = []
            for i in range(1, scale):
                num_patches_HW_this_level = i + 1
                patch_H = H // num_patches_HW_this_level + 1
                patch_W = W // num_patches_HW_this_level + 1
                for j in range(num_patches_HW_this_level):
                    for k in range(num_patches_HW_this_level):
                        patches.append(pixel_values[:, :, j*patch_H:(j+1)*patch_H, k*patch_W:(k+1)*patch_W])
            
            def encode_image(image):
                image = image * 2.0 - 1.0
                if image_size is not None:
                    image = _resize_with_antialiasing(image, image_size)
                else:
                    image = _resize_with_antialiasing(image, (224, 224))
                image = (image + 1.0) / 2.0

                # Normalize the image with for CLIP input
                image = self.feature_extractor(
                    images=image,
                    do_normalize=True,
                    do_center_crop=False,
                    do_resize=False,
                    do_rescale=False,
                    return_tensors="pt",
                ).pixel_values

                image = image.to(device=device, dtype=dtype)
                image_embeddings = self.image_encoder(image).image_embeds
                if len(image_embeddings.shape) < 3:
                    image_embeddings = image_embeddings.unsqueeze(1)
                return image_embeddings

            image_embeddings = []
            for patch in patches:
                image_embeddings.append(encode_image(patch))
            image_embeddings = torch.cat(image_embeddings, dim=1)    
        
        # duplicate image embeddings for each generation per prompt, using mps friendly method
        # import pdb
        # pdb.set_trace()
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def ecnode_video_vae(self, images, chunk_size: int = 14):
        if isinstance(images, list):
            width, height = images[0].size
        else:
            height, width = images[0].shape[:2]
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        device = self._execution_device
        images = self.video_processor.preprocess_video(images, height=height, width=width).to(device, self.vae.dtype) # torch type in range(-1, 1) with (1,3,h,w)
        images = images.squeeze(0) # from (1, c, t, h, w) -> (c, t, h, w)
        images = images.permute(1,0,2,3) # c, t, h, w -> (t, c, h, w)

        video_latents = []
        # chunk_size = 14
        for i in range(0, images.shape[0], chunk_size):                
            video_latents.append(self.vae.encode(images[i : i + chunk_size]).latent_dist.mode())
        image_latents = torch.cat(video_latents)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        return image_latents

    def pad_image(self, images, scale=64):
        def get_pad(newW, W):
            pad_W = (newW - W) // 2
            if W % 2 == 1:
                pad_Ws = [pad_W, pad_W + 1]
            else:
                pad_Ws = [pad_W, pad_W]
            return pad_Ws

        if type(images[0]) is np.ndarray:
            H, W = images[0].shape[:2]
        else:
            W, H = images[0].size

        if W % scale == 0 and H % scale == 0:
            return images, None
        newW = int(np.ceil(W / scale) * scale)
        newH = int(np.ceil(H / scale) * scale)

        pad_Ws = get_pad(newW, W)
        pad_Hs = get_pad(newH, H)
        
        new_images = []
        for image in images:
            if type(image) is np.ndarray:
                image = cv2.copyMakeBorder(image, *pad_Hs, *pad_Ws, cv2.BORDER_CONSTANT, value=(1.,1.,1.))
                new_images.append(image)
            else:
                image = np.array(image)
                image = cv2.copyMakeBorder(image, *pad_Hs, *pad_Ws, cv2.BORDER_CONSTANT, value=(255,255,255))
                new_images.append(Image.fromarray(image))
        return new_images, pad_Hs+pad_Ws
    
    def unpad_image(self, v, pad_HWs):
        t, b, l, r = pad_HWs
        if t > 0 or b > 0:
            v = v[:, :, t:-b]
        if l > 0 or r > 0:
            v = v[:, :, :, l:-r]
        return v
    
    @torch.no_grad()
    def __call__(
        self,
        images: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        decode_chunk_size: Optional[int] = None,
        time_step_size: Optional[int] = 1,
        window_size: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True
    ):
        images, pad_HWs = self.pad_image(images)
        
        # 0. Default height and width to unet
        width, height = images[0].size
        num_frames = len(images)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(images, height, width)

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = 1.0
        num_videos_per_prompt = 1
        do_classifier_free_guidance = False
        num_inference_steps = 1
        fps = 7
        motion_bucket_id = 127
        noise_aug_strength = 0.
        num_videos_per_prompt = 1
        output_type = "np"
        data_keys = ["normal"]
        use_linear_merge = True
        determineTrain = True
        encode_image_scale = 1
        encode_image_WH = None

        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else 7

        # 3. Encode input image using using clip. (num_image * num_videos_per_prompt, 1, 1024)
        image_embeddings = self._encode_image(images, device, num_videos_per_prompt, do_classifier_free_guidance=do_classifier_free_guidance, scale=encode_image_scale, image_size=encode_image_WH)
        # 4. Encode input image using VAE
        image_latents = self.ecnode_video_vae(images, chunk_size=decode_chunk_size).to(image_embeddings.dtype)
            
        # image_latents [num_frames, channels, height, width] ->[1, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(0)

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # get Start and End frame idx for each window
        def get_ses(num_frames):
            ses = []
            for i in range(0, num_frames, time_step_size):
                ses.append([i, i+window_size])
            num_to_remain = 0
            for se in ses:
                if se[1] > num_frames:
                    continue
                num_to_remain += 1
            ses = ses[:num_to_remain]

            if ses[-1][-1] < num_frames:
                ses.append([num_frames - window_size, num_frames])
            return ses
        ses = get_ses(num_frames)

        pred = None
        for i, se in enumerate(ses):
            window_num_frames = window_size
            window_image_embeddings = image_embeddings[se[0]:se[1]]
            window_image_latents = image_latents[:, se[0]:se[1]]
            window_added_time_ids = added_time_ids
            # import pdb
            # pdb.set_trace()
            if i == 0 or time_step_size == window_size:
                to_replace_latents = None
            else:
                last_se = ses[i-1]
                num_to_replace_latents = last_se[1] - se[0]
                to_replace_latents = pred[:, -num_to_replace_latents:]

            latents = self.generate(
                num_inference_steps,
                device,
                batch_size,
                num_videos_per_prompt,
                window_num_frames,
                height,
                width,
                window_image_embeddings,
                generator,
                determineTrain,
                to_replace_latents,
                do_classifier_free_guidance,
                window_image_latents,
                window_added_time_ids
            )
            
            # merge last_latents and current latents in overlap window
            if to_replace_latents is not None and use_linear_merge:
                num_img_condition = to_replace_latents.shape[1]
                weight = torch.linspace(1., 0., num_img_condition+2)[1:-1].to(device)
                weight = weight[None, :, None, None, None]
                latents[:, :num_img_condition] = to_replace_latents * weight + latents[:, :num_img_condition] * (1 - weight)

            if pred is None:
                pred = latents
            else:
                pred = torch.cat([pred[:, :se[0]], latents], dim=1)

        if not output_type == "latent":
            # cast back to fp16 if needed
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            # latents has shape (1, num_frames, 12, h, w)

            def decode_latents(latents, num_frames, decode_chunk_size):
                frames = self.decode_latents(latents, num_frames, decode_chunk_size) # in range(-1, 1)
                frames = self.video_processor.postprocess_video(video=frames, output_type="np")
                frames = frames * 2 - 1 # from range(0, 1) -> range(-1, 1)
                return frames
            
            frames = decode_latents(pred, num_frames, decode_chunk_size)
            if pad_HWs is not None:
                frames = self.unpad_image(frames, pad_HWs)
        else:
            frames = pred

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)


    def generate(
            self, 
            num_inference_steps,
            device,
            batch_size,
            num_videos_per_prompt,
            num_frames,
            height,
            width,
            image_embeddings,
            generator,
            determineTrain,
            to_replace_latents,
            do_classifier_free_guidance,
            image_latents,
            added_time_ids,
            latents=None,
        ):
        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        if determineTrain:
            latents[...] = 0.

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # replace part of latents with conditons. ToDo: t embedding should also replace
                if to_replace_latents is not None:
                    num_img_condition = to_replace_latents.shape[1]
                    if not determineTrain:
                        _noise = randn_tensor(to_replace_latents.shape, generator=generator, device=device, dtype=image_embeddings.dtype)
                        noisy_to_replace_latents = self.scheduler.add_noise(to_replace_latents, _noise, t.unsqueeze(0))
                        latents[:, :num_img_condition] = noisy_to_replace_latents
                    else:
                        latents[:, :num_img_condition] = to_replace_latents
                    

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                timestep = t
                # Concatenate image_latents over channels dimention
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_output = self.scheduler.step(noise_pred, t, latents)
                latents = scheduler_output.prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        return latents 
# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out
