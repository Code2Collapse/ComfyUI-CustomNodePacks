from diffusers import UNetSpatioTemporalConditionModel
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from diffusers.utils import is_torch_version
import torch
from typing import Any, Dict, Optional, Tuple, Union

def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)
        else:
            return module(*inputs)

    return custom_forward
CKPT_KWARGS = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}


class DiffusersUNetSpatioTemporalConditionModelNormalCrafter(UNetSpatioTemporalConditionModel):
    
    @staticmethod
    def forward_crossattn_down_block_dino(
        module,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        dino_down_block_res_samples = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()
        self = module
        blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in blocks:
            if self.training and self.gradient_checkpointing:  # TODO
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    image_only_indicator,
                    **CKPT_KWARGS,
                )
                
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn),
                    hidden_states,
                    encoder_hidden_states,
                    image_only_indicator,
                    False,
                    **CKPT_KWARGS,
                )[0]
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    return_dict=False,
                )[0]

            if dino_down_block_res_samples is not None:
                hidden_states += dino_down_block_res_samples.pop(0)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
                if dino_down_block_res_samples is not None:
                    hidden_states += dino_down_block_res_samples.pop(0)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states
    @staticmethod
    def forward_down_block_dino(
        module,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        dino_down_block_res_samples = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        self = module
        output_states = ()
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:
                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        image_only_indicator,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        image_only_indicator,
                    )
            else:
                hidden_states = resnet(
                    hidden_states,
                    temb,
                    image_only_indicator=image_only_indicator,
                )
            if dino_down_block_res_samples is not None:
                hidden_states += dino_down_block_res_samples.pop(0)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
                if dino_down_block_res_samples is not None:
                    hidden_states += dino_down_block_res_samples.pop(0)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states
    
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
        image_controlnet_down_block_res_samples = None,
        image_controlnet_mid_block_res_sample = None,
        dino_down_block_res_samples = None,

    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.FloatTensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead of a plain
                tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        if not hasattr(self, "custom_gradient_checkpointing"):
            self.custom_gradient_checkpointing = False

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        if len(timesteps.shape) == 1:
            timesteps = timesteps.expand(batch_size)
        else:
            timesteps = timesteps.reshape(batch_size * num_frames)
        t_emb = self.time_proj(timesteps) # (B, C)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb) # (B, C)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        if emb.shape[0] == 1:
            emb = emb + aug_emb
            # Repeat the embeddings num_video_frames times
            # emb: [batch, channels] -> [batch * frames, channels]
            emb = emb.repeat_interleave(num_frames, dim=0)
        else:
            aug_emb = aug_emb.repeat_interleave(num_frames, dim=0)
            emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)

        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        # here, our encoder_hidden_states is [batch * frames, 1, channels]        
            
        if not sample.shape[0] == encoder_hidden_states.shape[0]:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        if dino_down_block_res_samples is not None:
            dino_down_block_res_samples = [x for x in dino_down_block_res_samples]
            sample += dino_down_block_res_samples.pop(0)
    
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if dino_down_block_res_samples is None:
                if self.custom_gradient_checkpointing:
                    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                        sample, res_samples = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(downsample_block),
                            sample,
                            emb,
                            encoder_hidden_states,
                            image_only_indicator,
                            **CKPT_KWARGS,
                        )
                    else:
                        sample, res_samples = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(downsample_block),
                            sample,
                            emb,
                            image_only_indicator,
                            **CKPT_KWARGS,
                        )
                else:
                    if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            encoder_hidden_states=encoder_hidden_states,
                            image_only_indicator=image_only_indicator,
                        )
                    else:
                        sample, res_samples = downsample_block(
                            hidden_states=sample,
                            temb=emb,
                            image_only_indicator=image_only_indicator,
                        )
            else:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = self.forward_crossattn_down_block_dino(
                        downsample_block,
                        sample,
                        emb,
                        encoder_hidden_states,
                        image_only_indicator,
                        dino_down_block_res_samples,
                    )
                else:
                    sample, res_samples = self.forward_down_block_dino(
                        downsample_block,
                        sample,
                        emb,
                        image_only_indicator,
                        dino_down_block_res_samples,
                    )
            down_block_res_samples += res_samples

        if image_controlnet_down_block_res_samples is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, image_controlnet_down_block_res_sample in zip(
                down_block_res_samples, image_controlnet_down_block_res_samples
            ):
                down_block_res_sample = (down_block_res_sample + image_controlnet_down_block_res_sample) / 2
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.custom_gradient_checkpointing:
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block),
                sample,
                emb,
                encoder_hidden_states,
                image_only_indicator,
                **CKPT_KWARGS,
        )
        else:
            sample = self.mid_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
            )
        
        if image_controlnet_mid_block_res_sample is not None:
            sample = (sample + image_controlnet_mid_block_res_sample) / 2

        # 5. up
        mid_up_block_out_samples = [sample, ]
        down_block_out_sampels = []
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            down_block_out_sampels.append(res_samples[-1])
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                if self.custom_gradient_checkpointing:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(upsample_block),
                        sample,
                        res_samples,
                        emb,
                        encoder_hidden_states,
                        image_only_indicator,
                        **CKPT_KWARGS
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                    )
            else:
                if self.custom_gradient_checkpointing:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(upsample_block),
                        sample,
                        res_samples,
                        emb,
                        image_only_indicator,
                        **CKPT_KWARGS
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        image_only_indicator=image_only_indicator,
                    )
            mid_up_block_out_samples.append(sample)
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        if self.custom_gradient_checkpointing:
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.conv_out),
                    sample,
                    **CKPT_KWARGS
                )
        else:
            sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample, down_block_out_sampels[::-1], mid_up_block_out_samples)

        return UNetSpatioTemporalConditionOutput(sample=sample)