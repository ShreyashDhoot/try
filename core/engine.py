import torch
from tqdm import tqdm
from .utils import decode_to_pil
import config
from inpainter import SurgicalInpainter

class DiffusionEngine:
    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device

    def generate_with_audit(self, prompt, safe_target, auditor, surgeon, num_steps=50):
        # 1. Setup Initial Latents
        generator = torch.Generator(device=self.device).manual_seed(42)
        latents = torch.randn((1, 4, 64, 64), device=self.device, generator=generator).to(dtype=torch.float16)
        
        self.pipe.scheduler.set_timesteps(num_steps, device=self.device)
        latents = latents * self.pipe.scheduler.init_noise_sigma

        # 2. Encode Prompt
        text_embeddings = self.pipe._encode_prompt(prompt, self.device, 1, True).to(dtype=torch.float16)

        # 3. Diffusion Loop
        for i, t in enumerate(tqdm(self.pipe.scheduler.timesteps, desc="Generating")):
            # Standard Denoising Step
            print(f"denoising for step {i}")
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            with torch.no_grad():
                noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + config.GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample 

            current_pil = decode_to_pil(self.pipe, latents)

            score, binary_mask = auditor.audit(current_pil)
            print(f'score=====================> {score}')

            if score > config.GLOBAL_THRESHOLD:
                print(f"| Step {i} | Adversary Detected (Score: {score:.2f}) | Deploying Surgeon...")
                    
                # Call the modular inpainter logic
                # This handles the internal pixel-fix and the latent graft
                latents, _ = SurgicalInpainter.generate_fix(
                    prompt=safe_target,
                    current_pil=current_pil,
                    binary_mask=binary_mask,
                    t=t,
                    latents=latents
                )
        
        return decode_to_pil(self.pipe, latents)