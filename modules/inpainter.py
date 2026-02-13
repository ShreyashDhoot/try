import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline
from config import INPAINTER_MODEL_ID,dtype,device
import config

class SurgicalInpainter:
    def __init__(self, model_id, base_vae, base_scheduler, device):
        self.device = device
        self.vae = base_vae
        self.scheduler = base_scheduler

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16"
        ).to(device)

        # Share VAE for latent consistency
        self.pipe.vae = self.vae

    def generate_fix(self, prompt, image, mask_image):
        """Generate fixed pixels in RGB space."""
        return self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image
        ).images[0]

    @torch.no_grad()
    def run(self, prompt, current_pil, binary_mask, t, latents):
        """
        1. Generate safe patch
        2. Convert to latent
        3. Re-noise to timestep t (SDEdit)
        4. Blend into current latents
        """

        # ---- 1. Generate Fixed Image ----
        fixed_image = self.generate_fix(
            prompt=prompt,
            image=current_pil,
            mask_image=binary_mask
        )

        # ---- 2. Convert to latent ----
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        img_tensor = transform(fixed_image).unsqueeze(0).to(
            self.device, dtype=torch.float16
        )

        z_fixed = self.vae.encode(img_tensor).latent_dist.mode()
        z_fixed = z_fixed * config.VAE_SCALE_FACTOR

        # ---- 3. Re-noise to match timestep ----
        noise = torch.randn_like(z_fixed)
        z_fixed_noisy = self.scheduler.add_noise(z_fixed, noise, t)

        # ---- 4. Latent blending ----
        m = F.interpolate(
            binary_mask,
            size=config.LATENT_SIZE,
            mode="bilinear"
        )

        updated_latents = (1 - m) * latents + (m * z_fixed_noisy)

        return updated_latents