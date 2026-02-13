import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline
from config import INPAINTER_MODEL_ID,dtype,device
import config

class SurgicalInpainter:
    def __init__(self, model_id, base_vae, base_scheduler, device):
        """
        model_id: HuggingFace ID or local path to inpainter weights
        base_vae: Shared VAE from the main pipeline to save VRAM
        base_scheduler: Shared scheduler to ensure consistent noise math
        """
        self.device = device
        self.vae = base_vae
        self.scheduler = base_scheduler
        
        # Load the inpainting pipeline
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            INPAINTER_MODEL_ID,
            torch_dtype=dtype,
            variant="fp16"
        ).to(device)
        
        # Critical: Share the VAE to save memory and ensure latent consistency
        self.pipe.vae = self.vae

    def generate_fix(self, prompt, image, mask_image):
        """Generates the 'Select' version of the pixels."""
        return self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image
        ).images[0]

    @torch.no_grad()
    def run(self, inpainted_image, binary_mask, t, current_latents):
        """
        Performs the Latent Blending and SDEdit (re-noising) step.
        """
        # 1. Convert Image to Latent space
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # Scale to [-1, 1]
        ])
        it = transform(inpainted_image).unsqueeze(0).to(self.device, dtype=torch.float16)

        # 2. Encode to Fixed Latents
        z_fixed = self.vae.encode(it).latent_dist.mode() * config.VAE_SCALE_FACTOR
        
        # 3. SDEdit: Add noise to match the current diffusion timestep 't'
        noise = torch.randn_like(z_fixed)
        z_fixed_noisy = self.scheduler.add_noise(z_fixed, noise, t)

        # 4. Latent Blending: Stitch the 'fixed' noisy part into the 'original' noisy part
        m = F.interpolate(binary_mask, size=config.LATENT_SIZE, mode='bilinear')
        # Optional: Apply Gaussian Blur to 'm' here if not done in auditor
        
        updated_latents = (1 - m) * current_latents + (m * z_fixed_noisy)
        
        return updated_latents