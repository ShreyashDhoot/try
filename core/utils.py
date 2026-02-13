import torch
import torch.nn.functional as F
import config

def scale_latents(latents):
    """Scales latents for VAE decoding/encoding."""
    return latents * (1.0 / config.VAE_SCALE_FACTOR)

def unscale_latents(latents):
    """Scales latents back to the UNet manifold."""
    return latents * config.VAE_SCALE_FACTOR

def blend_latents(background_latents, foreground_latents, binary_mask):
    """
    Performs the Latent Blending (Surgical Graft).
    background_latents: The original noisy manifold.
    foreground_latents: The new, fixed noisy manifold (SDEdit).
    binary_mask: The [1, 1, 512, 512] risk map.
    """
    # Resize mask to match latent dimensions (e.g., 64x64)
    m = F.interpolate(
        binary_mask, 
        size=config.LATENT_SIZE, 
        mode='bilinear', 
        align_corners=False
    ).to(background_latents.device, dtype=background_latents.dtype)
    
    # Optional: Apply Gaussian blur to the mask here to soften edges
    # m = gaussian_blur(m, kernel_size=config.FEATHER_KERNEL_SIZE)

    return (1 - m) * background_latents + (m * foreground_latents)

def decode_to_pil(pipe, latents):
    """Helper to convert latents directly to a PIL image for the Auditor."""
    with torch.no_grad():
        image = pipe.decode_latents(latents.detach())
        return pipe.numpy_to_pil(image)[0]