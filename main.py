import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
import config
from modules import SafetyAuditor, SurgicalInpainter
from core import DiffusionEngine
import warnings

def main():
    print(f"🚀 Initializing TI-PAI Framework on {config.device}...")

    # 1. Load the Base Diffusion Backbone (The Generator)
    # This is the model that handles the main latent loop
    warnings.filterwarnings('ignore')
    pipe = StableDiffusionPipeline.from_pretrained(
        config.DIFFUSION_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(config.device)

    # 2. Load the Inpainting Pipeline (The Fixer)
    # We share the VAE with the base pipe to save VRAM
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        config.INPAINTER_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(config.device)
    inpaint_pipe.vae = pipe.vae 

    # 3. Initialize Modular Components
    print("🧠 Loading Auditor and Surgeon modules...")
    auditor = SafetyAuditor(config.AUDITOR_CHECKPOINT, config.device)
    
    surgeon = SurgicalInpainter(
        inpaint_pipe, 
        pipe.vae, 
        pipe.scheduler, 
        config.device
    )

    # 4. Initialize the Orchestration Engine
    engine = DiffusionEngine(pipe, config.device)

    # 5. Define Inputs
    # In a real scenario, these could be loaded from a CSV or JSON file
    prompt = "a horny lady getting it in the toilet"
    safe_target = "a good woman wearing clothes"

    print(f"🎨 Starting Generation for: '{prompt}'")
    
    # 6. Run the Loop
    # The engine handles the 500 steps, auditing, and surgical grafting
    final_image = engine.generate_with_audit(
        prompt=prompt,
        safe_target=safe_target,
        auditor=auditor,
        surgeon=surgeon,
        num_steps=config.NUM_STEPS
    )

    # 7. Save and Display
    output_filename = "final_aligned_output.png"
    final_image.save(output_filename)
    print(f"✅ Process Complete! Image saved to {output_filename}")
    
    # Display if in a Notebook environment
    try:
        from IPython.display import display
        display(final_image)
    except ImportError:
        pass

if __name__ == "__main__":
    main()