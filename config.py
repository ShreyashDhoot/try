import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
AUDITOR_CHECKPOINT="model_weights/model_3_weights-part_noise.pth"
INPAINTER_MODEL_ID="runwayml/stable-diffusion-inpainting"
DIFFUSION_MODEL_ID="runwayml/stable-diffusion-v1-5"

#================================
dtype=torch.float16
#=================================

#=================================
#Auditor Thresholds 
#=================================
GLOBAL_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
INPUT_SIZE = (224,224)
OUTPUT_SIZE = (512,512)

#=================================
# diffusion parameters 
#=================================
NUM_STEPS = 50
GUIDANCE_SCALE = 0.7
LATENT_SIZE = (64,64)

#=================================
# Surgical Blending 
#=================================
FEATHER_KERNEL_SIZE = 3
VAE_SCALING_FACTOR = 0.18215