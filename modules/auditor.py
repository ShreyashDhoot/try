import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
import config 

class Auditor_defination(nn.Module):
    def __init__(self, input_classes=2048):
        super().__init__()
        self.risk_conv = nn.Conv2d(input_classes, 1, kernel_size=1)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        risk_map = self.risk_conv(x)
        intermediate_logits = self.pool(risk_map)
        flattened_logits = torch.flatten(intermediate_logits, 1)
        logits = self.mlp(flattened_logits)
        return logits, risk_map # <=== To be replced by class prediction score, faithfulness etc 

class SafetyAuditor:
    def __init__(self, checkpoint_path, device):
        self.device = device
        
        # 1. Setup CLIP Backbone
        self.feature_extractor = timm.create_model(
            "resnet50_clip.openai",
            pretrained=True,
            features_only=True,
            out_indices=[4]
        ).to(device).eval()

        # 2. Setup Transform
        data_config = timm.data.resolve_model_data_config(self.feature_extractor)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

        # 3. Setup Custom Risk Model (Model_3)
        self.risk_model = Auditor_defination().to(device)
        self.risk_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.risk_model.eval()

    @torch.no_grad()
    def audit(self, image, threshold=None):
        """
        image: Can be a path (str) or a PIL Image
        returns: score (float), binary_mask (torch.Tensor)
        """
        if threshold is None:
            threshold = config.MASK_THRESHOLD

        # Preprocess
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        input_tensor = self.processor(image).unsqueeze(0).to(self.device)

        # Step A: Extract features
        features = self.feature_extractor(input_tensor)
        if isinstance(features, list):
            features = features[-1]

        # Step B: Get Score and Risk Map
        logits, risk_map = self.risk_model(features)
        score = torch.sigmoid(logits).item()

        # Step C: Generate Mask
        prob_map = torch.sigmoid(risk_map)
        upsampled_map = F.interpolate(
            prob_map, 
            size=config.OUTPUT_SIZE, 
            mode='bilinear', 
            align_corners=False
        )
        
        binary_mask = (upsampled_map > threshold).float()

        return score, binary_mask