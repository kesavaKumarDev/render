from huggingface_hub import hf_hub_download
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

class DeepFakeDetector:
    def __init__(self,
                 model_name="google/vit-huge-patch14-224-in21k",
                 repo_id="Mattupalli/vit_deepfake_receipt_tampering",
                 filename="vit_deepfake.pth"):
       
        print("🧠 Initializing DeepFakeDetector...")    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📟 Using device: {self.device}")
        
        # Load processor
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        print("✅ Loaded ViT processor")
        
        # Load model with default weights
        self.model = ViTForImageClassification.from_pretrained(model_name, num_labels=2).to(self.device)
        print("✅ Loaded base ViT model")
        
        try:
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"📥 Downloaded model checkpoint from Hugging Face: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print("✅ Loaded model_state_dict from checkpoint")
            else:
                self.model.load_state_dict(checkpoint)
                print("✅ Loaded full checkpoint directly")
        except Exception as e:
            print("⚠️ Warning: Could not load fine-tuned weights. Using base model instead.")
            print(f"Error details: {e}")
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        print("✅ Initialization complete")
    
    def predict(self, image_path):
        print(f"🔎 Predicting image: {image_path}")
        self.model.eval()
        
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(image).logits
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        
        label = "Real" if prediction.item() == 0 else "Deepfake"
        return {"label": label, "confidence": confidence.item()}

if __name__ == '__main__':
    detector = DeepFakeDetector()
    result = detector.predict("./processed_image.png")
    print(result)

