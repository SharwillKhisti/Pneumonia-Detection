from .resnet_model import get_resnet18, get_resnet50
from .densenet_model import get_densenet121

MODEL_REGISTRY = {
    "resnet18": get_resnet18,
    "resnet50": get_resnet50,
    "densenet121": get_densenet121,
}

def get_model(model_name, num_classes=2):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")
    print(f"🧠 Loading model: {model_name}")
    model = MODEL_REGISTRY[model_name](num_classes)
    print("✅ Model loaded and classifier head replaced.")
    return model