from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

MODELS = {
    "nano": RFDETRNano,      
    "small": RFDETRSmall,    
    "medium": RFDETRMedium,  
    "base": RFDETRBase,      
    "large": RFDETRLarge,    
}


def build_model(size="base", checkpoint=None, resolution=None):
    kwargs = {}
    if checkpoint:
        kwargs["pretrain_weights"] = checkpoint
    if resolution:
        kwargs["resolution"] = resolution
    return MODELS[size](**kwargs)
