from sklearn.metrics import jaccard_score
from models.nested_unet import nested_unet

def evaluate_model(model, test_images, test_masks):
    """Evaluate model performance using DICE Score."""
    preds = model.predict(test_images)
    dice_score = jaccard_score(test_masks.flatten(), preds.flatten(), average='binary')
    return dice_score
