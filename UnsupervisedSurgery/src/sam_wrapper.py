from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

class sam_wrapper:
    def __init__(self, sam_checkpoint_path, model_type, device):
        self.sam =  sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
    
    def generate_mask(self, image):
        return self.mask_generator.generate(image)
    
    def generate_display_mask(self, image):
        masks = self.generate_mask(image)
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_anns(masks)
        plt.axis('off')
        plt.show()
        