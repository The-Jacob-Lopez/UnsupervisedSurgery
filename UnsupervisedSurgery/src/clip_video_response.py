"""Module provides a class which wraps around CLIP and produces a video response curve"""

import numpy as np
import torch
import clip
import cv2
from PIL import Image

def get_num_video_frames(path):
    return convert_mp4_to_imgs(path, 1)[1]

def convert_mp4_to_imgs(path, img_size):
    images = []

    video_capture = cv2.VideoCapture(path)
    still_reading, image = video_capture.read()
    frame_count = 0
    while still_reading:
        images.append(Image.fromarray(cv2.cvtColor(cv2.resize(image, (img_size,img_size)), cv2.COLOR_BGR2RGB)))
        # read next image
        still_reading, image = video_capture.read()
        frame_count += 1

    return images, frame_count

class clip_video_response:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32")
        self.model.cuda().eval()
        self.input_resolution = self.model.visual.input_resolution
        self.context_length = self.model.context_length
        self.vocab_size = self.model.vocab_size

    def generate_response_curve(self, video_path, queries, sampling_rate):
        images, num_frames = convert_mp4_to_imgs(video_path, self.input_resolution)
        images = [self.preprocess(image) for image in images]
        sampled_images = images[0::sampling_rate]

        image_input = torch.tensor(np.stack(sampled_images)).permute(0,1,2,3).cuda()
        text_tokens = clip.tokenize(queries).cuda()
        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        
        return similarity
