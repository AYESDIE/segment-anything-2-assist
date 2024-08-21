import typing
import os
import tqdm
import requests
import torch

import sam2.build_sam
import sam2.automatic_mask_generator

import cv2 

SAM2_MODELS = {
    "sam2_hiera_tiny": {
      "download_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
      "model_path": ".tmp/checkpoints/sam2_hiera_tiny.pt",
      "config_file": "sam2_hiera_t.yaml"
    },
    "sam2_hiera_small": {
      "download_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
      "model_path": ".tmp/checkpoints/sam2_hiera_small.pt",
      "config_file": "sam2_hiera_s.yaml"
    },
    "sam2_hiera_base_plus": {
      "download_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
      "model_path": ".tmp/checkpoints/sam2_hiera_base_plus.pt",
      "config_file": "sam2_hiera_b+.yaml"
    },
    "sam2_hiera_large": {
      "download_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
      "model_path": ".tmp/checkpoints/sam2_hiera_large.pt",
      "config_file": "sam2_hiera_l.yaml"
    },
}
      
class SegmentAnything2Assist:
  def __init__(
    self,
    model_name: str | typing.Literal["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_base_plus", "sam2_hiera_large"] = "sam2_hiera_small",
    configuration: str |typing.Literal["Automatic Mask Generator", "Image"] = "Automatic Mask Generator",
    download_url: str | None = None,
    model_path: str | None = None,
    download: bool = True,
    device: str | torch.device = torch.device("cpu"),
    verbose: bool = True
  ) -> None:
    assert model_name in SAM2_MODELS.keys(), f"`model_name` should be either one of {list(SAM2_MODELS.keys())}"
    assert configuration in ["Automatic Mask Generator", "Image"]

    self.model_name = model_name
    self.configuration = configuration
    self.config_file = SAM2_MODELS[model_name]["config_file"]
    self.device = device

    self.download_url = download_url if download_url is not None else SAM2_MODELS[model_name]["download_url"]
    self.model_path = model_path if model_path is not None else SAM2_MODELS[model_name]["model_path"]
    os.makedirs(os.path.dirname(self.model_path), exist_ok = True)
    self.verbose = verbose

    if verbose:
      print(f"SegmentAnything2Assist::__init__::Model Name: {self.model_name}")
      print(f"SegmentAnything2Assist::__init__::Configuration: {self.configuration}")
      print(f"SegmentAnything2Assist::__init__::Download URL: {self.download_url}")
      print(f"SegmentAnything2Assist::__init__::Default Path: {self.model_path}")

    if download:
      self.download_model()

    if self.is_model_available():
      self.sam2 = sam2.build_sam.build_sam2(config_file = self.config_file, checkpoint = self.model_path, device = self.device)
      if verbose:
        print("SegmentAnything2Assist::__init__::SAM2 is loaded.")
    else:
      self.sam2 = None
      if verbose:
        print("SegmentAnything2Assist::__init__::SAM2 is not loaded.")


  def is_model_available(self) -> bool:
    return os.path.exists(self.model_path)

  def load_model(self) -> None:
    if self.is_model_available():
      self.sam2 = sam2.build_sam(checkpoint = self.model_path)

  def download_model(
    self, 
    force: bool = False
  ) -> None:
    if not force and self.is_model_available():
        print(f"{self.model_path} already exists. Skipping download.")
        return

    response = requests.get(self.download_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(self.model_path, 'wb') as file, tqdm.tqdm(total = total_size, unit = 'B', unit_scale = True) as progress_bar:
        for data in response.iter_content(chunk_size = 1024):
            file.write(data)
            progress_bar.update(len(data))

  def generate_automatic_masks(
    self,
    image
  ):
    if self.sam2 is None:
      print("SegmentAnything2Assist::generate_automatic_masks::SAM2 is not loaded.")
      return None
    
    agg = sam2.automatic_mask_generator.SAM2AutomaticMaskGenerator(self.sam2, 
                                                                   points_per_batch = 10,
                                                                   pred_iou_thresh = 0.1)
    print("Generating Masks")
    masks = agg.generate(image)
    print(masks)

image = cv2.imread("xd.jpg", cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = cv2.resize(image, dsize = None, fx = 0.15, fy = 0.15)
print(image.shape)
cv2.imshow("Image", image)

sgm = SegmentAnything2Assist(model_name = "sam2_hiera_small")
sgm.generate_automatic_masks(image)