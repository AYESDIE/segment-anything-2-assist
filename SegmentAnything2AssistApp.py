import gradio
import gradio_image_annotation 
import gradio_imageslider
import numpy

import src.SegmentAnything2Assist as SegmentAnything2Assist

example_image_annotation = {
    "image": "assets/cars.jpg",
    "boxes": [{'label': '+', 'color': (0, 255, 0), 'xmin': 886, 'ymin': 552, 'xmax': 886, 'ymax': 552}, {'label': '-', 'color': (255, 0, 0), 'xmin': 1239, 'ymin': 576, 'xmax': 1239, 'ymax': 576}, {'label': '-', 'color': (255, 0, 0), 'xmin': 610, 'ymin': 573, 'xmax': 610, 'ymax': 573}, {'label': '', 'color': (0, 0, 255), 'xmin': 331, 'ymin': 597, 'xmax': 1347, 'ymax': 1047}]
}

class SegmentAnything2AssistUI:
    def __init__(self):
        self.segment_anything2assist = SegmentAnything2Assist.SegmentAnything2Assist(model_name = "sam2_hiera_large")
        
        self.base_app = gradio.Blocks()
        self.title_markdown = gradio.Markdown("# SegmentAnything2Assist", render = False)
        
        self.image_tab = gradio.Tab(label = "Image Segmentation", id = "image_tab", render = False)
        self.image_input = gradio_image_annotation.image_annotator(example_image_annotation, render = False)
        self.image_output = gradio_imageslider.ImageSlider(render = False)
        self.image_generate_mask_button = gradio.Button("Generate Mask", render = False)
        self.image_generate_SAM_mask_threshold = gradio.Slider(0.0, 1.0, 0.5, label = "SAM Mask Threshold", render = False)
        self.__image_point_coords = []
        self.__image_point_labels = []
        self.__image_box = []
        
        self.video_tab = gradio.Tab(label = "Video Segmentation", id = "video_tab", render = False)
        
        self.generate_ui()
    
    def generate_ui(self):
        with self.base_app:
            self.title_markdown.render()
            
            self.generate_image_app_ui()
            self.generate_video_app_ui()
    
    def generate_image_app_ui(self):
        with self.image_tab:
            gradio.Markdown("Image Segmentation", render = True)
            with gradio.Column():
                self.image_input.render()
                self.image_generate_mask_button.render()
                self.image_generate_SAM_mask_threshold.render()
                self.image_output.render()
        self.image_tab.render()
        self.image_input.change(self.__post_process_annotator_inputs, inputs = [self.image_input])
        self.image_generate_mask_button.click(self.__generate_mask, inputs = [self.image_input, self.image_generate_SAM_mask_threshold], outputs = [self.image_output])
    
    def __post_process_annotator_inputs(self, value):
        new_boxes = []
        self.__image_point_coords = []
        self.__image_point_labels = []
        self.__image_box = []
        
        b_has_box = False
        for box in value["boxes"]:
            if box['label'] == '':
                if not b_has_box:
                    new_box = box.copy()
                    new_box['color'] = (0, 0, 255)
                    new_boxes.append(new_box)
                    b_has_box = True
                self.__image_box = [
                    box['xmin'],
                    box['ymin'],
                    box['xmax'],
                    box['ymax']
                ]

                    
            elif box['label'] == '+' or box['label'] == '-':
                new_box = box.copy()
                new_box['color'] = (0, 255, 0) if box['label'] == '+' else (255, 0, 0)
                new_box['xmin'] = int((box['xmin'] + box['xmax']) / 2)
                new_box['ymin'] = int((box['ymin'] + box['ymax']) / 2)
                new_box['xmax'] = new_box['xmin']
                new_box['ymax'] = new_box['ymin']
                new_boxes.append(new_box)
                
                self.__image_point_coords.append([new_box['xmin'], new_box['ymin']])
                self.__image_point_labels.append(1 if box['label'] == '+' else 0)
            
        return {"image": value["image"], "boxes": new_boxes}          
    
    def __generate_mask(self, value, mask_threshold):
        mask_chw, mask_iou = self.segment_anything2assist.generate_masks_from_image(
            value["image"],
            self.__image_point_coords,
            self.__image_point_labels,
            self.__image_box,
            mask_threshold
        )
        
        value = [value["image"], self.segment_anything2assist.apply_mask_to_image(value["image"], mask_chw[0])]
        
        
        return value
        
        
    
    def generate_video_app_ui(self):
        with self.video_tab:
            gradio.Markdown("Video Segmentation", render = True)
        self.video_tab.render()
    
    

            
            # def image_segmentation(image):
            #     return self.segment_anything2assist.segment_image(image)
            
            # self.image_output = gradio.Output(image_segmentation, type = "image")
    
    def launch(self):
        self.base_app.launch()
        
if __name__ == "__main__":
    segment_anything2assist_ui = SegmentAnything2AssistUI()
    segment_anything2assist_ui.launch()
        
        


