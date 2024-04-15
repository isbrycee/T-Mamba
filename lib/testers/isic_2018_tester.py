# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/01/01 00:33
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch

from torchvision import transforms


class ISIC2018Tester:
    """
    Tester class
    """

    def __init__(self, opt, model, metrics=None):
        self.opt = opt
        self.model = model
        self.metrics = metrics
        self.device = self.opt["device"]

        self.statistics_dict = self.init_statistics_dict()

    def inference(self, image_path):
        test_transforms = transforms.Compose([
            transforms.Resize(self.opt["resize_shape"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
        ])

        image_pil = Image.open(image_path)
        w, h = image_pil.size
        image = test_transforms(image_pil)
        dir_path, image_name = os.path.split(image_path)
        dot_pos = image_name.find(".")
        file_name = image_name[:dot_pos]
        if self.opt["save_dir"]:
            dir_path = self.opt["save_dir"]
        segmentation_image_path = os.path.join(dir_path, file_name + "_segmentation" + ".png")

        self.model.eval()
        with torch.no_grad():
            image = torch.unsqueeze(image, dim=0)
            image = image.to(self.device)
            output = self.model(image)

        segmented_image = torch.argmax(output, dim=1).squeeze(0).to(dtype=torch.uint8).cpu().numpy()
        segmented_image = cv2.resize(segmented_image, (w, h), interpolation=cv2.INTER_AREA)
        segmented_image[segmented_image == 1] = 255

        # cv2.imwrite(segmentation_image_path, segmented_image)
        # print("Save segmented image to {}".format(segmentation_image_path))
        if self.opt["is_visual"]:
            mask = segmented_image
            print(mask.shape)
            rect = (1770, 417, 2031, 660)
            # # Create an RGBA version of the mask
            # mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            # # Set transparent red for white pixels in the mask
            # mask_rgba[mask == 255] = [255, 0, 0, 50]  # Transparent red (R,G,B,A)
            # # Create a PIL image from the RGBA mask
            # mask_pil = Image.fromarray(mask_rgba, 'RGBA')
            # # Composite the RGBA mask onto the original PIL image
            # result = Image.alpha_composite(image_pil.convert("RGBA"), mask_pil)
            # result.save(segmentation_image_path)
            # print("Save segmented image to {}".format(segmentation_image_path))

            # Create an RGBA version of the mask
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[mask == 255] = [255, 0, 0, 60]  # Transparent red (R,G,B,A)

            # Create a PIL image from the RGBA mask
            mask_pil = Image.fromarray(mask_rgba, 'RGBA')

            # Composite the RGBA mask onto the original PIL image
            result = Image.alpha_composite(image_pil.convert("RGBA"), mask_pil)
            # Convert rectangle coordinates to integers
            rect = [int(coord) for coord in rect]

            # Crop the rectangle from the original image
            cropped_image = result.crop(rect)
            
            # Resize the cropped rectangle to desired size
            new_size = (int((rect[2]-rect[0]) * 2), int((rect[3]-rect[1]) * 2))  # Increase size by 50%
            enlarged_cropped_image = cropped_image.resize(new_size)
            
            # Paste the enlarged rectangle onto the result image at the bottom right corner
            result.paste(enlarged_cropped_image, (result.width - new_size[0], result.height - new_size[1]))
            
            # Create ImageDraw object
            draw = ImageDraw.Draw(result)
            
            # Draw a rectangle around the enlarged area
            draw.rectangle([(result.width - new_size[0], result.height - new_size[1]), (result.width, result.height)], outline="blue", width=2)
            
            # Draw a rectangle around the original selected area
            draw.rectangle(rect, outline="blue", width=2)
            
            # Draw a line from the top left corner of the rectangle to the bottom right corner of the result image
            line_color = (0, 255, 0)  # Green color for the line
            line_thickness = 2
            draw.line([(rect[2], rect[3]), (result.width - new_size[0], result.height - new_size[1])], fill=line_color, width=line_thickness)
            result.save(segmentation_image_path)
            print("Save segmented image to {}".format(segmentation_image_path))

    def evaluation(self, dataloader):
        self.reset_statistics_dict()
        self.model.eval()

        with torch.no_grad():
            for input_tensor, target in tqdm(dataloader, leave=True):
                input_tensor, target = input_tensor.to(self.device), target.to(self.device)
                output = self.model(input_tensor)
                self.calculate_metric_and_update_statistcs(output.cpu(), target.cpu(), len(target))

        class_IoU = self.statistics_dict["total_area_intersect"] / self.statistics_dict["total_area_union"]
        class_IoU = np.nan_to_num(class_IoU)
        dsc = self.statistics_dict["DSC_sum"] / self.statistics_dict["count"]
        JI = self.statistics_dict["JI_sum"] / self.statistics_dict["count"]
        ACC = self.statistics_dict["ACC_sum"] / self.statistics_dict["count"]

        print("valid_DSC:{:.6f}  valid_IoU:{:.6f}  valid_ACC:{:.6f}  valid_JI:{:.6f}".format(dsc, class_IoU[1], ACC, JI))

    def calculate_metric_and_update_statistcs(self, output, target, cur_batch_size):
        mask = torch.zeros(self.opt["classes"])
        unique_index = torch.unique(target).int()
        for index in unique_index:
            mask[index] = 1
        self.statistics_dict["count"] += cur_batch_size
        for i, class_name in self.opt["index_to_class_dict"].items():
            if mask[i] == 1:
                self.statistics_dict["class_count"][class_name] += cur_batch_size
        for metric_name, metric_func in self.metrics.items():
            if metric_name == "IoU":
                area_intersect, area_union, _, _ = metric_func(output, target)
                self.statistics_dict["total_area_intersect"] += area_intersect.numpy()
                self.statistics_dict["total_area_union"] += area_union.numpy()
            elif metric_name == "ACC":
                batch_mean_ACC = metric_func(output, target)
                self.statistics_dict["ACC_sum"] += batch_mean_ACC * cur_batch_size
            elif metric_name == "JI":
                batch_mean_JI = metric_func(output, target)
                self.statistics_dict["JI_sum"] += batch_mean_JI * cur_batch_size
            elif metric_name == "DSC":
                batch_mean_DSC = metric_func(output, target)
                self.statistics_dict["DSC_sum"] += batch_mean_DSC * cur_batch_size
            else:
                per_class_metric = metric_func(output, target)
                per_class_metric = per_class_metric * mask
                self.statistics_dict[metric_name]["avg"] += (torch.sum(per_class_metric) / torch.sum(mask)).item() * cur_batch_size
                for j, class_name in self.opt["index_to_class_dict"].items():
                    self.statistics_dict[metric_name][class_name] += per_class_metric[j].item() * cur_batch_size

    def init_statistics_dict(self):
        statistics_dict = {
            metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
            for metric_name in self.opt["metric_names"]
        }
        statistics_dict["total_area_intersect"] = np.zeros((self.opt["classes"],))
        statistics_dict["total_area_union"] = np.zeros((self.opt["classes"],))
        statistics_dict["JI_sum"] = 0.0
        statistics_dict["ACC_sum"] = 0.0
        statistics_dict["DSC_sum"] = 0.0
        for metric_name in self.opt["metric_names"]:
            statistics_dict[metric_name]["avg"] = 0.0
        statistics_dict["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["count"] = 0

        return statistics_dict

    def reset_statistics_dict(self):
        self.statistics_dict["count"] = 0
        self.statistics_dict["total_area_intersect"] = np.zeros((self.opt["classes"],))
        self.statistics_dict["total_area_union"] = np.zeros((self.opt["classes"],))
        self.statistics_dict["JI_sum"] = 0.0
        self.statistics_dict["ACC_sum"] = 0.0
        self.statistics_dict["DSC_sum"] = 0.0
        for _, class_name in self.opt["index_to_class_dict"].items():
            self.statistics_dict["class_count"][class_name] = 0
        for metric_name in self.opt["metric_names"]:
            self.statistics_dict[metric_name]["avg"] = 0.0
            for _, class_name in self.opt["index_to_class_dict"].items():
                self.statistics_dict[metric_name][class_name] = 0.0

    def load(self):
        pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
        # for multigpus
        for k, v in pretrain_state_dict.items():
            if "module." in k:
                pretrain_state_dict = {key.replace('module.', ''): value for key, value in pretrain_state_dict.items()}
                break
        model_state_dict = self.model.state_dict()
        load_count = 0
        for param_name in model_state_dict.keys():
            if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                load_count += 1
        self.model.load_state_dict(model_state_dict, strict=True)
        print("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)))
