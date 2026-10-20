import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import Resize
from model.model_of_SDNet import SDNet
from model.model_of_SegNet import SegNet
from model.model_of_ExtractNet import ExtractNet
from utils import seg_label_to7, seg_colors
import colorsys

'''
Introduce:
    A complete inference code that can be applied to Chinese stroke extraction.

'''



###################################################
#   SDNet输入所需要的数据处理类
###################################################

device = torch.device("cuda")

def apply_stroke(image, mask, color, t=False):
    """Apply the given mask to the image.
    """
    for c in range(3):
        if t:
            image[:, :, c] = np.where(mask[:, :] > 0.5, 0.6*color[c]+0.4*image[:, :, c] , image[:, :, c])
        else:
            image[:, :, c] = np.where(mask[:, :] > 0.5, color[c] , image[:, :, c])
    return image
def random_colors( N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors



class ExtractStroke():
    # Net
    def __init__(self, sdnet_model, segnet_model, extractnet_model,
                 inference_device):
        self.device = inference_device
        # 加载sdnet
        self.sd_net = SDNet()
        self.sd_net.load_state_dict(torch.load(sdnet_model, map_location='cpu')['net'])
        self.sd_net.eval().requires_grad_(False).to(self.device)
        print('SDNet模型加载成功')

        self.seg_net = SegNet(out_feature=True)
        state = torch.load(segnet_model, map_location='cpu')
        self.seg_net.load_state_dict(state['net'])
        self.seg_net.eval().requires_grad_(False).to(self.device)
        print('SegNet模型加载成功')

        self.extract_net = ExtractNet()
        state = torch.load(extractnet_model, map_location='cpu')
        self.extract_net.load_state_dict(state['net'])
        self.extract_net.to(self.device).eval().requires_grad_(False)
        print('ExtractNet模型加载成功')

    def __get_color_image(self, single_image, seg_label):
        color_kaiti = np.zeros(shape=(256, 256, 3))
        for i in range(single_image.shape[0]):
            color_kaiti = apply_stroke(color_kaiti, single_image[i], seg_colors[seg_label[i]])

        color_kaiti = np.transpose(color_kaiti, [2, 0, 1])
        return torch.from_numpy(color_kaiti).float().to(self.device).unsqueeze(0)


    def __calculate_linear_transformation_inference(self, reference_single_stroke,
                                                    grid_for_linear, stroke_num,
                                                    reference_single_stroke_centroid, seg_id):
        '''
        In Model Inference,
        calculate inverse local linear estimation transformation and transformed single reference stroke.
        '''
        transformed_reference_color = []
        transformed_single_reference_stroke = []
        for i in range(1):
            linear_tran_whole = []
            for j in range(int(stroke_num)):

                reference_single_stroke_ = reference_single_stroke[i][j].unsqueeze(0).unsqueeze(0)
                grid_ = grid_for_linear[i].unsqueeze(0)
                linear_grid = self.sd_net.get_linear_estimation(reference_single_stroke_, grid_,
                                                                reference_single_stroke_centroid[i][j], inverse=True)
                linear_tran = F.grid_sample(reference_single_stroke_, linear_grid)
                linear_tran_whole.append(linear_tran)
            transformed_single_reference_stroke.append(torch.cat(linear_tran_whole, dim=1).squeeze(0))

            linear_tran_whole = self.__get_color_image(
                torch.cat(linear_tran_whole, dim=1).squeeze(0).detach().to('cpu').numpy()
                , seg_id[i, :int(stroke_num)].detach().to('cpu').numpy())

            transformed_reference_color.append(linear_tran_whole)
        transformed_reference_color = torch.cat(transformed_reference_color, dim=0)

        return transformed_reference_color, transformed_single_reference_stroke

    def __get_cut_region(self, kaiti_imagae):
        if np.sum(kaiti_imagae) > 5:
            points = np.where(kaiti_imagae > 0.5)
            x_l = np.min(points[1])
            x_r = np.max(points[1])
            y_t = np.min(points[0])
            y_b = np.max(points[0])
        else:
            y_t = 0
            y_b = 255
            x_l = 0
            x_r = 255
        w = x_r - x_l + 1
        h = y_b - y_t + 1
        center_x = int((x_l+x_r)/2)
        center_y = int((y_t+y_b)/2)
        size = max(w, h)
        if size>32:
            cut_size = 256
            # y_t, y_b, x_l, x_r
            return [0, 256, 0, 256]
        elif size>16:
            cut_size = 128
        else:
            cut_size = 64

        if center_x - cut_size/2 <= 0:
            x_l = 0
            x_r = cut_size
        elif center_x+cut_size/2>=256:
            x_l = 256-cut_size
            x_r = 256
        else:
            x_l = int(center_x - cut_size/2)
            x_r = x_l + cut_size

        if center_y - cut_size/2 <= 0:
            y_t = 0
            y_b = cut_size
        elif center_y+cut_size/2>=256:
            y_t = 256-cut_size
            y_b = 256
        else:
            y_t = int(center_y - cut_size/2)
            y_b = y_t + cut_size

        return [y_t, y_b, x_l, x_r]

    def __create_color_image(self, image, id):
        image = image.repeat(1, 3, 1, 1)
        color = torch.from_numpy(np.array(seg_colors[id]).reshape(1, 3, 1, 1)).to(self.device)
        return image * color

    def __get_training_data_of_ExtarctNet(self, reference_transformed_single, seg_index,
                                          seg_out, reference_segment_transformation_data,
                                          target_data, seg_out_feature):
        '''
        get training_data of ExtractNet
        When the size of reference stroke is too small, the stroke area is clipped and enlarged
        to increase the discrimination ability.
        '''

        kaiti_tran_single_image = torch.reshape(reference_transformed_single, shape=(-1, 1, 256, 256))
        kaiti_trans_stage2_in = []
        kaiti_trans_seg_stage2_in = []
        seg_out_stage_in = []
        style_original_stage2_in = []
        cut_box_list = []
        for index in range(int(seg_index.size(0))):
            id = int(seg_index[index])
            id_7 = seg_label_to7(id)
            kaiti_single_image_trans = kaiti_tran_single_image[index:index + 1]

            # get cut region：
            kaiti_image__ = np.squeeze(kaiti_single_image_trans.to('cpu').numpy())
            cut_box = self.__get_cut_region(kaiti_image__)
            torch_resize = Resize([256, 256])
            y_t, y_b, x_l, x_r = cut_box
            cut_box_list.append(cut_box)

            # Reference Stroke Transformation Data
            kaiti_tran_in = kaiti_single_image_trans.detach()
            kaiti_tran_in = torch_resize(kaiti_tran_in[:, :, y_t:y_b, x_l:x_r])
            kaiti_tran_in = self.__create_color_image(kaiti_tran_in, id).float()
            kaiti_trans_stage2_in.append(kaiti_tran_in)


            # Segment Data
            seg_out_in = F.sigmoid(seg_out[:, id_7:id_7 + 1]).detach().float()
            seg_out_in = torch_resize(seg_out_in[:, :, y_t:y_b, x_l:x_r])
            seg_out_stage_in.append(seg_out_in)

            # Reference Segment Transformation Data
            kaiti_seg_in = reference_segment_transformation_data[:, id_7:id_7 + 1].detach().float()
            kaiti_seg_in = torch_resize(kaiti_seg_in[:, :, y_t:y_b, x_l:x_r])
            kaiti_trans_seg_stage2_in.append(kaiti_seg_in)

            # Target Data
            style_image_original_in = torch_resize(target_data[:, :, y_t:y_b, x_l:x_r])
            style_original_stage2_in.append(style_image_original_in)
        reference_stroke_transformation_data = torch.cat(kaiti_trans_stage2_in, dim=0)

        segment_data = torch.cat(seg_out_stage_in, dim=0)
        segNet_feature = seg_out_feature['out_64_32'].repeat(segment_data.size(0), 1, 1, 1)
        target_data = torch.cat(style_original_stage2_in, dim=0)
        reference_segment_transformation_data = torch.cat(kaiti_trans_seg_stage2_in, dim=0)

        return [target_data, reference_stroke_transformation_data, segment_data,
                reference_segment_transformation_data, segNet_feature, cut_box_list]

    def get_seg_image(self, reference_single, seg_label):
        reference_image = np.zeros(shape=(7, 256, 256), dtype=np.float32)
        for i in range(seg_label.shape[0]):
            id_7 = seg_label_to7(seg_label[i])
            reference_image[id_7] += reference_single[i]
        return np.clip(reference_image, 0, 1)

    @torch.no_grad()
    def get_extract_strokes(self, input_data):

        reference_single_stroke = torch.from_numpy(input_data['reference_single_image']).unsqueeze(0).to(self.device).float()
        target_data = torch.from_numpy(input_data['target_image']).unsqueeze(0).to(self.device).float()
        reference_color = torch.from_numpy(input_data['reference_color_image']).unsqueeze(0).to(self.device).float()
        stroke_num = input_data['stroke_label'].shape[0]
        reference_single_stroke_centroid = torch.from_numpy(input_data['reference_single_centroid']).unsqueeze(0).to(self.device).float()
        seg_id = torch.from_numpy(input_data['stroke_label']).unsqueeze(0).to(self.device).long()


        transformed_target_data, flow_global, grid_for_linear = self.sd_net.get_two_registration_field(reference_color,
                                                                                                       target_data)

        transformed_reference_color, transformed_single_reference_stroke = self.__calculate_linear_transformation_inference(
            reference_single_stroke,
            grid_for_linear, stroke_num,
            reference_single_stroke_centroid, seg_id)

        # 构建数据 for SegNet和ExtractNet
        style_original = target_data.repeat([1,3,1,1])
        reference_segment_transformation_data = self.get_seg_image(transformed_single_reference_stroke[0].detach().to('cpu').numpy(), seg_id[0][:stroke_num].detach().to('cpu').numpy())
        reference_segment_transformation_data = torch.from_numpy(reference_segment_transformation_data).unsqueeze(0).float().to(self.device)
        segnet_reference_prior = reference_segment_transformation_data
        # SegNet result
        seg_out, seg_out_feature = self.seg_net(style_original, transformed_reference_color)

        # get inputs of ExtractNet
        target_data, reference_stroke_transformation_data, segment_data, \
            reference_segment_transformation_data, segNet_feature, cut_box_list = self.__get_training_data_of_ExtarctNet(
            transformed_single_reference_stroke[0], seg_id[0][:stroke_num],
            seg_out, reference_segment_transformation_data,
            style_original, seg_out_feature)

        extract_out = self.extract_net(reference_stroke_transformation_data,
                                       reference_segment_transformation_data, segment_data,
                                       target_data, segNet_feature)
        extract_probabilities = torch.sigmoid(extract_out).detach()
        extract_result = extract_probabilities > 0.5
        extract_result = self.__to_original_stroke(extract_result, cut_box_list)
        extract_probabilities_original = self.__to_original_probability(
            extract_probabilities, cut_box_list)
        return {
            'sdnet_transformed_target': transformed_target_data[0].detach().cpu().numpy(),
            'sdnet_flow_global': flow_global[0].detach().cpu().numpy(),
            'sdnet_transformed_reference_color': transformed_reference_color[0].detach().cpu().numpy(),
            'sdnet_transformed_single_strokes': transformed_single_reference_stroke[0].detach().cpu().numpy(),
            'segnet_probabilities': torch.sigmoid(seg_out[0]).detach().cpu().numpy(),
            'segnet_reference_prior': segnet_reference_prior[0].detach().cpu().numpy(),
            'extractnet_strokes': np.asarray(extract_result, dtype=np.uint8),
            'extractnet_probabilities': extract_probabilities_original,
            'stroke_labels': np.asarray(input_data['stroke_label'], dtype=np.int64),
        }

    def __to_original_probability(self, probabilities, cut_box_list):
        """Restore local ExtractNet probabilities to the 256x256 target canvas."""
        values = probabilities.squeeze(1).to('cpu').numpy().astype(np.float32)
        restored = []
        for index, (y_t, y_b, x_l, x_r) in enumerate(cut_box_list):
            canvas = np.zeros((256, 256), dtype=np.float32)
            canvas[y_t:y_b, x_l:x_r] = cv2.resize(
                values[index], dsize=(x_r - x_l, y_b - y_t),
                interpolation=cv2.INTER_LINEAR)
            restored.append(canvas)
        return np.asarray(restored, dtype=np.float32)

    def __to_original_stroke(self, out, cut_box_list):
        '''
        Restore strokes to their original size
        :param cut_box_list: Adaptive size parameters
        :return:
        '''
        out = out.squeeze(1).to('cpu').numpy()
        out_re = []
        for i in range(len(cut_box_list)):
            y_t, y_b, x_l, x_r = cut_box_list[i]
            img = np.zeros_like(out[i])
            img[y_t:y_b, x_l:x_r] = cv2.resize(out[i].astype(np.uint8), dsize=(x_r - x_l, y_b - y_t))
            out_re.append(img)
        return out_re

    def get_reference_data(self, stroke_single_images, stroke_labels):
        # reference color
        color_kaiti = np.zeros(shape=(256, 256, 3))
        for i in range(stroke_labels.shape[0]):
            color_kaiti = apply_stroke(color_kaiti, stroke_single_images[i], seg_colors[stroke_labels[i]])
        color_kaiti = np.transpose(color_kaiti, [2, 0, 1])

        # reference single stroke centroid
        reference_single_centroid = []
        for i in range(30):
            if i >= stroke_labels.shape[0]:
                reference_single_centroid.append(np.array([127.5, 127.5]))
            else:
                point = np.where(stroke_single_images[i] > 0.5)
                center = np.array([np.mean(point[1]), np.mean(point[0])])
                reference_single_centroid.append(center)

        reference_single_centroid = np.array(reference_single_centroid)
        return color_kaiti, reference_single_centroid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract strokes from one RHSEDB NPZ sample.')
    parser.add_argument('--input', required=True, help='Input RHSEDB .npz file')
    parser.add_argument('--results-dir', default='results', help='Root output directory')
    parser.add_argument('--sdnet-model', default='out/SDNet_RHSEDB/model/sdnet_model.pth')
    parser.add_argument('--segnet-model', default='out/SegNet_RHSEDB/model/model.pth')
    parser.add_argument('--extractnet-model', default='out/ExtractNet_RHSEDB/model/model_extract.pth')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the current SDNet implementation.')
    device = torch.device('cuda', torch.cuda.current_device())
    print('使用设备: {} ({})'.format(device, torch.cuda.get_device_name(device)))
    model = ExtractStroke(args.sdnet_model, args.segnet_model,
                          args.extractnet_model, device)

    # 所有输入的值都是ndarray格式，被归一化到（0， 1）之间，参数含义参见dataset的介绍
    data = np.load(args.input)
    print('字符:', data['name'])

    # 需要提供 目标字图像、 参考字单笔画图像和笔画label
    # target_image,reference_single_image,stroke_label
    reference_color_image, reference_single_centroid = model.get_reference_data(data['reference_single_image'], data['stroke_label'])
    input_data= {
        'target_image': data['target_image'],  # (1, 256, 256)，待提取的目标字
        'reference_single_image': data['reference_single_image'],  # (N, 256, 256)
        'reference_color_image': reference_color_image, # (3, 256, 256)
        'reference_single_centroid': reference_single_centroid, # (N, 2)
        'stroke_label': data['stroke_label'], # (N)
    }
    stage_results = model.get_extract_strokes(input_data)
    extract_result = stage_results['extractnet_strokes']
    result_dir = Path(args.results_dir) / Path(args.input).stem
    result_dir.mkdir(parents=True, exist_ok=True)

    # Machine-readable outputs for downstream processing.
    np.save(result_dir / 'sdnet_transformed_target.npy', stage_results['sdnet_transformed_target'])
    np.save(result_dir / 'sdnet_flow_global.npy', stage_results['sdnet_flow_global'])
    np.save(result_dir / 'sdnet_transformed_reference_color.npy', stage_results['sdnet_transformed_reference_color'])
    np.save(result_dir / 'sdnet_transformed_single_strokes.npy', stage_results['sdnet_transformed_single_strokes'])
    np.save(result_dir / 'segnet_probabilities.npy', stage_results['segnet_probabilities'])
    np.save(result_dir / 'segnet_reference_prior.npy', stage_results['segnet_reference_prior'])
    np.save(result_dir / 'extractnet_strokes.npy', extract_result)
    np.savez_compressed(result_dir / 'all_stages.npz', **stage_results)
    metadata = {
        'input': str(Path(args.input).resolve()),
        'character': str(data['name']),
        'fields': {
            name: {'shape': list(value.shape), 'dtype': str(value.dtype)}
            for name, value in stage_results.items()
        },
    }
    with (result_dir / 'metadata.json').open('w', encoding='utf-8') as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    # Human-readable previews for each stage.
    plt.imsave(result_dir / 'input_target.png', input_data['target_image'][0], cmap='gray', vmin=0, vmax=1)
    plt.imsave(result_dir / 'input_reference_color.png', input_data['reference_color_image'].transpose(1, 2, 0))
    plt.imsave(result_dir / 'sdnet_transformed_target.png', stage_results['sdnet_transformed_target'][0], cmap='gray', vmin=0, vmax=1)
    plt.imsave(result_dir / 'sdnet_transformed_reference_color.png', np.clip(stage_results['sdnet_transformed_reference_color'].transpose(1, 2, 0), 0, 1))

    sdnet_strokes_preview = np.zeros((256, 256, 3), dtype=np.float32)
    for stroke, color in zip(stage_results['sdnet_transformed_single_strokes'],
                             random_colors(len(stage_results['sdnet_transformed_single_strokes']))):
        sdnet_strokes_preview = apply_stroke(sdnet_strokes_preview, stroke > 0.5, color)
    plt.imsave(result_dir / 'sdnet_transformed_single_strokes.png', sdnet_strokes_preview)

    seg_preview = np.zeros((256, 256, 3), dtype=np.float32)
    for i, color in enumerate(random_colors(7)):
        seg_preview = apply_stroke(seg_preview, stage_results['segnet_probabilities'][i] > 0.5, color)
    plt.imsave(result_dir / 'segnet_segmentation.png', seg_preview)

    # Final overlay preview.
    extract_result_show = np.zeros(shape=(256, 256, 3), dtype=np.float32) + input_data['target_image'].transpose(
        (1, 2, 0))
    r_colors = random_colors(len(extract_result))
    for i in range(len(extract_result)):
        extract_result_show = apply_stroke(extract_result_show, extract_result[i] > 0.5, r_colors[i], t=True)
    stroke_dir = result_dir / 'extractnet_strokes'
    stroke_dir.mkdir(exist_ok=True)
    for i, stroke in enumerate(extract_result):
        plt.imsave(stroke_dir / 'stroke_{:02d}.png'.format(i), stroke,
                   cmap='gray', vmin=0, vmax=1)
    plt.figure(figsize=(6, 6))
    plt.imshow(extract_result_show)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(result_dir / 'extractnet_overlay.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print('三阶段结果已保存:', result_dir)
