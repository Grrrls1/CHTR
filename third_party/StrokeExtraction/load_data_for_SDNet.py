import os.path
import torch.utils.data as data
import pickle
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

'''
Load input data for SDNet
'''


class SDNetLoader(data.Dataset):
    def __init__(self, is_training, dataset_path, is_inference=False):
        self.is_inference = is_inference
        if is_training:
            self.path = [os.path.join(dataset_path, 'train', each) for each in os.listdir(os.path.join(dataset_path, 'train'))]
        else:
            self.path = [os.path.join(dataset_path, 'test', each) for each in os.listdir(os.path.join(dataset_path, 'test'))]
        print("number of dataset：%d" % len(self.path))

    def get_data(self, item):
        """
        """
        data_frame = np.load(self.path[item])
        # RHSEDB stores binary images as bool. Normalize generated samples to
        # that native representation before collation; training converts them
        # to float32 only after moving the batch to the GPU.
        reference_color_image = data_frame['reference_color_image'].astype(np.float32)  # (3, 256, 256)
        reference_single_image = data_frame['reference_single_image'].astype(bool)  # (N, 256 256)
        reference_single_centroid = data_frame['reference_single_centroid'].astype(np.float32)
        target_image = data_frame['target_image'].astype(bool)  # (1, 256, 256)
        target_single_image = data_frame['target_single_image'].astype(bool)  # (N, 256 256)
        stroke_label = data_frame['stroke_label'].astype(np.int64)  # (N)

        stroke_num = reference_single_image.shape[0]
        pad_count = 30 - stroke_num
        if pad_count < 0:
            raise ValueError('Stroke count exceeds 30: {}'.format(stroke_num))
        expand_zeros = np.zeros((pad_count, 256, 256), dtype=bool)
        reference_single_image = np.concatenate([reference_single_image, expand_zeros], axis=0)
        target_single_image = np.concatenate([target_single_image, expand_zeros], axis=0)

        expand_single_centroid = np.full((pad_count, 2), 127.5, dtype=np.float32)
        reference_single_centroid = np.concatenate([reference_single_centroid, expand_single_centroid], axis=0)

        if not self.is_inference:
            return {
                'target_single_stroke': target_single_image,
                'reference_single_stroke': reference_single_image,
                'target_data': target_image,
                'reference_color': reference_color_image,
                'stroke_num': stroke_num,
                'reference_single_stroke_centroid': reference_single_centroid,
            }
        else:
            return {
                'target_single_stroke': target_single_image,
                'reference_single_stroke': reference_single_image,
                'target_data': target_image,
                'reference_color': reference_color_image,
                'stroke_num': stroke_num,
                'reference_single_stroke_centroid': reference_single_centroid,
                'stroke_label': stroke_label,
            }




    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        data = self.get_data(item)
        return data


def save_dict(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    c = SDNetLoader(is_training=False, dataset_path='dataset/CCSEDB')
    data_loader = data.DataLoader(c, batch_size=8, shuffle=False)

    for i in range(10):
        print(i)
        for i_batch, sample_batched in enumerate(data_loader):

            style_single_image = sample_batched['style_single_image'][0].numpy()
            kaiti_single_image = sample_batched['kaiti_single_image'][0].numpy()
            original = sample_batched['original_style'].numpy()
            kaiti_color = sample_batched['kaiti_color'].numpy()
            stroke_num = sample_batched['stroke_num'][0].numpy()
            kaiti_center = sample_batched['kaiti_center'].numpy()
            assert (not np.isnan(kaiti_center).any())
            stroke_num = int(stroke_num)
            plt.figure(0)
            for j in range(8):
                plt.subplot2grid((2, 8), (0, j))
                plt.imshow(kaiti_color[j].transpose((1, 2, 0)))
                plt.subplot2grid((2, 8), (1, j))
                plt.imshow(original[j].squeeze())
            plt.show()
            # plt.figure(1)
            #
            # for j in range(stroke_num):
            #     plt.subplot2grid((2, stroke_num), (0, j))
            #     plt.imshow(kaiti_single_image[j])
            #     plt.subplot2grid((2, stroke_num), (1, j))
            #     plt.imshow(style_single_image[j])
            #
            # plt.show()

            # seg_id = sample_batched['seg_id'][0].numpy()
            #
            # data = { 'style_single_image': style_single_image,
            # 'kaiti_single_image': kaiti_single_image,
            # 'style_image': style_image,
            # 'kaiti_image': kaiti_image,
            # 'seg_id':seg_id}
            # save_dict(data, os.path.join(path, str(train_num)+'.pkl'))
            # train_num+=1
            # print(train_num)
