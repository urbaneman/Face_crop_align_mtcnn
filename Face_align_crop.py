import os
import sys
import mxnet as mx
from tqdm import tqdm
import argparse
import cv2
from align_mtcnn.mtcnn_detector import MtcnnDetector

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='F:\images_5', help='the dir your dataset of face which need to crop')
    parser.add_argument('--output_path', type=str, default='F:\images_5_face', help='the dir the cropped faces of your dataset where to save')
    # parser.add_argument('--face-num', '-face_num', type=int, default=1, help='the max faces to crop in each image')
    parser.add_argument('--gpu', default=-1, type=int, help='gpu id， when the id == -1, use cpu')
    parser.add_argument('--face_size', type=str, default='224', help='the size of the face to save, the size x%2==0, and width equal height')
    args = parser.parse_args()
    return args

def crop_align_face(args):
    input_dir = args.input_path
    output_dir = args.output_path
    # face_num = args.face_num
    if not os.path.exists(input_dir):
        print('the input path is not exists!')
        sys.exit()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    mtcnn_path = os.path.join(os.path.dirname(__file__), 'align_mtcnn/mtcnn-model')

    mtcnn = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark=True)

    count_no_find_face = 0
    count_crop_images = 0

    for root, dirs, files in tqdm(os.walk(input_dir)):
        # print(root)
        output_root = root.replace(input_dir, output_dir)
        if not os.path.exists(output_root):
            os.mkdir(output_root)
        for file_name in files:
            '''
            the specific request of the datasets in file name
            if you not need, please comment out
            '''
            # not crop the file end with bmp
            if file_name.split('.')[-1] == 'bmp':
                continue
            file_path = os.path.join(root, file_name)
            face_img = cv2.imread(file_path)
            ret = mtcnn.detect_face(face_img)
            if ret is None:
                print('%s do not find face'%file_path)
                count_no_find_face += 1
                continue
            bbox, points = ret
            if bbox.shape[0] == 0:
                print('%s do not find face'%file_path)
                count_no_find_face += 1
                continue
            # print(bbox, points)
            for i in range(bbox.shape[0]):
                bbox_ = bbox[i, 0:4]
                points_ = points[i, :].reshape((2, 5)).T
                face = mtcnn.preprocess(face_img, bbox_, points_, image_size=args.face_size)
                face_name = '%s_%d.jpg'%(file_name.split('.')[0], i)
                file_path_save = os.path.join(output_root, face_name)
                cv2.imwrite(file_path_save, face)
                # cv2.imshow('face', face)
                # cv2.waitKey(0)
            count_crop_images += 1
    print('%d images crop successful!' % count_crop_images)
    print('%d images do not crop successful!' % count_no_find_face)

if __name__ == '__main__':
    args = getArgs()
    crop_align_face(args)
