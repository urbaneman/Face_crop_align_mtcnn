# Face_crop_align_mtcnn
Crop and Align Face by MTCNN

## Environment

1. The test environment is
    - Python 3.6.4
    - mxnet 1.2.0
    - numpy 1.15.0
    - scikit-image 0.14.0 
    - python-opencv 3.4.3
    - tqdm 4.28.1

## Reference

   some part of the content reference from [insightface](https://github.com/deepinsight/insightface)
   
## How to use it

   Just specify input path, output path, output face's size with '--input_path', '--output_path' and '--face_size', if you want use gpu, please specify gpu id with '--gpu'. 
   
   usage: Face_align_crop.py [-h] [--input_path INPUT_PATH]
                          [--output_path OUTPUT_PATH] [--face-num FACE_NUM]
                          [--gpu GPU] [--face_size FACE_SIZE]
                          
   Run python Face_align_crop.py --input_path ... --output_path ... --gpu ... --face_size ...
