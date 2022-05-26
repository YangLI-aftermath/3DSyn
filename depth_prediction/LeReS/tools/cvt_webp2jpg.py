from PIL import Image
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='convert path')
    args = parser.parse_args()
    return args

def cvt_webp2jpg(img,path):
    image = Image.open(img).convert('RGB')
    img_jpg = img.replace('.webp','.jpg')
    image.save(os.path.join(path,img_jpg),'jpeg')

def main():
    # args = parse_args()
    # folder_path = args.path
    folder_path = '/home/yangli/3DSyn/dataset/LSUN/bedroom_train'
    imgs_list = os.listdir(folder_path)
    print(len(imgs_list))
    if os.path.exists(os.path.join(folder_path,'jpgs')):
        os.mkdir(os.path.join(folder_path,'jpgs'))
    jpg_path = os.path.join(folder_path,'jpgs')
    imgs_abs_list = [os.path.join(folder_path,img_name) for img_name in imgs_list]

    for i, p in enumerate(imgs_abs_list):
        print('processing (%08d)/(%08d)-th image... %s' % (i, len(imgs_abs_list),p))
        cvt_webp2jpg(p,jpg_path)

if __name__ == '__main__':
    main()