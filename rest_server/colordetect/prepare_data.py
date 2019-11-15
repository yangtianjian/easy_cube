import cv2
import numpy as np
import pandas as pd
import os
import multiprocessing as mp
from multiprocessing import Pool
import argparse

v_ext_names = ['mov', 'mp4', 'avi']
img_ext_names = ['jpg', 'jpeg', 'png']


def _list_file_with_ext(dir_name: str, exts: list) -> list:
    '''
    List all visible files whose extension matches one in 'exts' regardless of capitalization
    :param dir_name: The path of directory to be listed.
    :param exts: The candidate extensions
    :return: A list of filenames [without path(just names)].
    '''
    return list(filter(lambda x: ('.' in x) and (x.split('.')[-1].lower() in exts
                                                 and not x.startswith('.')), os.listdir(dir_name)))


def _read_images(video_file_name):
    '''
    Turn one video into frame picture arrays.
    :param video_file_name: The path and filename of the video with suffix in f_ext_names
    :return: 4-D array, with shape [F, H, W, C] F: Total frames. H: The height. W: The width.
    C: Three channels of frames. In BGR format.
    '''
    video = cv2.VideoCapture(video_file_name)
    frames = []
    while True:
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
    return np.stack(frames, axis=0)


def _clip_video_into_frames(v_dir, v_name, frm_dir):
    '''
    Transform one video with name v_name in the directory v_dir
    :param v_dir: The directory where the video locates.
    :param v_name: The filename of the video.
    :param frm_dir: The location where these frame outputs.
    :return: None
    '''
    v_path = os.path.join(v_dir, v_name)
    pic_path = os.path.join(frm_dir, v_name)
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)
    frames = _read_images(v_path)
    frames_count = frames.shape[0]
    for i in np.arange(frames_count):
        frame = frames[i]
        cv2.imwrite(os.path.join(pic_path, str(i) + '.png'), frame)


# Step 1 for data preprocessing. Video => Original Frame
def clip_all_videos_in_dir(v_dir, out_dir, n_workers=mp.cpu_count()):
    '''
    Clip all video files in one directory.
    :param v_dir: The path of the directory of videos
    :param out_dir: The path where the frames will be in
    :param n_workers: How many processes are used to do the clipping. Default multiprocessing.cpu_count().
    If n_workers=1, we won't use multiprocessing
    :return: None
    '''

    video_names = _list_file_with_ext(v_dir, v_ext_names)
    if n_workers > 1:
        pool = Pool(n_workers)
        for v_name in video_names:
            pool.apply_async(_clip_video_into_frames, args=(v_dir, v_name, out_dir))
        pool.close()
        pool.join()
    else:
        for v_name in video_names:
            _clip_video_into_frames(v_dir, v_name, out_dir)


def _clip_one_picture(picture_file, ux, uy, dx, dy, output_file):
    '''
    Clip one picture with upper left corner(ux, uy) and down right corner (dx, dy). Then output file.
    :param picture_file: The input picture filename.
    :param ux: Upper left x
    :param uy: Upper left y
    :param dx: Down right x
    :param dy: Down right y
    :param output_file: The file where the clipped picture to be output.
    :return: None
    '''
    img = cv2.imread(picture_file)
    img = img[ux: dx + 1, uy: dy + 1, :]
    cv2.imwrite(output_file, img)


def _clip_multiple_pictures(input_file_names, ux, uy, dx, dy, output_dir_name, start_number):
    '''
    Clip a list of picture with the same uxs, uys, dxs, dys. Then output files in a directory, starting from filename
    labeling from the start number.
    :param input_file_names: The input filenames of pictures.
    :param ux: Upper left x
    :param uy: Upper left y
    :param dx: Down right x
    :param dy: Down right y
    :param output_dir_name: The directory where the clipped picture to be output.
    :param start_number: The start number of those pictures
    :return:
    '''
    idx = start_number
    for img_file_name in input_file_names:
        img_out_name0 = os.path.join(output_dir_name, str(idx)) + '.png'
        _clip_one_picture(img_file_name, ux, uy, dx, dy, img_out_name0)
        idx += 1


# Step 2 for data preprocessing. Frame => Clipped frame
def clip_frames_into_cubes(v_label_file, v_frame_dir, frame_cube_dir, frame_label_file, n_workers=mp.cpu_count()):
    '''
    Input the video label file and the directory of frames to be output, we clip the pictures.
    :param v_label_file: The video label file.
    :param v_frame_dir:  The directory for the input frames.
    :param frame_cube_dir: The directory for the output clipped frames.
    :param frame_label_file: The information of the frames will be stored in this file.
    :param n_workers: Total threads.
    :return: None
    '''

    video_dir_names = _list_file_with_ext(v_frame_dir,
                                          v_ext_names)  # Find directory that is named the same as the original video

    tb_video = pd.read_csv(v_label_file, sep='\t', index_col='id', dtype={'id': int})

    tb_picture = []

    dict_video_file_name_to_info = dict()
    columns = ['camera', 'light_color', 'cube_id', 'cube_color', 'file_name', 'ux', 'uy', 'dx', 'dy']
    ids = list(tb_video.index)
    for i in range(len(tb_video)):
        file_name = tb_video.loc[ids[i], 'file_name']
        dict_video_file_name_to_info[file_name] = {}
        for c in columns:
            dict_video_file_name_to_info[file_name][c] = tb_video.loc[ids[i], c]

    MULTITHREAD = (n_workers > 1)
    pool = None
    if MULTITHREAD:
        pool = Pool(n_workers)

    i = 0
    for video_dir_name in video_dir_names:
        video_dir_name0 = os.path.join(v_frame_dir, video_dir_name)
        image_file_names = _list_file_with_ext(video_dir_name0, img_ext_names)
        input_file_names = [os.path.join(video_dir_name0, name) for name in image_file_names]
        endi = i + len(image_file_names)

        if video_dir_name not in dict_video_file_name_to_info:  # some videos in video_frames directory may not have been labeled
            continue
        info = dict_video_file_name_to_info[video_dir_name]
        if MULTITHREAD:
            pool.apply_async(_clip_multiple_pictures, args=(input_file_names, info['ux'], info['uy'],
                                                            info['dx'], info['dy'], frame_cube_dir, i))
        else:
            _clip_multiple_pictures(input_file_names, info['ux'], info['uy'], info['dx'], info['dy'],
                                    frame_cube_dir, i)
        for j in np.arange(i, endi):
            pic_info = info.copy()
            pic_info.pop('file_name')
            pic_info.pop('ux')
            pic_info.pop('uy')
            pic_info.pop('dx')
            pic_info.pop('dy')
            pic_info['video_file_name'] = info['file_name']
            pic_info['picture_file_name'] = str(j) + '.png'
            pic_info['id'] = j
            tb_picture.append(pic_info)
        i = endi

    tb_picture = pd.DataFrame.from_records(tb_picture, index=['id'])

    if MULTITHREAD:
        pool.close()
        pool.join()
    tb_picture.to_csv(frame_label_file, sep='\t')


def _cut3x3(picture_file, output_directory, start_number):
    img = cv2.imread(picture_file)
    h, w, c = img.shape
    h0 = h // 3
    w0 = w // 3
    k = start_number
    for i in range(3):
        for j in range(3):
            img2 = img[i * h0: (i + 1) * h0, j * w0: (j + 1) * w0, :]
            cv2.imwrite(os.path.join(output_directory, str(k)) + '.png', img2)
            k += 1


# Step 3 for data preprocessing. Clipped frame => block
def separate_blocks(frame_cube_dir, frame_label_file, block_cube_dir, block_label_file):
    '''
    Read clipped frames. Clip blocks and align the dataset.
    :param frame_cube_dir:  The directory of clipped cubes.
    :param frame_label_file: The label file describing clipped cubes.
    :param block_cube_dir:  The output directory of blocks of cubes.
    :param block_label_file: The dataset of the blocks.
    :return: None
    '''
    label_pictures = pd.read_csv(frame_label_file, sep='\t', index_col='id')

    label_blocks = []
    columns = label_pictures
    ids = label_pictures.index

    pool = Pool(8)

    position_strings = ['UL', 'UM', 'UR', 'ML', 'MM', 'MR', 'DL', 'DM', 'DR']

    k = 0
    MULTITHREAD = True
    for i in range(len(label_pictures)):
        file_name = label_pictures.loc[ids[i], 'picture_file_name']
        picture_file = os.path.join(frame_cube_dir, file_name)
        if MULTITHREAD:
            pool.apply_async(_cut3x3, args=(picture_file, block_cube_dir, k))
        else:
            _cut3x3(picture_file, block_cube_dir, k)

        record = label_pictures.loc[ids[i]].to_dict()
        for j in range(9):
            record2 = record.copy()
            record2['block_file_name'] = str(k + j) + '.png'
            record2['block_position'] = position_strings[j]
            record2['id'] = k + j
            label_blocks.append(record2)
        k += 9

    label_blocks = pd.DataFrame.from_records(label_blocks, index=['id'])
    label_blocks.to_csv(block_label_file, sep='\t')
    pool.close()
    pool.join()


def make_dataset(clipped_block_label_file, clipped_block_dir, preshuffle=True, preprocess_fn=None, limit=999999999):
    '''
    Request dataset from block label file and block images
    :param clipped_block_label_file: The block label file
    :param clipped_block_dir: The block image directory
    :param preprocess_fn: A function f(img), which returns a processed image
    :param limit: Using for debug on small dataset.
    :param preshuffle: Shuffle the dataset when making it. For small data, it avoids sampling from the same category
    :return: The dataset X and y. They are all lists. Remember: The images in X may not in the same width or height
    '''
    table = pd.read_csv(clipped_block_label_file, sep='\t')
    ids = table.index

    N = min(len(table), limit)

    if preshuffle:
        perm = np.random.permutation(len(table))[: N]
        ids = ids[perm]
    X = []
    y = []
    for i in range(N):
        line = table.loc[ids[i]]
        label = {
            "color": line['cube_color']
        }
        block_file_name = os.path.join(clipped_block_dir, line['block_file_name'])
        img = cv2.imread(block_file_name)
        if preprocess_fn is not None:
            img = preprocess_fn(img)
        X.append(img)
        y.append(label)
    return X, y


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-vd", "--video_dir", default='video')
    ap.add_argument("-fd", "--frame_dir", default='video_frames')
    ap.add_argument("-cfd", "--clipped_frame_dir", default='clipped_video_frames')
    ap.add_argument("-cbd", "--clipped_block_dir", default='clipped_color_blocks')
    ap.add_argument('-vl', "--video_label", default='./label/label_video.tsv')
    ap.add_argument('-cfl', "--clipped_frame_label", default='./label/label_picture.tsv')
    ap.add_argument("-bl", "--block_label", default='./label/label_block.tsv')
    args = vars(ap.parse_args())

    print('Clipping videos')
    clip_all_videos_in_dir(args['video_dir'], args['frame_dir'])

    print('Clipping cubes')
    clip_frames_into_cubes(args['video_label'], args['frame_dir'], args['clipped_frame_dir'], args['clipped_frame_label'])

    print('Separating blocks')
    separate_blocks(args['clipped_frame_dir'], args['clipped_frame_label'], args['clipped_block_dir'], args['block_label'])

    print('Finish')
