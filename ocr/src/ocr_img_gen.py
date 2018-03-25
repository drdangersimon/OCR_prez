"""
Generators to generate images given text

"""
from __future__ import division, print_function
import os

# check if running in headless mode
if 'DISPLAY' not in os.environ:
    import matplotlib as mlp

    mlp.use('Agg')
import cairocffi as cairo
import numpy as np
import pylab as plt
from scipy import ndimage
from scipy.ndimage.interpolation import rotate
from skimage import draw
from skimage import io as skio
from skimage.color import rgb2grey
from sklearn.preprocessing import scale
from skimage import transform
from cv2 import addWeighted


def speckle(img_shape, max_severity=1):
    """
    Put speckle noise on image
    :param img_shape: ndarray
    :param max_severity: float, 
    :return: ndarray spekles
    """
    # choose how much speckle to have on image
    severity = np.random.rand() * max_severity
    img_speck = ndimage.gaussian_filter(np.random.randn(*img_shape) * severity, 1)
    # binarize speckle
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


def make_background(img_shape, mean, std):
    """
    Makes background noise for image
    :param img_shape: tupple ints
    :param mean: float
    :param std: float
    :return: 
    """
    img_back = np.random.randn(*img_shape) * std + mean
    return img_back


def blur_img(img, kernel_size=3):
    """
    Gaussian blur on images by some kernel size
    :param img: ndarray 
    :param kernel_size: int
    :return: ndarray
    """
    blurred_img = ndimage.gaussian_filter(img, sigma=kernel_size)
    return blurred_img


def make_wavey_backgroud(img_shape, fig, waves_x=11, waves_y=16):
    """
    Uses bezier curves to draw wavey lines
    :param img: 
    :param freq: 
    :param amp: 
    :param space: 
    :return: 
    """
    fig.clf()
    ax = fig.add_subplot(111)
    # turn off extra plotting stuff
    ax.margins(0, 0)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    ax.axis('off')
    # make function
    x = np.arange(img_shape[1])
    y = .07 * np.sin(2 * np.pi * (waves_x / float(img_shape[1])) * x) * (float(img_shape[0]) / waves_y) + .2
    # plot curve on y axis and trim
    for offset in range(0, waves_y + 6):
        ax.plot(x, y + offset, 'k', linewidth=1)
    # do trim
    ax.set_ylim((3, offset - 3))
    # save as buffer to get z valeus
    temp_buff, temp_img_dim = fig.canvas.print_to_buffer()
    z = np.fromstring(temp_buff, dtype=np.uint8).reshape((temp_img_dim[0], temp_img_dim[1], 4))
    z = rgb2grey(z)
    # clear figure
    fig.clf()
    return z


def make_text(text, img_shape, font_size, txt_points=None, font='cmbtt10', bold=False):
    """
    Renders text onto an image. 
    :param text: str text to put on image
    :param img_shape: tupple of image shape (w,h)
    :param font_size: int
    :param txt_points: tupple use if want to specify placement of text. Else, is random
    :param font: str 
    :param bold: bool if bold text or no
    :return: ndarray
    """
    h, w = img_shape
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    with cairo.Context(surface) as context:
        # Set background White
        context.set_source_rgb(1, 1, 1)
        context.paint()
        # set font properties
        if bold:
            context.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        else:
            context.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        context.set_font_size(font_size)
        # see if text is in box
        box = context.text_extents(text)
        border_w_h = (4, 4)
        img_in_box = (box[2] > (w - 2 * border_w_h[0]) or box[3] > (h - 2 * border_w_h[1])) == False
        # shift y into img or random
        max_shift_x = w - border_w_h[1] - box[2]
        max_shift_y = h - box[3] - border_w_h[0]
        top_left_x = np.random.randint(border_w_h[1], abs(int(max_shift_x)))
        top_left_y = np.random.randint(border_w_h[0], abs(int(max_shift_y)))
        if txt_points is None:
            context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
        else:
            context.move_to(txt_points[1], txt_points[0])

        context.set_source_rgb(0, 0, 0)
        context.show_text(text)

    buf = surface.get_data()
    out_img = np.frombuffer(buf, np.uint8)
    out_img.shape = (h, w, 4)
    # make image black and white
    out_img = out_img[:, :, 0]
    out_img = out_img.astype(np.float32) / 255
    # out_img = np.expand_dims(out_img, 0)
    return out_img, img_in_box


def make_box(text, img_shape, left_bottom, right_top):
    """
    Makes a box and some text in the top left corner
    :param text: str
    :param img_shape: tupple 
    :param left_bottom: tupple left corner of box
    :param right_top: tupple right top corner of box
    :return: ndarray
    """
    # calculated other coords
    left_top = (left_bottom[0], right_top[1])
    right_bottom = (right_top[0], left_bottom[1])
    row, col = zip(left_bottom, left_top, right_top, right_bottom)
    rr, cc = draw.polygon_perimeter(row, col, shape=img_shape, clip=True)
    box_img = np.ones(img_shape)
    box_img[rr, cc] = 0
    # add text near top left
    text_pos = (left_bottom[0] + 16, left_bottom[1] + 2)
    box_text, _ = make_text(text, img_shape, 18, txt_points=text_pos, font='Century Schoolbook', bold=True)
    return (box_img + box_text)


def insert_forward_slash(t, random_space=False):
    """
    Inserts a "/" into the 9 digit id number to make it look like a zambian ncr.
    "000000001" -> "000000/00/1"
    :param t: str
    :param random_space: bool if should put a random space in text
    :return: str
    """
    assert len(t) == 9, 'NRC must be of length 9'
    assert t.isdigit(), 'NRC is all digits'
    tmp_txt = list(t)
    tmp_txt.insert(-1, '/')
    tmp_txt.insert(-4, '/')
    if random_space:
        r_space_index = np.random.choice(len(tmp_txt))
        tmp_txt.insert(r_space_index, ' ')
    return ''.join(tmp_txt)


def unitscale(img):
    """
    Scale img so between (0,1)
    :param img: ndarray
    :return: ndarray
    """
    maxx = img.max()
    minn = img.min()
    return (img - minn) / (maxx - minn)


def make_nrc_text_box(w_real, h_real, text, fig):
    """
    Makes an NRC looking box for zambia
    :param w: width of image
    :param h: height of image
    :param text: str nrc to put
    :return: ndarray
    """
    # add extra space so can cut rotate later
    w = w_real * 2
    h = h_real * 2
    h_center, w_center = (h // 2, w // 2)
    # size of box
    box_w = 50
    box_h = 400
    # make background
    mu_back = np.random.rand()
    std_back = np.random.rand() * .02
    # back_img = make_background((h, w), mu_back, std_back)
    # make wavey lines
    n_waves = np.random.randint(10, 15)  # for double size images
    n_periods = np.random.randint(5, 10)
    wavey_img = make_wavey_backgroud((w, h), fig, n_periods, n_waves)
    # make text
    font_size = np.random.randint(15, 30)  # half size for double size images
    # move text and box in relation to center of image -30 from center+ (np.random.rand()-1) * 30,
    text_pos = (np.random.randint(65, 85), np.random.randint(h_center - h_center // 3, h_center + 40))
    nrc_text, text_in_img = make_text(text, (w, h), font_size, text_pos, bold=False)
    # make nrc like box with relation to center. keep aspect ratio
    left_bottom = np.array([w_center + (box_w // 2), h_center + (box_h // 2)])
    right_top = np.array([w_center - (box_w // 2), h_center - (box_h // 2)])
    box_img = make_box('Registration Number', (w, h), right_top, left_bottom)
    # scale 0,1
    box_img = np.round((box_img - box_img.min()) / (box_img.max() - box_img.min()))
    # rotate text
    text_angle = np.random.randint(-4, 4)
    nrc_text = np.round(rotate(nrc_text, text_angle, reshape=False, cval=1))
    # box and wavey background get ad same angle.
    background_angle = np.random.randint(-3, 3)
    wavey_rot = np.round(rotate(wavey_img, background_angle, reshape=False, cval=1))
    box_rot = np.round(rotate(box_img, background_angle, reshape=False, cval=1))
    # combine make wavey part in the background
    wavey_rot[box_rot != 1] = 0
    wavey_rot[nrc_text == 0] = 0
    # scale images
    wave_scale, box_scale, nrc_scale = np.random.randn(3)
    out_img = np.zeros_like(box_rot, dtype=np.float32)
    addWeighted(np.float32(box_rot), 1, np.float32(nrc_text), 1, 1, out_img)
    # out_img = (scale(box_rot, with_std=False) + box_scale) + (scale(nrc_text, with_std=False) + nrc_scale)
    addWeighted(out_img, 1, np.float32(wavey_rot), 1, 1, out_img)
    # change lighting
    img = out_img * tilted_plane(out_img.shape)
    # change perspective
    img = change_perspective(img, np.random.randint(0, 40))
    # crop image
    img = crop_img(img, w_center, h_center, box_h + 50, box_w + 50)
    #  blur first
    img = blur_img(img, abs(np.random.rand() * 2))
    # down sample
    if np.random.rand() > 0.5:
        img = transform.rescale(img, scale=np.random.rand() * .5 + .5)
    # resize
    img = transform.resize(img, (w_real, h_real))
    return img


def crop_img(img, h_center, w_center, crop_w, crop_h):
    """
    Crops image
    todo: make jitter around certroid
    :param img:
    :param h_center:
    :param w_center:
    :param crop_w:
    :param crop_h:
    :return:
    """
    # get xy coords for crop
    x_start = w_center - crop_w // 2
    x_stop = x_start + crop_w
    y_start = h_center - crop_h // 2
    y_stop = y_start + crop_h
    return img[y_start:y_stop, x_start:x_stop]


def change_perspective(img, angle):
    """
    changes perspctive of image along z axis only. Angle is rotation from front on image 90 down.
    :param img: ndarray image
    :param angle: int between 0-90
    :return: ndarray image
    """
    # get angle to move rectangle angles in
    x = np.tan(np.pi / 180. * angle) * img.shape[0]
    # make corners for original img
    dst = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
    src = np.array([[0, 0], [x, img.shape[0]], [img.shape[1] - x, img.shape[0]], [img.shape[1], 0]])
    # do transformation
    proj = transform.ProjectiveTransform()
    proj.estimate(dst, src)
    return transform.warp(img, proj)


def tilted_plane(img_shape):
    """
    Returns a randomly tilted plane normalized between [0,1] with some offset

    :param img_shape: tuple out output shape
    :return: ndarray
    """
    x, y = np.meshgrid(range(img_shape[1]), range(img_shape[0]))
    z = (np.random.rand() - 1) * 4 * x + (np.random.rand() - 1) * 4 * y
    # normalize
    z = z / (z.max() - z.min())
    return z


def sinusodal_plane(img_shape):
    pass


def make_nrc_training_imgs(text_list, max_repeats, out_shape, save_path):
    """
    Bulk method to make images. Will makame muliple repeats.
    :param text_path: path to nrc numbers
    :param max_repeats: int number of time to repeat an nrc
    :param out_shape: shape of img
    :param save_path: path to save
    :return: None
    """
    # make figure to be passed
    dpi = 70
    w, h = out_shape
    fig = plt.figure(figsize=((w * 2) / dpi, (h * 2) / dpi), dpi=dpi)
    # make output list
    out_list_file = open(os.path.join(save_path, 'stage2.csv'), 'w')
    # make blanks
    for file_number, text in enumerate(text_list):
        # make nrc
        if len(text) > 0:
            nrc = insert_forward_slash(text, False)
        else:
            nrc = text
        for r in range(max_repeats):
            img = make_nrc_text_box(out_shape[0], out_shape[1], nrc, fig)
            # scale between 0,1
            img_std = (img - img.min()) / (img.max() - img.min())
            img_scaled = img_std * (1) + 0
            img_path = os.path.join(save_path, '{}_{}.jpg'.format(file_number, r))
            skio.imsave(img_path, img_scaled)
            out_list_file.write('{}.jpg,{}\n'.format(img_path, nrc))


def gen_valid_nrc(no_nrc):
    """
    Generates valid nrc numbers (9 characters long and end with a 1)
    :param no_nrc: int
    :return: list
    """
    # make formater
    nrc_fun = '{0:08d}1'.format
    max_nrc_value = 10 ** 8 - 1
    unproc_nrc = np.arange(max_nrc_value, dtype=int)
    sampled_nrc = np.random.choice(unproc_nrc, no_nrc, replace=False)
    out = (map(nrc_fun, sampled_nrc))
    return list(out)


def make_stage_1_image(no_img, img_shape, save_path, valid_charaters, nchar=1):
    """
    Makes training images with only background and nchar
    :param no_img: number of images to make
    :param img_shape: output shape of image
    :param save_path: dir to save files to
    :param valid_charaters: list of charaters to use
    :param nchar: number of charaters to use
    :return:
    """
    out_list_file = open(os.path.join(save_path, 'stage1.csv'), 'w')
    for i in range(no_img):
        noise_param = np.random.choice([.1, .2, .3])
        # make 10% blank
        if np.random.rand() < .1:
            text = ''
        else:
            text = ''.join(np.random.choice(valid_charaters + [''], size=nchar))
        font = np.random.randint(20, 30)
        background = make_background(img_shape, .0, noise_param)
        # do transforms on text
        text_array = make_text(text, img_shape, font_size=font)[0]
        # add lightning
        text_array = text_array * tilted_plane(text_array.shape)
        # change perspective
        out_img = change_perspective(text_array, np.random.randint(0, 40))
        #  blur first
        # text_array = blur_img(text_array, abs(np.random.rand() * 1))
        # combine with noisy background
        out_img = unitscale(background + text_array)
        sp = os.path.join(save_path, '{}.jpg'.format(i))
        skio.imsav

        e(sp, out_img)
        out_list_file.write('{}.jpg,{}\n'.format(i, text))


if __name__ == '__main__':
    # make 3 stages of training images
    number_of_imgs = 5 * 10 ** 4
    out_shape = (64, 512)
    outpath = os.environ['TRANING_PATH']
    valid_charaters = list(map(str, range(10))) + ['/']
    ### Stage 1 ###
    # image with noise background and 1 character and blanks
    stage_1_path = os.path.join(outpath, 'stage1')
    print(stage_1_path)
    if not os.path.exists(stage_1_path):
        os.mkdir(stage_1_path)
    # make_stage_1_image(number_of_imgs, out_shape, stage_1_path, valid_charaters,8)
    ### Stage 2 ###
    stage_2_path = os.path.join(outpath, 'stage2')
    print(stage_2_path)
    if not os.path.exists(stage_2_path):
        os.mkdir(stage_2_path)
    # nrc lngth of characters on wavy background
    nrcs = np.unique(gen_valid_nrc(number_of_imgs))
    # append blank nrcs last 10%
    nrcs[-int(len(nrcs)*.1):] = ''
    make_nrc_training_imgs(nrcs, 4, out_shape, stage_2_path)
    ### Stage 3 ###
    # full nrc simulation

    ### Stage 4 ###
    # add rotation and lighting from Stage 3

    # text_path = 'id_numbers.csv'
