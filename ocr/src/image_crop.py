"""
Interactive cropping that outputs data like will receive during call
"""

import io
from os import path

import pandas as pd
import pygame
import urllib.request
from common.dbengine import DBEngine
from scipy.ndimage.interpolation import rotate
from skimage import io as skio
from numpy.random import permutation

pygame.init()


def download_image(url, return_ndarray=False):
    """
    Downloads image from url. can return a ndarray or pygame surface
    :param url: str
    :param return_ndarray: bool if
    :return: ndarray or surface
    """
    response = urllib.request.urlopen(url)
    buf_data = response.read()
    if return_ndarray:
        data = skio.imread(io.BytesIO(buf_data), as_grey=True)
    else:
        data = pygame.image.load(io.BytesIO(buf_data))
    return data


def displayImage(screen, px, topleft, prior):
    """
    Draws transparent box on screen
    :param screen:
    :param px:
    :param topleft:
    :param prior:
    :return:
    """
    # ensure that the rect always has positive width, height
    x, y = topleft
    width = pygame.mouse.get_pos()[0] - topleft[0]
    height = pygame.mouse.get_pos()[1] - topleft[1]
    if width < 0:
        x += width
        width = abs(width)
    if height < 0:
        y += height
        height = abs(height)

    # eliminate redundant drawing cycles (when mouse isn't moving)
    current = x, y, width, height
    if not (width and height):
        return current
    if current == prior:
        return current

    # draw transparent box and blit it onto canvas
    screen.blit(px, px.get_rect())
    im = pygame.Surface((width, height))
    im.fill((128, 128, 128))
    pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
    im.set_alpha(128)
    screen.blit(im, (x, y))
    pygame.display.flip()

    # return current box extents
    return (x, y, width, height)


def setup(px):
    """
    Displays image from a surface
    :param px:
    :return:
    """
    screen = pygame.display.set_mode(tuple(px.get_rect())[2:])
    screen.blit(px, px.get_rect())
    pygame.display.flip()
    return screen, px


def rotate_event(screen, px):
    """
    Handles keys to rotate screen and quit
    :param screen:
    :param px:
    :return:screen, px, total_rotation angle
    """
    total_rot = 0
    angle = 0
    stop_iter = True
    while stop_iter:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                angle = -90
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                angle = 90
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                angle = 5
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                angle = -5
            else:
                angle = 0
            total_rot += angle
            px, screen = rotate_image(px, angle)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                stop_iter = False
    return screen, px, total_rot


def rotate_image(px, angle):
    """
    Does rotation in pygame
    :param px:
    :param angle:
    :return:
    """
    px = pygame.transform.rotate(px, angle)
    screen_rot = pygame.display.set_mode(list(px.get_rect())[2:])
    screen_rot.blit(px, px.get_rect())
    pygame.display.flip()
    return px, screen_rot


def crop_image(screen, px):
    """
    Handles cropping events. From the mouse
    :param screen:
    :param px:
    :return:
    """
    topleft = bottomright = prior = None
    stop_iter = True
    while stop_iter:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                if not topleft:
                    topleft = event.pos
                else:
                    bottomright = event.pos
                    stop_iter = False

        if topleft:
            prior = displayImage(screen, px, topleft, prior)
    return (topleft + bottomright)


def get_image_urls(dbengine):
    """
    Queries DB for nrc images
    :param dbengine:
    :return:
    """
    query = '''select image.id, CONCAT(cdn.url,cdn_image_id) as url, value as id_number,  enum as id_type
                from image
                inner join image_capture on image.id = image_capture.image_id 
                inner join image_type on image_type.id = image.image_type_id
                inner join content_delivery_network cdn on cdn.id = image.cdn_id
                where image.id > 10 -- remove test images
                        and enum = 'ZM_NRC_FRONT';'''
    result = pd.read_sql(query, dbengine.conn)
    return result


def scale(img):
    """
    Scales image between 0,1
    :param img: ndarray
    :return: ndarray
    """
    img_std = (img - img.min()) / (img.max() - img.min())
    img_scaled = img_std * (1) + 0
    return img_scaled

class Pane(object):
    def __init__(self, screen):
        self.black = (0,0,0)
        self.red = (255,0,0)
        self.font = pygame.font.SysFont('Arial', 25)
        self.screen = screen
        pygame.display.update()


    def addRect(self, left, upper, right, lower):
        self.rect = pygame.draw.rect(self.screen, (self.red), (left, lower, right-left, upper-lower), 2)
        pygame.display.update()

    def addText(self, text, left, upper, right, lower):
        self.screen.blit(self.font.render(text, True, (self.red)), (left, lower))
        pygame.display.update()

def main(dbengine, save_path, duplicates=4, resume=True):
    """
    Create files for OCR validation in standard format. black and white images, with
    :param file_path:
    :param save_path:
    :param duplicates:
    :param resume:
    :return:
    """
    # get all files to edit in dir
    urls_df = get_image_urls(dbengine)
    # randomize
    urls_df = urls_df.loc[permutation(urls_df.index)]
    # if resume remove files that have been worked on
    save_list = path.join(save_path, 'stage3.csv')
    if path.exists(save_list):
        out_data = pd.read_csv(save_list)
        # get saved index
        ids = out_data[out_data.columns[0]].apply(lambda s: int(s.split('_')[0])).unique()
        urls_df = urls_df[urls_df.id.isin(ids) == False]
        out_data = out_data.set_index(out_data.columns[0])
    else:
        out_data = pd.DataFrame([], columns=[1, 'xy_start', 'width', 'height'])
    j = 0
    for i, row in urls_df.iterrows():
        print('{} out of {} images'.format(j, urls_df.shape[0]))
        print(row.url)
        j+=1
        # get image
        screen, px = setup(download_image(row.url))
        # check if actually an NRC
        answer = input('Do you want to edit image any key to edit?(Y|N) ').lower()
        if answer == 'n':
            continue
        nrc = input('Type in correct nrc if not {} '.format(row.id_number))
        if len(nrc) == 0:
            nrc = row.id_number
        # do this part for multiple repeats
        im_orig = download_image(row.url, True)
        print(im_orig.shape)
        for repeat in range(duplicates):
            print (
                'Use arrow <- -> to rotate 90 degrease \n use up and down arrows to rotate 5 degrease and q to exit.')
            screen, px_rot, total_rot = rotate_event(screen, px)
            print('select area to crop out with mouse.')
            (left, upper, right, lower) = crop_image(screen, px_rot)
            # ensure output rect always has positive width, height
            if right < left:
                left, right = right, left
            if lower > upper:
                lower, upper = upper, lower
            # do rotation and crop
            im = rotate(im_orig, total_rot)
            im = scale(im[lower:upper, left:right])
            # save img and text
            save_str = '{}_{}.jpg'.format(row.id, repeat)
            skio.imsave(path.join(save_path, save_str), im)
            # save to data frame [nrc xy_start, width and height
            out_data.loc[save_str] = nrc, (left, right), right - left, upper - lower
        # save out_data as check up
        out_data.to_csv(save_list)

    pygame.display.quit()


if __name__ == "__main__":
    dbengine = DBEngine('zoona_arch')['android']
    main(dbengine, 'stage3')
