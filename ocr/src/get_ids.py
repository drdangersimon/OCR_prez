import os
import urllib

import pandas as pd


def get_image_path(engine):
    """
    Queryies database to get path to image stores
    :param engine: sqlalchemy connection engine
    :return: dict {id: http path}
    """
    query = '''
            select CONCAT(cdn.url,cdn_image_id) as url,cdn_image_id as img_name, value as id_number,  enum as id_type 
            from image
             inner join image_capture on image.id = image_capture.image_id 
            inner join image_type on image_type.id = image.image_type_id
            inner join content_delivery_network cdn on cdn.id = image.cdn_id
            where value <> '' 
            '''
    """and enum = 'ZM_NRC_FRONT'"""
    images = pd.read_sql_query(query, engine)
    # todo: may need parsing
    # make into dictionary
    #out_dict = images.set_index('url')['id_number'].to_dict()
    return images


def download_images_nrc(image_paths, save_path='ids'):
    """
    Grabs
    :param image_paths: dict {id: http path}
    :return: 
    """
    image_paths = image_paths.set_index('url')['id_number'].to_dict()
    assert os.path.exists(save_path), '{} does not exists.'.format(save_path)
    for url in image_paths:
        # remove / if in id
        file_name = os.path.join(save_path, image_paths[url].replace('/', '') + '.jpg')
        # skip if file exists
        if os.path.exists(file_name):
            continue
        try:
            resource = urllib.urlopen(url)
        except IOError:
            print '{}'.format(url)
            continue

        with open(os.path.join(save_path, image_paths[url] + '.jpg'), "wb") as output:
            output.write(resource.read())

def download_images(image_paths, save_path='ids'):
    """
    Grabs
    :param image_paths: dict {id: http path}
    :return: 
    """
    image_paths = image_paths.set_index('url')['img_name'].to_dict()
    assert os.path.exists(save_path), '{} does not exists.'.format(save_path)
    for url in image_paths:
        # remove / if in id
        file_name = os.path.join(save_path, image_paths[url].replace('/', '') + '.jpg')
        # skip if file exists
        if os.path.exists(file_name):
            continue
        try:
            resource = urllib.urlopen(url)
        except IOError:
            print '{}'.format(url)
            continue

        with open(os.path.join(save_path, image_paths[url] + '.jpg'), "wb") as output:
            output.write(resource.read())


if __name__ == '__main__':
    from common.dbengine import DBEngine

    dbengine = DBEngine('zoona_arch')
    dbengine.get_database('android')
    paths = get_image_path(dbengine.android.conn)
    download_images(paths,'all_ids')
