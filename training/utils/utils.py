import os
import subprocess
import xml.etree.ElementTree
from typing import *

import requests


def download_img(url: str, folder: str, name: str) -> bool:
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    try:
        with open(os.path.join(folder, name + '.JPEG'), 'wb') as handle:
            response = requests.get(url, allow_redirects=False, stream=True, timeout=1)
            if not response.ok:
                # print('Failed to download image from {}: {}'.format(url, response))
                return False
            for block in response.iter_content(1024):
                if not block:
                    break
                handle.write(block)
        return True
    except Exception as e:
        # print('Failed to download image from {}: {}'.format(url, e))
        return False


def extract_archive(archive_path: str, output_dir: str = None, async: bool = False) -> bool:
    if output_dir is not None:
        args = ['tar', '-xf', archive_path, '-C', output_dir]
        if async:
            subprocess.Popen(args)
            return True
        else:
            ret = subprocess.call(args)
            return not ret
    else:
        args = ['tar', '-xf', archive_path]
        if async:
            subprocess.Popen(args)
            return True
        else:
            ret = subprocess.call(args)
            return not ret


def read_web_file(url: str) -> str:
    r = requests.get(url, allow_redirects=True)
    return r.text


def download_web_file(url: str, path: str) -> None:
    r = requests.get(url, allow_redirects=True)
    # Handle redirecting API.
    while r.apparent_encoding == 'ascii' and 'url=' in r.text:
        domain_end = url.find('/', url.find('//') + 2)
        if domain_end < 0:
            domain_end = len(url)
        domain = url[:domain_end]
        resource_start = r.text.find('url=') + 4
        resource_end = r.text.find('"', resource_start)
        resource_url = r.text[resource_start:resource_end]
        redirected_url = domain + resource_url
        r = requests.get(redirected_url, allow_redirects=True)
    with open(path, 'wb') as f:
        f.write(r.content)


def load_synsets() -> (List[str], List[str], Dict[str, int]):
    synset_list_url = 'http://www.image-net.org/api/text/imagenet.bbox.obtain_synset_list'
    synset_list_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'synset_list.txt')

    if not os.path.isfile(synset_list_file):
        download_web_file(synset_list_url, synset_list_file)

    list = []
    wnid2id = {}
    with open(synset_list_file, 'r') as f:
        for i, line in enumerate(f):
            wnid = line.rstrip()
            list.append(wnid)
            wnid2id[wnid] = i

    desc_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'synset_desc.txt')
    desc = [None] * len(list)
    num_desc_loaded = 0
    if os.path.isfile(desc_file):
        with open(desc_file, 'r') as f:
            for i, line in enumerate(f):
                if len(line):
                    desc[i] = line
                    num_desc_loaded += 1

    if num_desc_loaded != len(list):
        # Retrieve synset descriptions.
        print('Retrieving synset descriptions.')
        word_api = 'http://www.image-net.org/api/text/wordnet.synset.getwords?wnid='
        with open(desc_file, 'w') as f:
            for i, synset in enumerate(list):
                desc[i] = read_web_file(word_api + synset).replace('\r', '').rstrip('\n').replace('\n', ', ')
                f.write(desc[i] + '\n')
                print('{}/{}: {}'.format(i, len(list), desc[i]))

    return list, desc, wnid2id


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def _get_unique_element(xml_block: xml.etree.ElementTree.Element, name: str) -> xml.etree.ElementTree.Element:
    """
    Get a unique element from a XML block by name.
    """
    return xml_block.iter(name).__next__()


def read_annotation(annotation_fn: str) -> dict:
    e = xml.etree.ElementTree.parse(annotation_fn).getroot()

    size_block = _get_unique_element(e, 'size')
    width = int(_get_unique_element(size_block, 'width').text)
    height = int(_get_unique_element(size_block, 'height').text)
    folder = _get_unique_element(e, 'folder').text
    filename = _get_unique_element(e, 'filename').text

    objects = []
    for obj_block in e.iter('object'):
        obj = {}
        for element in iter(obj_block):
            if element.tag == 'bndbox':
                obj['xmax'] = int(_get_unique_element(element, 'xmax').text)
                obj['xmin'] = int(_get_unique_element(element, 'xmin').text)
                obj['ymax'] = int(_get_unique_element(element, 'ymax').text)
                obj['ymin'] = int(_get_unique_element(element, 'ymin').text)
            else:
                obj[element.tag] = element.text
        objects.append(obj)

    return {'width':    width,
            'height':   height,
            'folder':   folder,
            'filename': filename,
            'objects':  objects}
