import xml.etree.ElementTree
import urllib.request
import os
import subprocess

import requests


def download_img(url: str, folder: str, name: str) -> bool:
    with open(os.path.join(folder, name + '.JPEG'), 'wb') as handle:
        try:
            response = requests.get(url, stream=True, timeout=60)
        except IOError as e:
            print('Failed to download image from {}: {}'.format(url, e))
            return False
        if not response.ok:
            print('Failed to download image from {}: {}'.format(url, response))
            return False
        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)
    return True


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
    response = urllib.request.urlopen(url)
    data = response.read()  # a `bytes` object
    text = data.decode('utf-8')  # a `str`; this step can't be used if data is binary
    return text


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


def load_synsets() -> (list, dict):
    synset_list_url = 'http://www.image-net.org/api/text/imagenet.bbox.obtain_synset_list'
    synset_list_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'synset_list.txt')

    if not os.path.isfile(synset_list_file):
        download_web_file(synset_list_url, synset_list_file)

    list = []
    wnid2id = {}
    with open(synset_list_file, 'r') as f:
        for idx, line in enumerate(f):
            wnid = line.rstrip()
            list.append(wnid)
            wnid2id[wnid] = idx
    return list, wnid2id


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
        name = _get_unique_element(obj_block, 'name').text
        bndbox_block = _get_unique_element(obj_block, 'bndbox')
        xmax = int(_get_unique_element(bndbox_block, 'xmax').text)
        xmin = int(_get_unique_element(bndbox_block, 'xmin').text)
        ymax = int(_get_unique_element(bndbox_block, 'ymax').text)
        ymin = int(_get_unique_element(bndbox_block, 'ymin').text)
        objects.append({'name': name,
                        'xmax': xmax,
                        'xmin': xmin,
                        'ymax': ymax,
                        'ymin': ymin})

    return {'width':    width,
            'height':   height,
            'folder':   folder,
            'filename': filename,
            'objects':  objects}
