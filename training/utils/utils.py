import xml.etree.ElementTree


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


def _get_unique_element(xml_block, name):
    """
    Get a unique element from a XML block by name.
    Args:
        xml_block (xml.etree.ElementTree.Element): XML block.
        name (str): element name.
    Returns:
        xml.etree.ElementTree.Element: the corresponding element block.
    """
    return xml_block.iter(name).__next__()


def read_annotation(annotation_fn):
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
        objects.append({'name': name, 'xmax': xmax, 'xmin': xmin, 'ymax': ymax, 'ymin': ymin})

    return {'width': width, 'height': height, 'folder': folder, 'filename': filename, 'objects': objects}
