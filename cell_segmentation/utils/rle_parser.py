from pycocotools import mask as mutils
from skimage import measure
import numpy as np
from typing import List, Tuple


def rle_decode(
    mask_rle: List[int], height: int, width: int, channels: int, color: int = 1
) -> np.ndarray:
    """
    Convert RLE encoded mask to numpy array.

    source: https://www.kaggle.com/ammarnassanalhajali/sartorius-segmentation-detectron2-training#Loading-Dataset

    Parameters:
    -----------
        mask_rle: List[int]
            RLE encoded pixels
        height: int
            height of image
        width: int
            width of image
        channels: int
            number of channels
        color: int
            color of mask

    Returns:
    -----------
        mask: np.ndarray
            pixel segmentation mask
    """
    s = mask_rle.split()

    starts = list(map(lambda x: int(x) - 1, s[0::2]))
    lengths = list(map(int, s[1::2]))

    ends = [x + y for x, y in zip(starts, lengths)]
    img = np.zeros((height * width, channels), dtype=np.float32)

    for start, end in zip(starts, ends):
        img[start:end] = color

    return img.reshape((height, width, channels))


def get_mask_rle(
    ann: List[int], image_height: int, image_width: int, channels: int = 1
) -> np.ndarray:
    """
    Convert annotation run length encoded(RLE) pixels to pixel mask.

    Run-length encoding(RLE) is a form of lossless data compression.
    It is a simple yet efficient format for storing binary masks. RLE first divides a vector (or vectorized image)
    into a series of piecewise constant regions and then for each piece simply stores the length of that piece.
    For example, given M=[0 0 1 1 1 0 1] the RLE counts would be [2 3 1 1], or for M=[1 1 1 1 1 1 0] the
    counts would be [0 6 1] (note that the odd counts are always the numbers of zeros).


    source: https://www.kaggle.com/ammarnassanalhajali/sartorius-segmentation-detectron2-training#Loading-Dataset

    Parameters:
    -----------
        ann: List[int]
            annotation of mask
        image_height: int
            height of image
        image_width: int
            width of image
        channels: int
            number of channels

    Returns:
    -----------
        mask: np.ndarray
            pixel segmentation mask
    """
    decoded_mask = rle_decode(
        ann, height=image_height, width=image_width, channels=channels, color=1
    )
    mask = decoded_mask[:, :, 0]
    mask = np.array(mask, dtype=np.uint8)
    return mask


def get_segmentation_area(
    segmentation: List[List[float]], height: int, width: int
) -> int:
    RLEs = mutils.frPyObjects(segmentation, height, width)
    RLE = mutils.merge(RLEs)
    return mutils.area(RLE)


def validate_bbox(bbox: List[int], image_height: int, image_width: int) -> bool:
    """
    Validate the bounding box of mask is inside the image.

    Parameters:
    -----------
        bbox: List[int]
            bounding box of mask
        image_height: int
            height of image
        image_width: int
            width of image

    Returns:
    --------
        valid_annotation: bool
            True if bounding box is inside the image, False otherwise
    """
    valid_annotation = True
    if (
        (bbox[2] <= 0)
        or (bbox[3] <= 0)
        or (bbox[0] + bbox[2] >= image_width)
        or (bbox[1] + bbox[3] >= image_height)
    ):
        valid_annotation = False
    return valid_annotation


def validate_segmentation_length(segmentations: List[int]) -> bool:
    """
    Validate the length of segmentation and ensure they are greater than 7.

    Parameters:
    -----------
        segmentations: List[int]
            segmentation of mask

    Returns:
    --------
        valid_annotation: bool
            True if length of segmentation is greater than 7, False otherwise
    """
    valid_annotation = True

    for segmentation in segmentations:
        if len(segmentation) < 7:
            valid_annotation = False
    return valid_annotation


def contours_to_segmentation(binary_mask: np.ndarray, level=0.5) -> List[int]:
    segmentations = []
    contours = measure.find_contours(binary_mask, level)
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentations.append(segmentation)
    return segmentations


def decode_rle(mask_rle: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Convert RLE encoded mask to numpy array and extract bounding box of mask.

    source: https://www.kaggle.com/ammarnassanalhajali/sartorius-segmentation-detectron2-training#Loading-Dataset

    Parameters:
    -----------
    mask_rle: np.ndarray
        RLE encoded pixels

    Returns:
        mask (np.ndarray): pixel segmentation mask
        bbox (List[int]): bounding box of mask
    """
    ground_truth_binary_mask = mask_rle
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mutils.encode(fortran_ground_truth_binary_mask)

    ground_truth_bounding_box = mutils.toBbox(encoded_ground_truth)
    bbox = ground_truth_bounding_box.tolist()
    segmentations = contours_to_segmentation(binary_mask=ground_truth_binary_mask)

    return segmentations, bbox
