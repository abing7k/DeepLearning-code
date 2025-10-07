import os
import time
import numpy as np
import torch
from typing import Sequence
from d2l import torch as d2l


IMG_PATH = '../img/catdog.jpg'
RATIOS = [1, 2, 0.5]
ANCHOR_CHANNELS = 10  # 与书中示例保持一致

def load_image(path: str) -> np.ndarray:
    """Load demo image or fall back to a gray placeholder with expected size."""
    if os.path.exists(path):
        return d2l.plt.imread(path)
    height, width = 561, 728
    placeholder = np.full((height, width, 3), 0.6, dtype=np.float32)
    d2l.plt.figure()
    d2l.plt.text(0.5, 0.5, 'Placeholder image', ha='center', va='center')
    d2l.plt.axis('off')
    d2l.plt.close()
    return placeholder


def display_anchors(img: np.ndarray, fmap_w: int, fmap_h: int, sizes: Sequence[float], bbox_scale: torch.Tensor) -> None:
    """Generate anchors on a feature map and overlay them on the image."""
    d2l.set_figsize()
    fmap = torch.zeros((1, ANCHOR_CHANNELS, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=sizes, ratios=RATIOS)
    axes = d2l.plt.imshow(img).axes
    d2l.show_bboxes(axes, anchors[0] * bbox_scale)
    d2l.plt.title(f'fmap: {fmap_h}x{fmap_w}, sizes={sizes}')
    d2l.plt.axis('off')
    d2l.plt.show()


def main() -> None:
    start_time = time.time()
    torch.set_printoptions(precision=2)

    img = load_image(IMG_PATH)
    height, width = img.shape[:2]
    print(f'Loaded image with shape: {height} x {width}')

    bbox_scale = torch.tensor((width, height, width, height))

    print('Small-scale anchors on dense 4x4 grid (size=0.15)')
    display_anchors(img, fmap_w=4, fmap_h=4, sizes=[0.15], bbox_scale=bbox_scale)

    print('Medium anchors on 2x2 grid (size=0.4)')
    display_anchors(img, fmap_w=2, fmap_h=2, sizes=[0.4], bbox_scale=bbox_scale)

    print('Large anchors on 1x1 grid (size=0.8)')
    display_anchors(img, fmap_w=1, fmap_h=1, sizes=[0.8], bbox_scale=bbox_scale)

    elapsed = time.time() - start_time
    print(f'Finished in {elapsed:.2f} seconds.')


if __name__ == '__main__':
    main()
