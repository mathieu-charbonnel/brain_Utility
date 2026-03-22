from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def display_image(
    image: sitk.Image,
    overlay: Optional[np.ndarray] = None,
    x: int = 0,
    y: int = 0,
    z: int = 0,
    window: float = 1000,
    level: float = 400,
) -> None:
    img_array = sitk.GetArrayFromImage(image)

    vmin = level - window / 2
    vmax = level + window / 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        img_array[z, :, :], cmap="gray", vmin=vmin, vmax=vmax
    )
    axes[0].set_title(f"Axial (z={z})")

    axes[1].imshow(
        img_array[:, y, :], cmap="gray", vmin=vmin, vmax=vmax
    )
    axes[1].set_title(f"Coronal (y={y})")

    axes[2].imshow(
        img_array[:, :, x], cmap="gray", vmin=vmin, vmax=vmax
    )
    axes[2].set_title(f"Sagittal (x={x})")

    if overlay is not None:
        alpha = 0.3
        axes[0].imshow(
            overlay[z, :, :], cmap="Reds", alpha=alpha, vmin=0, vmax=vmax
        )
        axes[1].imshow(
            overlay[:, y, :], cmap="Reds", alpha=alpha, vmin=0, vmax=vmax
        )
        axes[2].imshow(
            overlay[:, :, x], cmap="Reds", alpha=alpha, vmin=0, vmax=vmax
        )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
