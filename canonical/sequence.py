"""
Canonical-space sequence representation for 4-D cardiac cine MRI.

Provides the CanonicalSequence class (previously in kwatsch/temporal_canonical_space.py).
"""

import SimpleITK as sitk
import numpy as np

from canonical.image import CanonicalImage
from utils.coords import (
    KEY_SAX_VIEW,
    KEY_4CH_VIEW,
    KEY_4CH_SEG_VIEW,
    KEY_2CH_VIEW,
    KEY_2CH_SEG_VIEW,
)


class CanonicalSequence:
    """
    Build a list of per-timepoint CanonicalImage objects from a 4-D cine sequence.

    Accepts either a 4-D sitk.Image (X, Y, Z, T) or a list of 3-D sitk.Image objects.
    """

    def __init__(
        self,
        image_seq,
        mask_seq,
        label,
        device,
        normalize=True,
        do_flip=False,
        rv_lv_rot_matrix=None,
        lax_img_seq=None,
        lax_mask_seq=None,
    ):
        self.frames = []
        self.T = image_seq.GetSize()[-1]
        self.device = device

        for t in range(self.T):
            img_t = self.ensure_3d(image_seq, t)
            mask_t = self.ensure_3d(mask_seq, t)
            cimg = CanonicalImage(
                img_t,
                mask_t,
                label=label,
                device=device,
                normalize=normalize,
                z_flip=do_flip,
            )
            cimg.align_images(rv_lv_rot_matrix=rv_lv_rot_matrix)

            if lax_img_seq is not None and lax_mask_seq is not None:
                lax_t = self.ensure_3d(lax_img_seq, t)
                lax_mask_t = self.ensure_3d(lax_mask_seq, t)
                cimg.add_view(
                    lax_t,
                    key=KEY_4CH_VIEW,
                    dtype=np.float32,
                    normalize=normalize,
                    keep_3d=False,
                    origin_offset=None,
                )
                cimg.add_view(
                    lax_mask_t,
                    key=KEY_4CH_SEG_VIEW,
                    dtype=np.int32,
                    keep_3d=False,
                    origin_offset=None,
                )

            self.frames.append(cimg)

    def __getitem__(self, t):
        return self.frames[t]

    def __len__(self):
        return len(self.frames)

    def get_all_frames(self):
        """Return the list of all CanonicalImage frames."""
        return self.frames

    def ensure_3d(self, img_or_list, t: int):
        """
        Accept a 4-D sitk.Image, a list of 3-D sitk.Images, or a 3-D sitk.Image
        and return the 3-D image at timepoint t.
        """
        if isinstance(img_or_list, sitk.Image):
            if img_or_list.GetDimension() == 4:
                return self.extract_3d_at(img_or_list, t)
            return img_or_list
        elif isinstance(img_or_list, (list, tuple)):
            return img_or_list[int(t)]
        raise TypeError(
            "ensure_3d expects a sitk.Image (3D/4D) or a list/tuple of 3D sitk.Images."
        )

    def extract_3d_at(self, img4d: sitk.Image, t: int) -> sitk.Image:
        """
        Return a true 3-D SimpleITK image for timepoint t extracted from a 4-D image.
        Falls back to ExtractImageFilter if slicing does not collapse the time dimension.
        """
        t = int(t)
        try:
            out = img4d[:, :, :, t]
            if isinstance(out, sitk.Image) and out.GetDimension() == 3:
                return out
        except Exception:
            pass

        X, Y, Z, T = img4d.GetSize()
        ef = sitk.ExtractImageFilter()
        if hasattr(ef, "SetDirectionCollapseToStrategy"):
            try:
                ef.SetDirectionCollapseToStrategy(
                    sitk.ExtractImageFilter.DIRECTIONCOLLAPSETOIDENTITY
                )
            except Exception:
                if hasattr(sitk, "sitkDirectionCollapseToIdentity"):
                    ef.SetDirectionCollapseToStrategy(
                        sitk.sitkDirectionCollapseToIdentity
                    )
        ef.SetIndex((0, 0, 0, t))
        ef.SetSize((X, Y, Z, 0))
        return ef.Execute(img4d)
