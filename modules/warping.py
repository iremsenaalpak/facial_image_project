import os

import cv2
import numpy as np

from utils.image_utils import blend_with_mask, create_polygon_mask, recolor_preserve_luminance
from modules.hair_segmenter import get_hair_segmenter


class FaceWarper:
    """
    Geometric expression warping and creative face overlays.

    Expression modes: smile, eyebrow_raise, lip_widen, face_slim
    Creative overlays: lip_color, glasses, hair_color, frame_photo
    """

    # --- MediaPipe 468-landmark index constants ---

    SMILE_CORNERS = [61, 291]
    SMILE_UPPER = [37, 267]
    SMILE_LOWER = [84, 314]
    SMILE_CENTER = [17, 0]

    EYEBROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    EYEBROW_R = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

    LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                 375, 321, 405, 314, 17, 84, 181, 91, 146]
    LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                 324, 318, 402, 317, 14, 87, 178, 88, 95]

    JAW_SLIM = [234, 127, 93, 132, 58, 172, 136, 150, 149, 176,
                148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356]

    FACE_OVAL = [10, 109, 67, 103, 54, 21, 162, 127, 234,
                 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
                 377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
                 356, 389, 251, 284, 332, 297, 338, 10]

    FOREHEAD_TOP = [10, 338, 297, 332, 284, 251, 389, 356,
                    454, 323, 361, 288, 397, 365, 379, 378, 400,
                    377, 152, 148, 176, 149, 150, 136, 172, 58,
                    132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

    # Eye groups for glasses — use only genuine eye landmarks (146 is a mouth point, excluded)
    EYE_L_GROUP = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145]
    EYE_R_GROUP = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381]
    EYE_L_CORNERS = (33, 133)
    EYE_R_CORNERS = (362, 263)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def warp_expression(self, image, landmarks, mode, intensity=1.0):
        """
        Apply expression warping.

        smile, face_slim, lip_widen and eyebrow_raise use dense Gaussian
        displacement fields + cv2.remap (liquify-style) to avoid Delaunay
        triangle tearing and eye duplication.
        """
        if not landmarks or len(landmarks) < 468:
            return image.copy()

        if mode == "smile":
            return self._smile_remap(image, landmarks, intensity)
        if mode == "face_slim":
            return self._face_slim_remap(image, landmarks, intensity)
        if mode == "lip_widen":
            return self._lip_widen_remap(image, landmarks, intensity)
        if mode == "eyebrow_raise":
            return self._eyebrow_raise_remap(image, landmarks, intensity)

        return image.copy()

    def _smile_remap(self, image, landmarks, intensity):
        """
        Anatomically-aware smile warp using multiple control regions.

        A natural smile involves several simultaneous facial movements:
        1. Mouth corners lift upward and pull slightly outward (zygomatic major)
        2. Cheeks bunch upward toward the eyes (orbicularis oculi / cheek apple)
        3. Upper lip curves into a gentle bow
        4. Lower lip and chin remain mostly stable

        Each region uses an independent Gaussian displacement blob with tuned
        sigma and magnitude.  cv2.remap backward-mapping: positive delta in
        map_y pulls content *upward* in the output (the source row is below).

        Compared to the previous 2-Gaussian approach, this uses:
        - 6+ control centres instead of 2
        - ~3× smaller peak displacement to avoid grotesque stretching
        - Directional gating so each blob only affects its intended zone
        - Cheek lift for the natural "bunching" that accompanies a real smile
        """
        h, w = image.shape[:2]
        pts = np.array(landmarks, dtype=np.float64)
        xs, ys = pts[:, 0], pts[:, 1]
        fw = float(xs.max() - xs.min())
        fh = float(ys.max() - ys.min())

        # Key anatomical anchors
        mouth_L = pts[61]            # left mouth corner
        mouth_R = pts[291]           # right mouth corner
        cx_mouth = float((mouth_L[0] + mouth_R[0]) / 2.0)
        cy_mouth = float((mouth_L[1] + mouth_R[1]) / 2.0)
        upper_lip_center = pts[0]    # top of upper lip bow
        chin = pts[152]              # bottom of chin (stabiliser)

        # Cheek apples: midpoint between mouth corner and outer eye corner,
        # biased slightly toward the mouth corner (where the zygomatic pull is)
        eye_L_outer = pts[33]
        eye_R_outer = pts[263]
        cheek_L = mouth_L * 0.55 + eye_L_outer * 0.45
        cheek_R = mouth_R * 0.55 + eye_R_outer * 0.45

        # Nasolabial fold anchors (between nose and mouth corners)
        naso_L = pts[216] if len(pts) > 216 else (pts[61] + pts[48]) / 2.0
        naso_R = pts[436] if len(pts) > 436 else (pts[291] + pts[278]) / 2.0

        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = X.copy()
        map_y = Y.copy()

        def _gauss(cx, cy, sigma):
            """Radial Gaussian weight centred at (cx, cy)."""
            return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (sigma ** 2))

        def _vgate_above(cy_ref, falloff):
            """Soft gate: full strength at/above cy_ref, decays below."""
            below = np.maximum(0.0, Y - cy_ref)
            return np.exp(-(below / falloff) ** 2)

        def _vgate_below(cy_ref, falloff):
            """Soft gate: full strength at/below cy_ref, decays above."""
            above = np.maximum(0.0, cy_ref - Y)
            return np.exp(-(above / falloff) ** 2)

        # ---- Region 1: Mouth corners — gentle upward + outward lift ----
        for corner in [mouth_L, mouth_R]:
            cx_c, cy_c = float(corner[0]), float(corner[1])
            sigma = float(fw * 0.14)
            g = _gauss(cx_c, cy_c, sigma)

            # Vertical gate: affect mostly at and above the corner, cut
            # off sharply below so the chin/lower lip isn't dragged up
            vg = _vgate_above(cy_c, float(fh * 0.12))
            map_y += g * vg * float(intensity * 0.028 * fh)

            # Outward horizontal: push corner away from mouth centre
            sign = 1.0 if cx_c > cx_mouth else -1.0
            map_x -= g * float(sign * intensity * 0.012 * fw)

        # ---- Region 2: Cheek apple lift (zygomatic bunching) ----
        for cheek in [cheek_L, cheek_R]:
            cx_c, cy_c = float(cheek[0]), float(cheek[1])
            sigma = float(fw * 0.18)
            g = _gauss(cx_c, cy_c, sigma)
            # Lift cheeks upward — gentler than mouth corners
            map_y += g * float(intensity * 0.016 * fh)

        # ---- Region 3: Upper lip bow — subtle upward curvature ----
        cx_lip, cy_lip = float(upper_lip_center[0]), float(upper_lip_center[1])
        sigma_lip = float(fw * 0.10)
        g_lip = _gauss(cx_lip, cy_lip, sigma_lip)
        # Gate: only affect the lip, not the nose above
        vg_lip = _vgate_below(cy_lip, float(fh * 0.06))
        map_y += g_lip * vg_lip * float(intensity * 0.010 * fh)

        # ---- Region 4: Nasolabial fold softening ----
        for naso in [naso_L, naso_R]:
            cx_n, cy_n = float(naso[0]), float(naso[1])
            sigma_n = float(fw * 0.08)
            g_n = _gauss(cx_n, cy_n, sigma_n)
            # Tiny outward push to soften the crease
            sign = 1.0 if cx_n > cx_mouth else -1.0
            map_x -= g_n * float(sign * intensity * 0.005 * fw)

        # ---- Region 5: Chin / lower lip stabiliser ----
        # A weak *downward* push on the chin prevents it from being
        # sucked upward by the mouth corner displacement above.
        cx_chin, cy_chin = float(chin[0]), float(chin[1])
        sigma_chin = float(fw * 0.16)
        g_chin = _gauss(cx_chin, cy_chin, sigma_chin)
        vg_chin = _vgate_below(cy_mouth, float(fh * 0.10))
        map_y -= g_chin * vg_chin * float(intensity * 0.008 * fh)

        result = cv2.remap(image, map_x, map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Composite inside face oval only so background stays untouched
        face_pts = [landmarks[i] for i in self.FACE_OVAL if i < len(landmarks)]
        face_mask = create_polygon_mask((h, w), face_pts)
        face_float = cv2.GaussianBlur(
            face_mask.astype(np.float32) / 255.0, (51, 51), 15)[:, :, np.newaxis]
        return (result.astype(np.float32) * face_float +
                image.astype(np.float32) * (1.0 - face_float)).clip(0, 255).astype(np.uint8)

    def _face_slim_remap(self, image, landmarks, intensity):
        """
        Liquify-style face slimming using cv2.remap.

        Pushes the jawline and lower cheeks inward toward the face
        centre line.  Multiple control points are sampled along the
        jaw contour so the displacement is smooth and continuous —
        no Delaunay triangle seams.

        A vertical gate restricts the effect to the lower face
        (below nose level) so the eyes and forehead are untouched.
        """
        h, w = image.shape[:2]
        pts = np.array(landmarks, dtype=np.float64)
        xs, ys = pts[:, 0], pts[:, 1]
        fw = float(xs.max() - xs.min())
        fh = float(ys.max() - ys.min())
        cx = float((xs.min() + xs.max()) / 2.0)

        # Nose bottom = vertical gate line (nothing above this moves)
        nose_bottom_y = float(pts[152][1] + pts[0][1]) / 2.0  # midpoint chin—lip top
        # Use a more accurate reference: nose tip (landmark 1) or base (landmark 2)
        nose_y = float(pts[2][1])  # nose base

        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = X.copy()
        map_y = Y.copy()

        # Jaw contour control points (left jaw → chin → right jaw)
        jaw_indices = [234, 127, 93, 132, 58, 172, 136, 150, 149, 176,
                       148, 152, 377, 400, 378, 379, 365, 397, 288,
                       361, 323, 454]

        for idx in jaw_indices:
            if idx >= len(landmarks):
                continue
            jx, jy = float(pts[idx][0]), float(pts[idx][1])

            # Gaussian blob centred on this jaw point
            sigma = float(fw * 0.14)
            g = np.exp(-((X - jx) ** 2 + (Y - jy) ** 2) / (sigma ** 2))

            # Vertical gate: only below nose level
            dist_below = Y - nose_y
            vgate = np.clip(dist_below / (fh * 0.15 + 1e-6), 0.0, 1.0)

            # Inward horizontal push: signed by which side of centre
            rel_x = (jx - cx) / (fw / 2.0 + 1e-6)
            # Stronger at the sides, weaker near chin centre
            push = float(rel_x * abs(rel_x) ** 0.4 * intensity * 0.04 * fw)
            map_x += g * vgate * push

        result = cv2.remap(image, map_x, map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        return result.astype(np.uint8)

    def _lip_widen_remap(self, image, landmarks, intensity):
        """
        Liquify-style lip widening using cv2.remap.

        Pushes the mouth corners outward horizontally, confined to
        the mouth region so nose, jaw, and chin are unaffected.
        """
        h, w = image.shape[:2]
        pts = np.array(landmarks, dtype=np.float64)
        xs, ys = pts[:, 0], pts[:, 1]
        fw = float(xs.max() - xs.min())
        fh = float(ys.max() - ys.min())

        mouth_L = pts[61]   # left corner
        mouth_R = pts[291]  # right corner
        cx_mouth = float((mouth_L[0] + mouth_R[0]) / 2.0)
        cy_mouth = float((mouth_L[1] + mouth_R[1]) / 2.0)

        # Nose bottom and chin for vertical gating
        nose_bottom_y = float(pts[2][1])
        chin_y = float(pts[152][1])

        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = X.copy()
        map_y = Y.copy()

        # --- Outward push at each mouth corner ---
        for corner in [mouth_L, mouth_R]:
            cx_c, cy_c = float(corner[0]), float(corner[1])
            sigma = float(fw * 0.10)  # tight to the mouth region
            g = np.exp(-((X - cx_c) ** 2 + (Y - cy_c) ** 2) / (sigma ** 2))

            # Vertical band gate: only between nose bottom and chin
            above_nose = np.clip((Y - nose_bottom_y) / (fh * 0.05 + 1e-6), 0.0, 1.0)
            below_chin = np.clip((chin_y - Y) / (fh * 0.05 + 1e-6), 0.0, 1.0)
            vgate = above_nose * below_chin

            # Push outward from mouth centre
            sign = 1.0 if cx_c > cx_mouth else -1.0
            map_x -= g * vgate * float(sign * intensity * 0.025 * fw)

        # --- Subtle horizontal stretch of the entire lip contour ---
        sigma_wide = float(fw * 0.14)
        g_wide = np.exp(-((X - cx_mouth) ** 2 + (Y - cy_mouth) ** 2) / (sigma_wide ** 2))
        above_nose2 = np.clip((Y - nose_bottom_y) / (fh * 0.05 + 1e-6), 0.0, 1.0)
        below_chin2 = np.clip((chin_y - Y) / (fh * 0.05 + 1e-6), 0.0, 1.0)
        vgate2 = above_nose2 * below_chin2
        rel_x = (X - cx_mouth) / (fw / 2.0 + 1e-6)
        map_x -= g_wide * vgate2 * rel_x * float(intensity * 0.012 * fw)

        result = cv2.remap(image, map_x, map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Composite inside face oval
        face_pts = [landmarks[i] for i in self.FACE_OVAL if i < len(landmarks)]
        face_mask = create_polygon_mask((h, w), face_pts)
        face_float = cv2.GaussianBlur(
            face_mask.astype(np.float32) / 255.0, (51, 51), 15)[:, :, np.newaxis]
        return (result.astype(np.float32) * face_float +
                image.astype(np.float32) * (1.0 - face_float)).clip(0, 255).astype(np.uint8)

    def _eyebrow_raise_remap(self, image, landmarks, intensity):
        """
        Eyebrow raise using cv2.remap + masked compositing.

        Strategy: warp the entire image with an upward displacement field
        centred on the brow, then composite ONLY the brow-to-forehead
        pixels from the warped result onto the original image.  The eye
        pixels are always taken from the untouched original, making
        duplication/distortion impossible regardless of displacement size.

        The composite mask covers the region above each upper eyelid
        and extends up into the forehead, with feathered edges for a
        smooth transition.
        """
        h, w = image.shape[:2]
        pts = np.array(landmarks, dtype=np.float64)
        xs, ys = pts[:, 0], pts[:, 1]
        fw = float(xs.max() - xs.min())
        fh = float(ys.max() - ys.min())

        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = X.copy()
        map_y = Y.copy()

        # Build a composite mask: 1.0 in the brow/forehead zone, 0.0 at/below the eyes
        comp_mask = np.zeros((h, w), dtype=np.float32)

        brow_eye_pairs = [
            (self.EYEBROW_L, self.EYE_L_GROUP),
            (self.EYEBROW_R, self.EYE_R_GROUP),
        ]

        for brow_indices, eye_indices in brow_eye_pairs:
            brow_pts = pts[brow_indices]
            eye_pts = pts[eye_indices]

            brow_cx = float(brow_pts[:, 0].mean())
            brow_cy = float(brow_pts[:, 1].mean())

            # Upper eyelid line: the mask cutoff
            eye_top_y = float(eye_pts[:, 1].min())

            # Displacement: simple upward lift centred on brow
            sigma = float(fw * 0.14)
            g = np.exp(-((X - brow_cx) ** 2 + (Y - brow_cy) ** 2) / (sigma ** 2))
            map_y += g * float(intensity * 0.05 * fh)

            # Build per-brow mask: full effect well above the eye,
            # feathered transition from eye_top_y upward over ~transition pixels
            brow_bottom_y = float(brow_pts[:, 1].max())
            # Transition zone: the gap between eye top and brow bottom
            transition = max(float(brow_bottom_y - eye_top_y) * 1.5, fh * 0.04)

            # Horizontal extent: brow width + some padding
            brow_left = float(brow_pts[:, 0].min()) - fw * 0.06
            brow_right = float(brow_pts[:, 0].max()) + fw * 0.06
            h_sigma = (brow_right - brow_left) * 0.6
            h_center = (brow_left + brow_right) / 2.0
            h_weight = np.exp(-((X - h_center) ** 2) / (h_sigma ** 2))

            # Vertical ramp: 0 at eye_top_y, 1 at (eye_top_y - transition)
            dist_above = eye_top_y - Y  # positive when above eye
            v_ramp = np.clip(dist_above / (transition + 1e-6), 0.0, 1.0)

            comp_mask = np.maximum(comp_mask, v_ramp * h_weight)

        # Feather the mask edges
        comp_mask = cv2.GaussianBlur(comp_mask, (21, 21), 7)
        comp_mask_3 = comp_mask[:, :, np.newaxis]

        # Warp the image
        warped = cv2.remap(image, map_x, map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Composite: brow/forehead from warped, everything else from original
        result = (warped.astype(np.float32) * comp_mask_3 +
                  image.astype(np.float32) * (1.0 - comp_mask_3))
        return result.clip(0, 255).astype(np.uint8)

    def apply_lip_color(self, image, landmarks, color=(0, 0, 200), alpha=0.55):
        """
        Paint lips with a semi-transparent color overlay, feathered at the edges.

        Parameters
        ----------
        color : BGR tuple, e.g. (0, 0, 200) = red lips
        alpha : blend strength 0–1
        """
        if not landmarks or len(landmarks) < 468:
            return image.copy()

        h, w = image.shape[:2]
        outer_pts = [landmarks[i] for i in self.LIP_OUTER if i < len(landmarks)]
        inner_pts = [landmarks[i] for i in self.LIP_INNER if i < len(landmarks)]

        outer_mask = create_polygon_mask((h, w), outer_pts)
        inner_mask = create_polygon_mask((h, w), inner_pts)

        lip_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(inner_mask))
        lip_mask_blur = cv2.GaussianBlur(lip_mask.astype(np.float32) / 255.0,
                                         (7, 7), 3)
        return blend_with_mask(image, color, lip_mask_blur, alpha)

    def apply_glasses(self, image, landmarks, style="round", assets_dir="assets/glasses"):
        """
        Overlay a pre-generated glasses PNG (with transparency) onto the face.

        The PNG is scaled and positioned so the two lens centres align with the
        detected eye centres, regardless of face size.

        Parameters
        ----------
        style     : one of 'round' | 'square' | 'aviator' | 'cateye'
        assets_dir: folder containing <style>.png files
        """
        if not landmarks or len(landmarks) < 468:
            return image.copy()

        import math
        from PIL import Image as PILImage

        # --- Accurate eye positioning & rotation ---
        # Vertical & Horizontal: iris/pupil center landmarks
        if len(landmarks) > 473:
            l_eye = np.array([float(landmarks[468][0]), float(landmarks[468][1])])
            r_eye = np.array([float(landmarks[473][0]), float(landmarks[473][1])])
        else:
            l_pts = np.array([landmarks[i] for i in self.EYE_L_GROUP], dtype=np.float32)
            r_pts = np.array([landmarks[i] for i in self.EYE_R_GROUP], dtype=np.float32)
            l_eye = l_pts.mean(axis=0)
            r_eye = r_pts.mean(axis=0)
        
        # Calculate angle for rotation
        dy = r_eye[1] - l_eye[1]
        dx = r_eye[0] - l_eye[0]
        angle = math.degrees(math.atan2(dy, dx))

        pupil_mid_x = (l_eye[0] + r_eye[0]) / 2.0
        pupil_mid_y = (l_eye[1] + r_eye[1]) / 2.0

        # Anchor horizontally to the nose bridge (168) so the glasses don't slide off 
        # the nose when the face is turned/yawed in 3D perspective.
        nose_x = float(landmarks[168][0])
        paste_x = nose_x
        paste_y = pupil_mid_y

        # Euclidean distance ensures correct scaling even when face is tilted
        eye_dist = np.linalg.norm(r_eye - l_eye)

        is_native = False

        if is_native:
            # Generate at high resolution (1200px) for super-sampling anti-aliasing.
            # This completely solves the "pixelated" problem.
            BASE_W = 1200
            glasses_pil, lens_dist_px = FaceWarper._generate_custom_glasses(style, BASE_W)
            
            # Scale so the lens distance perfectly matches the eye distance.
            # This prevents them from looking comically huge.
            scale = eye_dist / lens_dist_px
            
            new_w = max(1, int(BASE_W * scale))
            new_h = max(1, int(glasses_pil.height * scale))
            glasses_resized = glasses_pil.resize((new_w, new_h), PILImage.LANCZOS)
        else:
            # Fallback for custom user-uploaded PNGs
            png_path = os.path.join(assets_dir, f"{style}.png")
            if not os.path.exists(png_path):
                return image.copy()
            glasses_pil = PILImage.open(png_path).convert("RGBA")
            bbox = glasses_pil.getbbox()
            if bbox:
                glasses_pil = glasses_pil.crop((bbox[0], 0, bbox[2], glasses_pil.height))
            
            # Target width is 1.85 * eye_dist. This guarantees that for standard proportional glasses,
            # the frame sits beautifully within the face without looking comically huge.
            target_width = eye_dist * 1.85
            scale = target_width / float(glasses_pil.width)
            new_w = max(1, int(glasses_pil.width * scale))
            new_h = max(1, int(glasses_pil.height * scale))
            glasses_resized = glasses_pil.resize((new_w, new_h), PILImage.LANCZOS)
        
        # Rotate the glasses (using expand=True to prevent corner clipping)
        glasses_rotated = glasses_resized.rotate(-angle, resample=PILImage.BICUBIC, expand=True)

        # Top-left corner to paste the center of the rotated glasses onto the nose/pupil anchor
        ox = int(paste_x - glasses_rotated.width / 2.0)
        oy = int(paste_y - glasses_rotated.height / 2.0)

        # Draw dynamic temples directly on a copy of the image BEFORE compositing glasses.
        # This ensures they connect exactly to the ears and sit behind the lenses.
        img_copy = image.copy()
        ear_l = landmarks[127] if len(landmarks) > 127 else None
        ear_r = landmarks[356] if len(landmarks) > 356 else None
        
        # Because we cropped the image horizontally to its exact bounding box,
        # the hinges are exactly at the left and right edges (new_w / 2 from center).
        hinge_dist = new_w / 2.0
        rad = math.radians(angle)
        
        hx_l = int(paste_x - hinge_dist * math.cos(rad))
        hy_l = int(paste_y - hinge_dist * math.sin(rad))
        
        hx_r = int(paste_x + hinge_dist * math.cos(rad))
        hy_r = int(paste_y + hinge_dist * math.sin(rad))
        
        # Make temples slightly thinner for elegance
        thickness = max(1, int(new_w * 0.008))
        frame_color = (15, 15, 15)
        
        if ear_l is not None:
            tx_l, ty_l = int(ear_l[0]), int(ear_l[1])
            # Glasses sit slightly above the earlobe
            ty_l = int(ty_l - (hx_r - hx_l) * 0.02)
            cv2.line(img_copy, (hx_l, hy_l), (tx_l, ty_l), frame_color, thickness, cv2.LINE_AA)
            
        if ear_r is not None:
            tx_r, ty_r = int(ear_r[0]), int(ear_r[1])
            ty_r = int(ty_r - (hx_r - hx_l) * 0.02)
            cv2.line(img_copy, (hx_r, hy_r), (tx_r, ty_r), frame_color, thickness, cv2.LINE_AA)

        # Alpha-composite glasses onto the face
        h, w = image.shape[:2]
        face_pil = PILImage.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)).convert("RGBA")
        canvas = PILImage.new("RGBA", (w, h), (0, 0, 0, 0))
        canvas.paste(glasses_rotated, (ox, oy), glasses_rotated)
        composited = PILImage.alpha_composite(face_pil, canvas)
        return cv2.cvtColor(np.array(composited.convert("RGB")), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Glasses asset generation
    # ------------------------------------------------------------------

    @staticmethod
    def ensure_glasses_assets(assets_dir="assets/glasses"):
        """Generate all four glasses PNG files if they don't already exist or need resizing."""
        from PIL import Image as PILImage
        os.makedirs(assets_dir, exist_ok=True)
        for style in ("round", "square", "aviator", "cateye"):
            path = os.path.join(assets_dir, f"{style}.png")
            needs_regen = True
            if os.path.exists(path):
                try:
                    with PILImage.open(path) as img:
                        if img.size == (800, 300):
                            needs_regen = False
                except:
                    pass
            if needs_regen:
                img = FaceWarper._create_glasses_image(style)
                img.save(path)

    @staticmethod
    def _create_glasses_image(style):
        """
        Return a PIL RGBA image (800×300) of the requested glasses style.

        Internal geometry:
          Left  lens centre: (250, 150)
          Right lens centre: (550, 150)
          Canvas width     : 800   height: 300
        """
        from PIL import Image as PILImage, ImageDraw

        W, H = 800, 300
        img = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        FRAME = (15, 15, 15, 255)
        LENS  = (50, 50, 50, 60)       # semi-transparent dark tint
        FW    = 9                       # frame stroke width

        L = (250, 150)   # left  lens centre
        R = (550, 150)   # right lens centre

        def bridge(lx, rx, y, drop=18):
            """Curved nose bridge: two verticals + horizontal at bottom."""
            draw.line([(lx, y), (lx, y + drop)], fill=FRAME, width=FW - 2)
            draw.line([(rx, y), (rx, y + drop)], fill=FRAME, width=FW - 2)
            draw.line([(lx, y + drop), (rx, y + drop)], fill=FRAME, width=FW - 2)

        if style == "round":
            rx, ry = 82, 72
            for cx, cy in (L, R):
                draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=LENS, outline=FRAME, width=FW)
            bridge(L[0]+rx, R[0]-rx, L[1])

        elif style == "square":
            rx, ry, rad = 88, 58, 18
            for cx, cy in (L, R):
                box = [cx-rx, cy-ry, cx+rx, cy+ry]
                try:
                    draw.rounded_rectangle(box, radius=rad, fill=LENS, outline=FRAME, width=FW)
                except AttributeError:
                    draw.rectangle(box, fill=LENS, outline=FRAME, width=FW)
            bridge(L[0]+rx, R[0]-rx, L[1])

        elif style == "aviator":
            import math
            for cx, cy in (L, R):
                pts = []
                for deg in range(0, 360, 4):
                    a = math.radians(deg)
                    ex = 82 + 12 * math.cos(a + math.pi)
                    ey = 60 + 22 * math.sin(a)
                    pts.append((int(cx + ex * math.cos(a)),
                                int(cy + ey * math.sin(a))))
                draw.polygon(pts, fill=LENS, outline=FRAME)
            bridge(L[0]+90, R[0]-90, L[1]-5)

        elif style == "cateye":
            def cateye_lens(cx, cy, flip):
                pts = [
                    (cx - 88, cy + 52),
                    (cx - 90, cy + 5),
                    (cx + flip * 20, cy - 75),
                    (cx + 50, cy - 52),
                    (cx + 85, cy - 15),
                    (cx + 80, cy + 52),
                ]
                if flip == 1:
                    pts = [(2*cx - x, y) for x, y in pts]
                draw.polygon(pts, fill=LENS, outline=FRAME)
            cateye_lens(*L, flip=-1)
            cateye_lens(*R, flip=1)
            bridge(L[0]+85, R[0]-85, L[1])

        return img

    @staticmethod
    def _generate_custom_glasses(style, W=1200):
        """
        Dynamically generate a glasses image at high resolution for supersampling.
        Returns a tuple: (PILImage, lens_distance_in_pixels)
        """
        from PIL import Image as PILImage, ImageDraw
        import math

        H = int(W * 0.45)
        img = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        FRAME = (15, 15, 15, 255)
        LENS  = (50, 50, 50, 60)
        FW    = max(2, int(W * 0.015))  # Elegant, thinner wireframe thickness

        # A natural bridge is about 12% of the frame width
        bridge_w = W * 0.12

        def bridge(lx, rx_pos, y, drop):
            draw.line([(lx, y), (lx, y + drop)], fill=FRAME, width=FW)
            draw.line([(rx_pos, y), (rx_pos, y + drop)], fill=FRAME, width=FW)
            draw.line([(lx, y + drop), (rx_pos, y + drop)], fill=FRAME, width=FW)

        if style == "round" or style == "square":
            rx = (W - bridge_w) / 4.0
            L = (rx, H / 2.0)
            R = (W - rx, H / 2.0)
            lens_dist = R[0] - L[0]

            if style == "round":
                ry = rx * 0.9
                for cx, cy in (L, R):
                    draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=LENS, outline=FRAME, width=FW)
                bridge(L[0]+rx, R[0]-rx, L[1], drop=rx*0.2)
            else:
                ry = rx * 0.7
                rad = rx * 0.2
                for cx, cy in (L, R):
                    box = [cx-rx, cy-ry, cx+rx, cy+ry]
                    try:
                        draw.rounded_rectangle(box, radius=rad, fill=LENS, outline=FRAME, width=FW)
                    except AttributeError:
                        draw.rectangle(box, fill=LENS, outline=FRAME, width=FW)
                bridge(L[0]+rx, R[0]-rx, L[1], drop=rx*0.2)

        elif style == "aviator":
            # Aviator has teardrop lenses that lean outwards. 
            rx = (W - bridge_w) / 4.0
            L = (1.15 * rx, H / 2.0)
            R = (W - 1.15 * rx, H / 2.0)
            lens_dist = R[0] - L[0]
            
            def aviator_lens(cx, cy, flip):
                pts = []
                for deg in range(0, 360, 4):
                    a = math.radians(deg)
                    ex = rx - (rx * 0.15) * math.cos(a)
                    ey = (rx * 0.75) + (rx * 0.25) * math.sin(a)
                    px = ex * math.cos(a)
                    py = ey * math.sin(a)
                    if flip == -1:
                        px = -px 
                    pts.append((int(cx + px), int(cy + py)))
                draw.polygon(pts, fill=LENS)
                draw.line(pts + [pts[0]], fill=FRAME, width=FW, joint="curve")
            
            aviator_lens(L[0], L[1], flip=1)
            aviator_lens(R[0], R[1], flip=-1)
            
            # Bridge
            bridge(L[0] + 0.85*rx, R[0] - 0.85*rx, L[1] - rx*0.1, drop=rx*0.2)

        elif style == "cateye":
            # Cateye has swept up outer corners. 
            rx = (W - bridge_w) / 3.92
            L = (rx, H / 2.0)
            R = (W - rx, H / 2.0)
            lens_dist = R[0] - L[0]
            
            def cateye_lens(cx, cy, flip):
                pts = [
                    (cx - rx, cy + rx*0.6),               
                    (cx - rx*1.02, cy + rx*0.05),         
                    (cx - rx*0.25, cy - rx*0.85),         
                    (cx + rx*0.57, cy - rx*0.6),          
                    (cx + rx*0.96, cy - rx*0.17),         
                    (cx + rx*0.9, cy + rx*0.6),           
                ]
                if flip == -1:
                    pts = [(2*cx - x, y) for x, y in pts]
                draw.polygon(pts, fill=LENS)
                draw.line(pts + [pts[0]], fill=FRAME, width=FW, joint="curve")
            
            cateye_lens(L[0], L[1], flip=1)
            cateye_lens(R[0], R[1], flip=-1)
            
            bridge(L[0] + rx*0.96, R[0] - rx*0.96, L[1], drop=rx*0.2)

        return img, lens_dist

    def apply_hair_color(self, image, landmarks, color=(80, 40, 10), strength=0.6):
        """Hair recolor using MediaPipe SelfieMulticlassSegmentation. The category
        mask gives a binary 1/0 per pixel for the hair class. We dilate slightly
        to catch wispy outliers, then feather the edges before LAB recolor."""
        mask = get_hair_segmenter().segment(image)
        if not np.any(mask > 0.5):
            return image.copy()
        # Dilate to grab strand outliers the model may have just-barely missed
        binary = (mask > 0.5).astype(np.uint8) * 255
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
        mask = binary.astype(np.float32) / 255.0
        # Feather for natural edges
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
        return recolor_preserve_luminance(image, color, mask, strength=strength)

    def apply_eye_color(self, image, landmarks, color=(180, 100, 30), strength=0.85):
        """Recolor irises using MediaPipe FaceMesh iris landmarks (478-point set,
        indices 468-472 = left iris, 473-477 = right iris). LAB recolor preserves
        the dark pupil and the bright catchlight."""
        if not landmarks or len(landmarks) < 478:
            return image.copy()

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        for center_idx, ring in [(468, (469, 470, 471, 472)),
                                 (473, (474, 475, 476, 477))]:
            cx, cy = landmarks[center_idx]
            r_h = abs(landmarks[ring[0]][0] - cx)
            r_v = abs(landmarks[ring[1]][1] - cy)
            radius = max(2, int(round(max(r_h, r_v) * 0.95)))
            cv2.circle(mask, (int(cx), int(cy)), radius, 1.0, -1)

        if not np.any(mask):
            return image.copy()

        mask = cv2.GaussianBlur(mask, (0, 0), 0.8)
        return recolor_preserve_luminance(image, color, mask, strength=strength)


    def apply_frame_photo(self, image, landmarks=None, style="classic"):
        """
        Add a decorative photo frame around the image.

        Unlike the previous implementation that warped a gradient triangle
        onto the face, this creates proper picture frames by adding
        padding and drawing decorative borders AROUND the image.

        Styles
        ------
        classic  : ornate gold border with double line inset
        modern   : thin black frame with generous white mat
        polaroid : white border all around, extra space at bottom
        vintage  : dark wood-coloured border with inner gold accent

        Parameters
        ----------
        image  : the face image (BGR)
        landmarks : unused (kept for API compatibility)
        style  : one of 'classic', 'modern', 'polaroid', 'vintage'
        """
        h, w = image.shape[:2]
        dim = min(h, w)

        if style == "modern":
            # Clean thin black frame with white mat
            mat = max(12, int(dim * 0.06))
            frame = max(3, int(dim * 0.012))
            total = mat + frame
            canvas = np.full((h + total * 2, w + total * 2, 3), 255, dtype=np.uint8)
            # Place image centred
            canvas[total:total + h, total:total + w] = image
            # Outer black frame
            cv2.rectangle(canvas, (0, 0),
                          (canvas.shape[1] - 1, canvas.shape[0] - 1),
                          (30, 30, 30), frame)
            # Inner line just around the mat
            cv2.rectangle(canvas, (frame + mat - 1, frame + mat - 1),
                          (canvas.shape[1] - frame - mat,
                           canvas.shape[0] - frame - mat),
                          (180, 180, 180), 1)
            return canvas

        elif style == "polaroid":
            # Classic Polaroid: even white on top/left/right, larger at bottom
            pad_side = max(10, int(dim * 0.05))
            pad_bottom = max(30, int(dim * 0.16))
            ch = h + pad_side + pad_bottom
            cw = w + pad_side * 2
            canvas = np.full((ch, cw, 3), 250, dtype=np.uint8)
            canvas[pad_side:pad_side + h, pad_side:pad_side + w] = image
            # Subtle outer shadow line
            cv2.rectangle(canvas, (0, 0), (cw - 1, ch - 1), (210, 210, 210), 2)
            # Thin inner border around the photo
            cv2.rectangle(canvas, (pad_side - 1, pad_side - 1),
                          (pad_side + w, pad_side + h), (200, 200, 200), 1)
            return canvas

        elif style == "vintage":
            # Dark wood border with inner gold accent
            border = max(10, int(dim * 0.055))
            inner_line = max(2, int(dim * 0.008))
            total = border + inner_line + 2
            wood_color = (30, 42, 62)   # dark walnut (BGR)
            gold = (0, 180, 255)        # gold accent (BGR)
            canvas = np.full((h + total * 2, w + total * 2, 3),
                             wood_color[0], dtype=np.uint8)
            canvas[:, :, 0] = wood_color[0]
            canvas[:, :, 1] = wood_color[1]
            canvas[:, :, 2] = wood_color[2]
            canvas[total:total + h, total:total + w] = image
            # Gold inner accent
            cv2.rectangle(canvas,
                          (border, border),
                          (canvas.shape[1] - border - 1,
                           canvas.shape[0] - border - 1),
                          gold, inner_line)
            # Thin dark line right around the photo
            cv2.rectangle(canvas,
                          (total - 1, total - 1),
                          (total + w, total + h),
                          (20, 20, 20), 1)
            return canvas

        else:  # "classic" (default)
            # Ornate gold frame with double inset
            border = max(8, int(dim * 0.045))
            canvas_h = h + border * 2
            canvas_w = w + border * 2
            gold_outer = (0, 170, 240)
            gold_inner = (30, 200, 255)
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            # Fill frame area with gradient-like gold
            canvas[:, :] = gold_outer
            # Place image
            canvas[border:border + h, border:border + w] = image
            # Outer edge highlight
            cv2.rectangle(canvas, (1, 1),
                          (canvas_w - 2, canvas_h - 2),
                          (0, 140, 200), 2)
            # Inner edge
            inset = max(2, border // 3)
            cv2.rectangle(canvas,
                          (border - inset, border - inset),
                          (border + w + inset - 1, border + h + inset - 1),
                          gold_inner, max(1, inset // 2))
            # Shadow line right around the photo
            cv2.rectangle(canvas,
                          (border - 1, border - 1),
                          (border + w, border + h),
                          (20, 20, 20), 1)
            return canvas

