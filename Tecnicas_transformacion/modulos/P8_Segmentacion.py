"""
P8_Segmentacion.py - Segmentación por Template Matching
"""

import cv2
import numpy as np


def _to_gray(img):
    """Convierte imagen (GRAY/BGR/BGRA) a escala de grises de forma segura."""
    if img is None:
        raise ValueError("Imagen None")
    if len(img.shape) == 2:
        return img
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            return img[:, :, 0]
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape[2] == 4:
            # BGRA -> GRAY
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Formato de imagen no soportado: shape={img.shape}")


def _nms_boxes(boxes, scores, iou_threshold=0.3):
    """
    Non-Maximum Suppression simple por IoU.
    boxes: list[(x1,y1,x2,y2)]
    scores: list[float]
    """
    if not boxes:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def template_matching(image_bgr, template_bgr, method='TM_CCOEFF_NORMED', threshold=0.7, nms_iou=0.3):
    img_gray = _to_gray(image_bgr)
    template_gray = _to_gray(template_bgr)

    if template_gray.shape[0] > img_gray.shape[0] or template_gray.shape[1] > img_gray.shape[1]:
        raise ValueError("La plantilla no puede ser más grande que la imagen de búsqueda")

    method_map = {
        'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
        'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED,
        'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
    }
    cv_method = method_map.get(method, cv2.TM_CCOEFF_NORMED)

    result = cv2.matchTemplate(img_gray, template_gray, cv_method)

    h, w = template_gray.shape[:2]

    # Obtener candidatos
    if method in ['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED']:
        ys, xs = np.where(result >= threshold)
        scores = result[ys, xs]
    else:
        # TM_SQDIFF_NORMED: menor es mejor. threshold=0.8 => aceptar <= 0.2
        ys, xs = np.where(result <= (1.0 - threshold))
        scores = 1.0 - result[ys, xs]  # lo convertimos a "similitud" para ordenar/dibujar

    # Convertir a cajas y aplicar NMS
    boxes = []
    score_list = []
    match_list = []

    for x, y, s in zip(xs, ys, scores):
        boxes.append((x, y, x + w, y + h))
        score_list.append(float(s))

    keep = _nms_boxes(boxes, score_list, iou_threshold=nms_iou)

    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        s = score_list[i]
        match_list.append((int(x1), int(y1), float(s)))

    # Dibujar
    result_img = image_bgr.copy()
    for x, y, val in match_list:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_img, f"{val:.2f}", (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return result_img, match_list


def template_matching_multiscale(
    image_bgr, template_bgr,
    scales=(0.5, 0.75, 1.0, 1.25, 1.5),
    method='TM_CCOEFF_NORMED',
    threshold=0.7,
    nms_iou=0.3
):
    img_gray = _to_gray(image_bgr)
    template_gray = _to_gray(template_bgr)

    method_map = {
        'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
        'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED,
        'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
    }
    cv_method = method_map.get(method, cv2.TM_CCOEFF_NORMED)

    result_img = image_bgr.copy()
    all_matches = []

    h0, w0 = template_gray.shape[:2]
    max_scale = max(scales) if len(scales) else 1.0

    # Para NMS global entre escalas
    boxes = []
    score_list = []
    meta = []  # (x,y,score,scale,w,h)

    for scale in scales:
        new_h, new_w = int(h0 * scale), int(w0 * scale)
        if new_h <= 0 or new_w <= 0:
            continue
        if new_h > img_gray.shape[0] or new_w > img_gray.shape[1]:
            continue

        template_scaled = cv2.resize(template_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(img_gray, template_scaled, cv_method)

        if method in ['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED']:
            ys, xs = np.where(result >= threshold)
            scores = result[ys, xs]
        else:
            ys, xs = np.where(result <= (1.0 - threshold))
            scores = 1.0 - result[ys, xs]

        for x, y, s in zip(xs, ys, scores):
            boxes.append((x, y, x + new_w, y + new_h))
            score_list.append(float(s))
            meta.append((int(x), int(y), float(s), float(scale), int(new_w), int(new_h)))

    keep = _nms_boxes(boxes, score_list, iou_threshold=nms_iou)

    for i in keep:
        x, y, s, scale, ww, hh = meta[i]
        all_matches.append((x, y, s, scale))
        c1 = int(np.clip(255 * (scale / max_scale), 0, 255))
        c3 = int(np.clip(255 - c1, 0, 255))
        color = (c1, 100, c3)
        cv2.rectangle(result_img, (x, y), (x + ww, y + hh), color, 2)

    return result_img, all_matches


def feature_matching(image_bgr, template_bgr, method='SIFT'):
    img_gray = _to_gray(image_bgr)
    template_gray = _to_gray(template_bgr)

    if method == 'SIFT':
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2
    elif method == 'ORB':
        detector = cv2.ORB_create(nfeatures=2000)
        norm = cv2.NORM_HAMMING
    else:
        detector = cv2.SIFT_create()
        norm = cv2.NORM_L2

    kp1, des1 = detector.detectAndCompute(img_gray, None)       # imagen grande
    kp2, des2 = detector.detectAndCompute(template_gray, None)  # template

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return image_bgr.copy(), 0

    bf = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(des2, des1, k=2)

    good = []
    ratio = 0.75 if method == 'SIFT' else 0.85  # ORB suele requerir ratio menos estricto
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

    result_img = cv2.drawMatches(
        template_gray, kp2, img_gray, kp1, good,
        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    if len(result_img.shape) == 2:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)

    return result_img, len(good)


def contour_matching(image_bgr, template_bgr, threshold=0.6):
    img_gray = _to_gray(image_bgr)
    template_gray = _to_gray(template_bgr)

    # Binarización robusta (Otsu)
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, template_bin = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    template_contours, _ = cv2.findContours(template_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not template_contours:
        return image_bgr.copy(), []

    template_cnt = max(template_contours, key=cv2.contourArea)

    result_img = image_bgr.copy()
    matches = []

    # matchShapes: 0 es perfecto, mayor = peor
    # Convertimos threshold (0..1) a un max_error razonable
    max_error = (1.0 - threshold) * 5.0

    for contour in img_contours:
        if cv2.contourArea(contour) < 10:  # filtrar ruido pequeño
            continue

        match_value = cv2.matchShapes(template_cnt, contour, cv2.CONTOURS_MATCH_I3, 0)
        if match_value <= max_error:
            matches.append((contour, float(match_value)))
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return result_img, matches
