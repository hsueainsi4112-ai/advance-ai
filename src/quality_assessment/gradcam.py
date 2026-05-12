import base64
import io

import cv2
import numpy as np
from PIL import Image

_IMG_SIZE = 224
_FEATURE_LAYER = "resnet50"


def _preprocess_for_resnet(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf

    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img_pil.size
    if w <= h:
        new_w, new_h = _IMG_SIZE, int(round(h * _IMG_SIZE / w))
    else:
        new_h, new_w = _IMG_SIZE, int(round(w * _IMG_SIZE / h))
    resized = np.array(img_pil.resize((new_w, new_h), Image.LANCZOS), dtype=np.float32)
    off_h = (new_h - _IMG_SIZE) // 2
    off_w = (new_w - _IMG_SIZE) // 2
    crop = resized[off_h:off_h + _IMG_SIZE, off_w:off_w + _IMG_SIZE]
    batch = crop[np.newaxis, ...]
    resnet_input = tf.keras.applications.resnet50.preprocess_input(batch.copy())
    return crop, resnet_input


def compute_heatmap(
    image_bytes: bytes,
    predicted_class: str,
    *,
    resnet,
    class_index: int,
) -> dict:
    import tensorflow as tf

    crop, resnet_input = _preprocess_for_resnet(image_bytes)

    inner_resnet = resnet.get_layer(_FEATURE_LAYER)
    feature_idx = next(i for i, lyr in enumerate(resnet.layers) if lyr.name == _FEATURE_LAYER)
    head_layers = resnet.layers[feature_idx + 1:]

    input_tensor = tf.convert_to_tensor(resnet_input)
    with tf.GradientTape() as tape:
        conv_out = inner_resnet(input_tensor, training=False)
        tape.watch(conv_out)
        x = conv_out
        for layer in head_layers:
            x = layer(x, training=False)
        target = x[:, class_index]
    grads = tape.gradient(target, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out_0 = conv_out[0]
    heatmap = tf.reduce_sum(conv_out_0 * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap_np = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap_np, (_IMG_SIZE, _IMG_SIZE))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colour = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    crop_bgr = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(crop_bgr, 0.6, colour, 0.4, 0)

    _, buf = cv2.imencode(".png", overlay)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "heatmap_base64": b64,
        "predicted_class": predicted_class,
        "explanation": (
            f"Highlighted regions show the image areas most influential for the "
            f"{predicted_class} classification by the ResNet50 branch."
        ),
    }
