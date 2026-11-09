
import io
import os
import threading
import time
import math

import numpy as np
import cv2
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

# ==========================================
# 0. SETTINGS
# ==========================================

_here = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(_here, path)


BACKEND = os.environ.get("EMOTION_BACKEND", "keras")
MODEL_PATH = resolve_path(
    os.environ.get("MODEL_PATH", "autism_emotion_model.keras")
)
HEADS_PATH = resolve_path(
    os.environ.get("HEADS_PATH", "autism_emotion_heads.joblib")
)

CASCADE_PATH = resolve_path(
    os.environ.get(
        "CASCADE_PATH",
        "haarcascade_frontalface_default.xml"
    )
)

CLASS_NAMES = ["Natural", "anger", "fear", "joy", "sadness", "surprise"]

NEGATIVE_THRESHOLD = 50.0
MIN_FACE_PX = 60

CONFIRM_MIN_NEGATIVE = 3
CONFIRM_MIN_VALID = 3

SAVE_PHOTOS = os.environ.get("SAVE_PHOTOS", "1") != "0"

SAVE_DIR = resolve_path(
    os.environ.get("SAVE_DIR", "received_photos")
)

if SAVE_PHOTOS:
    os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================
# 1. SHARED SERVER STATE
# ==========================================

# Only /confirm is allowed to update this decision.
latest_confirmed_state = {
    "iot_command": "X",
    "decision": "inconclusive",
    "is_sensory_overload": None,
    "actuator_control": None
}

# Latest sensor readings from ESP32 DevKit.
latest_sensor_data = None

state_lock = threading.Lock()
sensor_lock = threading.Lock()
predict_lock = threading.Lock()


# ==========================================
# 2. FACE DETECTOR
# ==========================================

if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(
        f"Face detector not found: {CASCADE_PATH}"
    )

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade.")

print("Face detector loaded:", CASCADE_PATH)


# ==========================================
# 3. LOAD EMOTION MODEL
# ==========================================

if BACKEND == "keras":

    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import preprocess_input

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("Loading VGG16 emotion model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    _has_preprocess_inside = any(
        "subtract" in layer.name
        or "getitem" in layer.name
        or "get_item" in layer.name
        or layer.name.startswith("preproc_inside")
        for layer in model.layers
    )

    print(
        "Model loaded. Internal preprocessing:",
        _has_preprocess_inside
    )

    def predict_probs(rgb224):
        x = rgb224[None].astype("float32")

        if not _has_preprocess_inside:
            x = preprocess_input(x)

        return model.predict(x, verbose=0)[0]


elif BACKEND == "heads":

    import joblib
    import tensorflow as tf
    from sklearn.preprocessing import normalize

    if not os.path.exists(HEADS_PATH):
        raise FileNotFoundError(
            f"Classifier heads not found: {HEADS_PATH}"
        )

    _bundle = joblib.load(HEADS_PATH)
    _heads = _bundle["heads"]

    if list(_bundle["class_names"]) != CLASS_NAMES:
        raise RuntimeError("Model class order mismatch.")

    K, A, L = tf.keras, tf.keras.applications, tf.keras.layers

    _xc = A.Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    _xc_feat = K.Model(
        _xc.input,
        [
            L.GlobalAveragePooling2D()(_xc.get_layer(n).output)
            for n in ("add_5", "add_8", "block14_sepconv2_act")
        ]
    )

    _vg = A.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    _vg_feat = K.Model(
        _vg.input,
        [
            L.GlobalAveragePooling2D()(_vg.get_layer(n).output)
            for n in ("block4_pool", "block5_pool")
        ]
    )

    def _extract(xb):
        r = (
            _xc_feat.predict(
                A.xception.preprocess_input(xb.copy()),
                verbose=0
            )
            +
            _vg_feat.predict(
                A.vgg16.preprocess_input(xb.copy()),
                verbose=0
            )
        )

        return np.hstack([normalize(v) for v in r])

    def predict_probs(rgb224):
        x = rgb224[None].astype("float32")

        F = _extract(x)
        Ff = _extract(x[:, :, ::-1].copy())

        sc, svm, lr = _heads

        ps = [
            m.predict_proba(sc.transform(a))
            for m in (svm, lr)
            for a in (F, Ff)
        ]

        return np.mean(ps, axis=0)[0]


else:
    raise ValueError(
        "EMOTION_BACKEND must be 'keras' or 'heads'"
    )


print("Emotion backend:", BACKEND)


# ==========================================
# 4. FACE DETECTION
# ==========================================

def find_face(pil_img):

    w, h = pil_img.size

    gray = cv2.cvtColor(
        np.array(pil_img),
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(MIN_FACE_PX, MIN_FACE_PX)
    )

    if len(faces) == 0:
        return None

    fx, fy, fw, fh = max(
        faces,
        key=lambda b: b[2] * b[3]
    )

    pad_x = int(fw * 0.10)
    pad_y = int(fh * 0.10)

    return pil_img.crop((
        max(0, fx - pad_x),
        max(0, fy - pad_y),
        min(w, fx + fw + pad_x),
        min(h, fy + fh + pad_y)
    ))


# ==========================================
# 5. EMOTION RESULT
# ==========================================

def build_result(probs):

    top_idx = int(np.argmax(probs))

    negative_score = float(
        probs[1] + probs[2] + probs[4]
    ) * 100

    positive_score = float(
        probs[0] + probs[3] + probs[5]
    ) * 100

    is_overload = negative_score >= NEGATIVE_THRESHOLD

    return {
        "status": "success",
        "face_detected": True,
        "detected_emotion": CLASS_NAMES[top_idx],
        "confidence": f"{float(probs[top_idx]) * 100:.2f}%",
        "negative_distress_score": f"{negative_score:.2f}%",
        "positive_calm_score": f"{positive_score:.2f}%",
        "is_sensory_overload": is_overload,
        "emotion": CLASS_NAMES[top_idx],
        "negative": is_overload,
        "iot_command": "O" if is_overload else "N",
        "actuator_control": {
            "light_pwm": 40 if is_overload else 180,
            "sound_mode": (
                "CALMING_WHITE_NOISE"
                if is_overload
                else "NORMAL_DAYLIGHT"
            )
        }
    }


def no_face_result():

    return {
        "status": "no_face",
        "face_detected": False,
        "detected_emotion": None,
        "is_sensory_overload": False,
        "emotion": None,
        "negative": False,
        "iot_command": "X",
        "actuator_control": None
    }


# ==========================================
# 6. PHOTO STORAGE
# ==========================================

def save_photo(image_bytes, prefix="predict"):

    if not SAVE_PHOTOS:
        return None

    try:

        ext = "jpg"

        try:
            fmt = Image.open(
                io.BytesIO(image_bytes)
            ).format

            if fmt:
                ext = fmt.lower().replace("jpeg", "jpg")

        except Exception:
            pass

        ts = time.strftime("%Y%m%d-%H%M%S")
        ms = int((time.time() % 1) * 1000)

        filename = f"{ts}-{ms:03d}_{prefix}.{ext}"

        path = os.path.join(SAVE_DIR, filename)

        with open(path, "wb") as f:
            f.write(image_bytes)

        return path

    except Exception as e:
        print("Photo saving failed:", e)
        return None


# ==========================================
# 7. IMAGE ANALYSIS
# ==========================================

def analyze(image_bytes, force=False, save_prefix="predict"):

    saved_path = save_photo(
        image_bytes,
        prefix=save_prefix
    )

    try:

        pil_img = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as e:

        print("Image processing error:", e)
        return None

    face = pil_img if force else find_face(pil_img)

    if face is None:

        result = no_face_result()

    else:

        rgb224 = np.array(
            face.resize(
                (224, 224),
                Image.Resampling.BILINEAR
            ),
            dtype=np.uint8
        )

        with predict_lock:
            probs = predict_probs(rgb224)

        result = build_result(probs)

    if saved_path:
        result["saved_photo"] = saved_path

    return result


# ==========================================
# 8. GET /
# SERVER HEALTH
# ==========================================

@app.route("/", methods=["GET"])
def health_check():

    return jsonify({
        "message": "Autism Emotion Recognition IoT Server is active!",
        "backend": BACKEND,
        "save_photos": SAVE_PHOTOS,
        "endpoints": {
            "GET /": "Server health",
            "POST /predict": "Single image prediction",
            "POST /confirm": "3-of-5 confirmation",
            "GET /state": "Latest confirmed actuator state",
            "POST /sensor": "Receive ESP32 sensor readings"
        }
    })


# ==========================================
# 9. POST /predict
# SINGLE IMAGE PREDICTION
# DOES NOT CHANGE ACTUATOR STATE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" in request.files:
        image_bytes = request.files["image"].read()

    elif request.data:
        image_bytes = request.data

    else:
        return jsonify({
            "error": "No image data provided"
        }), 400

    result = analyze(
        image_bytes,
        force=request.args.get("force") == "1",
        save_prefix="predict"
    )

    if result is None:
        return jsonify({
            "error": "Invalid or corrupted image"
        }), 400

    print(
        f"Predict -> emotion={result['detected_emotion']} "
        f"status={result['status']}"
    )

    return jsonify(result), 200


# ==========================================
# 10. POST /confirm
# 3-OF-5 DECISION
# ONLY THIS ENDPOINT UPDATES ACTUATOR STATE
# ==========================================

@app.route("/confirm", methods=["POST"])
def confirm():

    global latest_confirmed_state

    files = request.files.getlist("image")

    if not files:
        return jsonify({
            "error": "Send images as repeated form-data field 'image'"
        }), 400

    frames = []
    negatives = 0
    valid = 0

    for i, f in enumerate(files[:5]):

        r = analyze(
            f.read(),
            save_prefix=f"confirm{i + 1}"
        )

        if r is None:
            frames.append({
                "status": "invalid_image"
            })
            continue

        if r["status"] == "success":

            valid += 1
            negatives += int(r["negative"])

        frames.append({
            k: r.get(k)
            for k in (
                "status",
                "detected_emotion",
                "confidence",
                "negative_distress_score",
                "negative",
                "saved_photo"
            )
        })

    # Make the 3-of-5 decision.

    if valid < CONFIRM_MIN_VALID:
        decision = "inconclusive"

    elif negatives >= CONFIRM_MIN_NEGATIVE:
        decision = "overload"

    else:
        decision = "normal"

    overload = decision == "overload"

    # Only conclusive decisions update the shared state.

    if decision in ("overload", "normal"):

        new_state = {
            "iot_command": "O" if overload else "N",
            "decision": decision,
            "is_sensory_overload": overload,
            "actuator_control": (
                {
                    "light_pwm": 40,
                    "sound_mode": "CALMING_WHITE_NOISE"
                }
                if overload
                else {
                    "light_pwm": 180,
                    "sound_mode": "NORMAL_DAYLIGHT"
                }
            )
        }

        with state_lock:
            latest_confirmed_state = new_state.copy()

    # If inconclusive, preserve the previous confirmed state.

    with state_lock:
        current_state = latest_confirmed_state.copy()

    output = {
        "status": "success",
        "decision": decision,
        "negative_frames": negatives,
        "frames_with_face": valid,
        "frames_received": len(frames),
        "is_sensory_overload": (
            overload if decision != "inconclusive" else None
        ),
        "negative": (
            overload if decision != "inconclusive" else None
        ),
        "iot_command": (
            "O" if overload
            else "N" if decision == "normal"
            else "X"
        ),
        "actuator_control": (
            current_state["actuator_control"]
            if decision != "inconclusive"
            else None
        ),
        "frames": frames,
        "latest_confirmed_state": current_state
    }

    print(
        f"Confirm -> {decision} "
        f"({negatives} negative of {valid} valid frames)"
    )

    return jsonify(output), 200


# ==========================================
# 11. GET /state
# ESP32 DEVKIT POLLS THIS ENDPOINT
# ==========================================

@app.route("/state", methods=["GET"])
def get_state():

    with state_lock:
        current_state = latest_confirmed_state.copy()

    return jsonify({
        "status": "success",
        **current_state
    }), 200


# ==========================================
# 12. POST /sensor
# ESP32 DEVKIT SENDS LDR + SOUND DATA
# ==========================================

@app.route("/sensor", methods=["POST"])
def receive_sensor():

    global latest_sensor_data

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Expected JSON request body"
        }), 400

    light = data.get("light")
    sound = data.get("sound")
    device_state = data.get("state")
    stale = data.get("stale")

    if (
        isinstance(light, bool)
        or not isinstance(light, (int, float))
        or not math.isfinite(light)
    ):
        return jsonify({
            "error": "Invalid light value"
        }), 400

    if (
        isinstance(sound, bool)
        or not isinstance(sound, (int, float))
        or not math.isfinite(sound)
    ):
        return jsonify({
            "error": "Invalid sound value"
        }), 400

    if device_state not in ("normal", "overload"):
        return jsonify({
            "error": "state must be normal or overload"
        }), 400

    if not isinstance(stale, bool):
        return jsonify({
            "error": "stale must be true or false"
        }), 400

    sensor_data = {
        "light": float(light),
        "sound": float(sound),
        "state": device_state,
        "stale": stale,
        "received_at": time.time()
    }

    with sensor_lock:
        latest_sensor_data = sensor_data

    print(
        f"Sensor -> light={light}, "
        f"sound={sound}, "
        f"state={device_state}, "
        f"stale={stale}"
    )

    return jsonify({
        "status": "success",
        "message": "Sensor data received"
    }), 200


# ==========================================
# 13. LOCAL SERVER START
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("SafeSense Flask server starting...")
    print("Local address: http://127.0.0.1:5000")
    print("=" * 55)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
