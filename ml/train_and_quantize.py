"""Train and export the 64-4-1 Cat/Dog model used by the RTL core."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

SEED = 42
INPUT_SCALE = 1.0 / 8.0
OUT_DIR = Path(__file__).resolve().parent / "artifacts"


def preprocess(example, cat_label):
    image = tf.image.rgb_to_grayscale(example["image"])
    image = tf.image.resize_with_pad(image, 8, 8)
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    image = tf.reshape(image, [64])
    label = tf.cast(tf.equal(example["label"], cat_label), tf.float32)
    return image, label


def make_dataset(split, cat_label, training):
    dataset = tfds.load("cats_vs_dogs", split=split, shuffle_files=training)
    dataset = dataset.map(
        lambda item: preprocess(item, cat_label), num_parallel_calls=tf.data.AUTOTUNE
    )
    if training:
        dataset = dataset.shuffle(4096, seed=SEED)
    return dataset.batch(128).prefetch(tf.data.AUTOTUNE)


def collect(dataset):
    xs, ys = [], []
    for x, y in dataset:
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys).astype(np.int64)


def quantize(values, scale, low, high):
    return np.clip(np.rint(values / scale), low, high).astype(np.int64)


def scale_candidates(weights):
    base = max(float(np.max(np.abs(weights))) / 7.0, 1e-8)
    return [base * factor for factor in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]


def integer_inference(xq, w1q, b1q, shift1, w2q, b2q, shift2):
    acc1 = xq @ w1q + b1q
    hidden = np.clip(np.right_shift(np.maximum(acc1, 0), shift1), 0, 7)
    acc2 = hidden @ w2q + b2q
    score = np.clip(np.right_shift(acc2, shift2), -8, 7)
    prediction = (acc2 >= 0).astype(np.int64)
    return score, prediction, acc1, acc2


def accumulator_bounds(weights, bias, input_min, input_max):
    """Return exact worst-case bounds for independent bounded inputs."""
    low_terms = np.minimum(weights * input_min, weights * input_max)
    high_terms = np.maximum(weights * input_min, weights * input_max)
    return low_terms.sum(axis=0) + bias, high_terms.sum(axis=0) + bias


def find_integer_model(model, x_val, y_val):
    w1, b1 = model.layers[0].get_weights()
    w2, b2 = model.layers[1].get_weights()
    xq = quantize(x_val, INPUT_SCALE, -8, 7)
    best = None

    for w1_scale in scale_candidates(w1):
        w1q = quantize(w1, w1_scale, -8, 7)
        b1q = quantize(b1, INPUT_SCALE * w1_scale, -128, 127)
        for shift1 in range(11):
            hidden_scale = INPUT_SCALE * w1_scale * (2 ** shift1)
            for w2_scale in scale_candidates(w2):
                w2q = quantize(w2, w2_scale, -8, 7)
                b2q = quantize(b2, hidden_scale * w2_scale, -128, 127)
                for shift2 in range(5):
                    result = integer_inference(
                        xq, w1q, b1q, shift1, w2q, b2q, shift2
                    )
                    accuracy = float(np.mean(result[1].reshape(-1) == y_val.reshape(-1)))
                    if best is None or accuracy > best["accuracy"]:
                        best = {"accuracy": accuracy, "w1q": w1q, "b1q": b1q,
                                "shift1": shift1, "w2q": w2q, "b2q": b2q,
                                "shift2": shift2, "w1_scale": w1_scale,
                                "w2_scale": w2_scale, "hidden_scale": hidden_scale}
    return best


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    label_names = list(tfds.builder("cats_vs_dogs").info.features["label"].names)
    cat_label = label_names.index("cat")
    train_ds = make_dataset("train[:70%]", cat_label, True)
    val_ds = make_dataset("train[70%:85%]", cat_label, False)
    test_ds = make_dataset("train[85%:]", cat_label, False)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(64,)),
        tf.keras.layers.Dense(4, activation="relu", name="hidden"),
        tf.keras.layers.Dense(1, activation=None, name="score"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.0, name="accuracy")],
    )
    model.fit(
        train_ds, validation_data=val_ds, epochs=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, mode="max", restore_best_weights=True
        )],
    )
    _, fp_accuracy = model.evaluate(test_ds, verbose=0)
    model.save(OUT_DIR / "catdog_64_4_1.keras")

    x_val, y_val = collect(val_ds)
    chosen = find_integer_model(model, x_val, y_val)
    x_test, y_test = collect(test_ds)
    xq_test = quantize(x_test, INPUT_SCALE, -8, 7)
    _, prediction, _, _ = integer_inference(
        xq_test, chosen["w1q"], chosen["b1q"], chosen["shift1"],
        chosen["w2q"], chosen["b2q"], chosen["shift2"]
    )
    int_accuracy = float(np.mean(prediction.reshape(-1) == y_test.reshape(-1)))

    min1, max1 = accumulator_bounds(chosen["w1q"], chosen["b1q"], -8, 7)
    min2, max2 = accumulator_bounds(chosen["w2q"], chosen["b2q"], 0, 7)
    theoretical_min = int(min(np.min(min1), np.min(min2)))
    theoretical_max = int(max(np.max(max1), np.max(max2)))
    if theoretical_min < -32768 or theoretical_max > 32767:
        raise RuntimeError("Trained integer model exceeds the signed 16-bit accumulator")

    export = {
        "project": "CatDog TinyNPU 64-4-1",
        "classes": {"0": "dog", "1": "cat"},
        "input": {"width": 8, "height": 8, "channels": 1,
                  "quantization": "signed_int4", "scale": INPUT_SCALE},
        "layer1": {"weights": chosen["w1q"].T.tolist(),
                   "bias": chosen["b1q"].tolist(),
                   "shift": int(chosen["shift1"]), "activation": "relu"},
        "layer2": {"weights": chosen["w2q"].T.tolist(),
                   "bias": chosen["b2q"].tolist(),
                   "shift": int(chosen["shift2"]), "activation": "linear"},
        "decision": "raw_output_accumulator >= 0 means cat; otherwise dog",
        "metrics": {"fp32_test_accuracy": float(fp_accuracy),
                    "int4_validation_accuracy": chosen["accuracy"],
                    "int4_test_accuracy": int_accuracy,
                    "theoretical_accumulator_min": theoretical_min,
                    "theoretical_accumulator_max": theoretical_max},
    }
    output_path = OUT_DIR / "model_int4.json"
    output_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    print(json.dumps(export["metrics"], indent=2))
    print(f"Exported {output_path}")


if __name__ == "__main__":
    main()
