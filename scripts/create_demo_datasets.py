from __future__ import annotations

import csv
import io
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
TABULAR_DIR = ROOT / "data" / "tabular"
IMAGE_DIR = ROOT / "data" / "image_demo"
IMAGE_SIZE = 192


def create_tabular_datasets() -> None:
    TABULAR_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261109)
    rows = 1000
    target = np.tile(np.array([0, 1]), rows // 2)
    rng.shuffle(target)
    genuine_signal = (target * 2 - 1) + rng.normal(0, 1.15, size=rows)

    scenarios = [
        ("01_intended_signal_only.csv", None, "Intended signal only"),
        ("02_balanced_no_shortcut.csv", 0.50, "Background cue independent of class"),
        ("03_strong_shortcut.csv", 0.95, "Background cue agrees with class 95% of the time"),
        ("04_reversed_shortcut.csv", 0.05, "Background cue agrees with class 5% of the time"),
    ]
    for filename, agreement, _ in scenarios:
        frame = {
            "animal": np.where(target == 1, "wolf", "dog"),
            "genuine_signal": np.round(genuine_signal, 5),
        }
        if agreement is not None:
            shortcut_agrees = rng.random(rows) < agreement
            background_shortcut = np.where(shortcut_agrees, target, 1 - target)
            frame["background_shortcut"] = np.where(
                background_shortcut == 1, "snow", "green_field"
            )
        destination = TABULAR_DIR / filename
        columns = list(frame)
        with destination.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns)
            writer.writerows(zip(*(frame[column] for column in columns)))

    with (TABULAR_DIR / "README.txt").open("w", encoding="utf-8") as readme:
        readme.write(
            "MERIT generated tabular teaching data (not real animal measurements).\n"
            "All four files have 1,000 balanced dog/wolf labels and the same noisy\n"
            "genuine_signal construction. background_shortcut is a controlled\n"
            "categorical cue: wolf -> snow and dog -> green_field at the stated\n"
            "agreement rate. Values are generated with seed 20261109.\n\n"
            "Suggested comparison in MERIT: upload each file, select animal as\n"
            "target, select background_shortcut as the feature to investigate,\n"
            "and shuffle it on the held-out split. Compare baseline scores across\n"
            "the intended-only, independent, strong, and reversed scenarios.\n"
            "These data illustrate an experimental design; they are not evidence\n"
            "about real dogs, wolves, or computer-vision systems.\n"
        )


def _draw_background(draw: ImageDraw.ImageDraw, group: str, rng: random.Random) -> None:
    if group == "snow":
        draw.rectangle((0, 0, IMAGE_SIZE, IMAGE_SIZE), fill=(177, 210, 230))
        draw.rectangle((0, 112, IMAGE_SIZE, IMAGE_SIZE), fill=(239, 246, 249))
        for _ in range(150):
            x = rng.randrange(IMAGE_SIZE)
            y = rng.randrange(120, IMAGE_SIZE)
            radius = rng.choice((1, 1, 2))
            draw.ellipse((x, y, x + radius * 2, y + radius), fill=(219, 234, 241))
    elif group in {"green_field", "grassland"}:
        draw.rectangle((0, 0, IMAGE_SIZE, IMAGE_SIZE), fill=(180, 214, 231))
        draw.rectangle((0, 110, IMAGE_SIZE, IMAGE_SIZE), fill=(88, 143, 68))
        for _ in range(150):
            x = rng.randrange(IMAGE_SIZE)
            y = rng.randrange(112, IMAGE_SIZE)
            shade = rng.choice(((105, 159, 73), (69, 125, 57), (133, 165, 76)))
            draw.line((x, y, x + rng.randrange(-4, 5), y - rng.randrange(3, 10)), fill=shade)
    elif group == "lake":
        draw.rectangle((0, 0, IMAGE_SIZE, IMAGE_SIZE), fill=(179, 214, 231))
        draw.rectangle((0, 105, IMAGE_SIZE, IMAGE_SIZE), fill=(70, 145, 180))
        for _ in range(90):
            x = rng.randrange(IMAGE_SIZE)
            y = rng.randrange(108, IMAGE_SIZE)
            draw.line((x, y, min(IMAGE_SIZE, x + rng.randrange(5, 24)), y), fill=(130, 185, 202))
    elif group in {"forest", "desert"}:
        if group == "forest":
            draw.rectangle((0, 0, IMAGE_SIZE, IMAGE_SIZE), fill=(180, 214, 231))
            draw.rectangle((0, 112, IMAGE_SIZE, IMAGE_SIZE), fill=(71, 116, 65))
            for x in range(8, IMAGE_SIZE, 25):
                top = rng.randrange(22, 75)
                draw.polygon(
                    [(x, top), (x - 15, 110), (x + 15, 110)],
                    fill=rng.choice(((48, 101, 57), (58, 118, 63), (71, 126, 67))),
                )
        else:
            draw.rectangle((0, 0, IMAGE_SIZE, IMAGE_SIZE), fill=(231, 193, 139))
            draw.polygon(
                [(0, 100), (48, 85), (98, 102), (151, 82), (192, 99), (192, 192), (0, 192)],
                fill=(213, 168, 105),
            )
            draw.rectangle((0, 133, IMAGE_SIZE, IMAGE_SIZE), fill=(219, 178, 116))


def _draw_animal(draw: ImageDraw.ImageDraw, label: str, rng: random.Random) -> None:
    dark = rng.choice(((68, 58, 52), (91, 77, 65), (54, 60, 58)))
    light = (196, 177, 151)
    shift_x = rng.randint(-7, 7)
    shift_y = rng.randint(-5, 5)

    def point(x: int, y: int) -> tuple[int, int]:
        return x + shift_x, y + shift_y

    def box(x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        return (*point(x0, y0), *point(x1, y1))

    # These stylized shapes encode a simple intended cue; they are not photographs.
    if label == "wolf":
        draw.ellipse(box(44, 80, 139, 128), fill=dark)
        draw.polygon([point(117, 91), point(145, 65), point(150, 117), point(119, 124)], fill=dark)
        draw.ellipse(box(129, 50, 163, 83), fill=dark)
        draw.polygon([point(132, 58), point(136, 34), point(146, 57)], fill=dark)
        draw.polygon([point(151, 59), point(162, 39), point(163, 69)], fill=dark)
        draw.polygon([point(153, 66), point(180, 76), point(157, 84)], fill=dark)
        draw.polygon([point(57, 91), point(31, 82), point(16, 68), point(28, 99), point(55, 110)], fill=dark)
    elif label == "dog":
        draw.ellipse(box(43, 78, 143, 132), fill=dark)
        draw.ellipse(box(122, 82, 157, 113), fill=dark)
        draw.ellipse(box(138, 64, 170, 97), fill=dark)
        draw.ellipse(box(133, 55, 144, 76), fill=dark)
        draw.ellipse(box(151, 56, 167, 80), fill=dark)
        draw.ellipse(box(161, 72, 179, 84), fill=dark)
        draw.ellipse(box(171, 75, 181, 81), fill=light)
        draw.arc(box(13, 61, 58, 110), start=180, end=310, fill=dark, width=8)
    elif label in {"waterbird", "landbird"}:
        draw.ellipse(box(58, 78, 136, 124), fill=dark)
        draw.ellipse(box(117, 61, 149, 94), fill=dark)
        draw.polygon([point(146, 70), point(171, 78), point(146, 83)], fill=light)
        draw.polygon([point(84, 87), point(116, 66), point(110, 108)], fill=light)
        draw.line([point(81, 118), point(78, 151)], fill=dark, width=5)
        draw.line([point(111, 118), point(112, 151)], fill=dark, width=5)
        if label == "waterbird":
            draw.polygon([point(70, 145), point(58, 152), point(83, 152)], fill=dark)
            draw.polygon([point(104, 145), point(93, 152), point(119, 152)], fill=dark)
        else:
            draw.polygon([point(43, 101), point(21, 94), point(38, 111)], fill=dark)
    elif label in {"cow", "camel"}:
        draw.ellipse(box(48, 79, 139, 128), fill=dark)
        draw.polygon([point(124, 92), point(148, 57), point(157, 111), point(137, 118)], fill=dark)
        draw.ellipse(box(140, 48, 166, 75), fill=dark)
        draw.polygon([point(160, 57), point(181, 64), point(163, 72)], fill=dark)
        if label == "camel":
            draw.ellipse(box(68, 61, 113, 95), fill=dark)
            draw.polygon([point(144, 52), point(139, 42), point(150, 49)], fill=dark)
        else:
            draw.polygon([point(144, 51), point(137, 41), point(151, 49)], fill=light)
            draw.polygon([point(154, 50), point(164, 40), point(162, 55)], fill=light)
            draw.ellipse(box(82, 91, 98, 107), fill=light)
            draw.ellipse(box(108, 99, 123, 112), fill=light)
    else:
        raise ValueError(f"Unknown synthetic animal label: {label}")

    leg_y = rng.randint(119, 128)
    leg_positions = (61, 82, 117, 136) if label in {"dog", "wolf", "cow", "camel"} else ()
    for x in leg_positions:
        x += shift_x
        draw.rounded_rectangle(
            (x, leg_y + shift_y, x + rng.randint(8, 12), 168 + shift_y),
            radius=4,
            fill=dark,
        )


def create_synthetic_image_demo(
    filename: str,
    class_backgrounds: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = IMAGE_DIR / f"{filename}.zip"
    manifest_path = IMAGE_DIR / f"{filename}.csv"
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for label, common_background in class_backgrounds:
            alternate_background = {
                "green_field": "snow",
                "snow": "green_field",
                "lake": "forest",
                "forest": "lake",
                "grassland": "desert",
                "desert": "grassland",
            }[common_background]
            for background, count in ((common_background, 56), (alternate_background, 24)):
                for index in range(count):
                    image_rng = random.Random(f"20261109-{label}-{background}-{index}")
                    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
                    draw = ImageDraw.Draw(image)
                    _draw_background(draw, background, image_rng)
                    animal_layer = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0))
                    _draw_animal(ImageDraw.Draw(animal_layer), label, image_rng)
                    animal_bounds = animal_layer.getbbox()
                    if animal_bounds is None:
                        raise RuntimeError("The generated animal illustration is empty.")
                    animal_crop = animal_layer.crop(animal_bounds)
                    animal_crop.thumbnail((78, 64), Image.Resampling.LANCZOS)
                    x = (IMAGE_SIZE - animal_crop.width) // 2
                    y = 93 + image_rng.randint(-5, 5)
                    image.paste(animal_crop, (x, y), animal_crop)
                    image_path = f"images/{label}/{background}/{label}_{background}_{index:03d}.jpg"
                    encoded = io.BytesIO()
                    image.save(encoded, format="JPEG", quality=86)
                    archive.writestr(image_path, encoded.getvalue())
                    rows.append(
                        {
                            "image_path": image_path,
                            "label": label,
                            "background_group": background,
                        }
                    )

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "label", "background_group"])
        writer.writeheader()
        writer.writerows(rows)

    (IMAGE_DIR / "README.txt").write_text(
        "All synthetic_*.zip archives contain generated illustrations, not photos.\n"
        "Each archive has 160 images: 80 of each stylized class; each class has 56\n"
        "images on its common background and 24 on the alternate background. The\n"
        "intended silhouette cue is deliberately small while background color and\n"
        "texture are prominent. These demonstrate the app workflow only; they are\n"
        "not real research images or evidence about actual animal recognition.\n"
        "See the project README for real benchmark sources and limitations.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    create_tabular_datasets()
    create_synthetic_image_demo(
        "synthetic_dogs_wolves", (("dog", "green_field"), ("wolf", "snow"))
    )
    create_synthetic_image_demo(
        "synthetic_waterbirds", (("waterbird", "lake"), ("landbird", "forest"))
    )
    create_synthetic_image_demo(
        "synthetic_cows_camels", (("cow", "grassland"), ("camel", "desert"))
    )
    print(f"Created tabular demos in {TABULAR_DIR}")
    print(f"Created image demo in {IMAGE_DIR}")
