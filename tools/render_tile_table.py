"""Draw the tile table over the tile images, to check it by eye.

Reads the JSON that dump_tiles writes and, for every tile type, draws on
tiles/<type>.png (the picture in rotation 0, north up):

  - each side as a coloured bar (green grass, orange city, white road, blue
    river) labelled with its link number: sides with the same number are one
    feature on this tile;
  - the field number of each of the 8 half-edges in a coloured dot, near that
    half of its side; the inner field, if any, in the centre;
  - in yellow, next to the side they are written on, the marks of a city or
    road: SH shield, PR princess, WI / CL / WH goods, INN inn;
  - under the picture: the copies, the tile marks (monastery, dragon, volcano,
    portal), and the city sides each field borders.

One sheet per expansion, results/tile_sheet_<expansion>.png. See
docs/adding_tiles.md.

Usage: python render_tile_table.py [tile_table.json] [--tiles DIR] [--out DIR]
"""

import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

CELL = 240        # picture size on the sheet
CAPTION = 72      # text under each picture
GAP = 12
COLUMNS = 6
BAR = 12          # side bar thickness

EDGE_COLOURS = {
    "GRASS": (60, 170, 60),
    "CITY": (230, 120, 20),
    "ROAD": (245, 245, 245),
    "RIVER": (40, 120, 230),
    "NONE": (255, 0, 255),
}
EDGE_LETTERS = {"GRASS": "G", "CITY": "C", "ROAD": "R", "RIVER": "W", "NONE": "?"}
FIELD_COLOURS = [(220, 40, 40), (40, 90, 220), (200, 160, 0), (150, 40, 190)]
SIDE_NAMES = "NESW"
# Short name -> bit, read from the JSON (dump_tiles writes them from tile.hpp).
MARK_BITS = {}
TILE_MARK_BITS = {}


def load_font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def half_edge_point(e, inset):
    """Where half-edge e (0 N-west, 1 N-east, ... 7 W-north) is drawn."""
    side, half = divmod(e, 2)
    near, far = CELL * 0.33, CELL * 0.67
    if side == 0:
        return (near if half == 0 else far, inset)
    if side == 1:
        return (CELL - inset, near if half == 0 else far)
    if side == 2:
        return (far if half == 0 else near, CELL - inset)
    return (inset, far if half == 0 else near)


def text_centered(draw, xy, text, font, fill, stroke=(0, 0, 0)):
    draw.text(xy, text, font=font, fill=fill, anchor="mm", stroke_width=2, stroke_fill=stroke)


def draw_tile(tile, tiles_dir, fonts):
    big, small = fonts
    canvas = Image.new("RGB", (CELL, CELL + CAPTION), "white")
    path = os.path.join(tiles_dir, f"{tile['type']}.png")
    if os.path.exists(path):
        picture = Image.open(path).convert("RGB").resize((CELL, CELL))
        # Dimmed, so the marks stand out.
        picture = Image.blend(picture, Image.new("RGB", picture.size, "black"), 0.25)
        canvas.paste(picture, (0, 0))
    else:
        canvas.paste((200, 200, 200), (0, 0, CELL, CELL))
    draw = ImageDraw.Draw(canvas)
    if not os.path.exists(path):
        text_centered(draw, (CELL / 2, CELL / 2), f"no tiles/{tile['type']}.png", small, (255, 255, 255))

    # Sides, as trapezoids so each corner is split between its two sides.
    c, b = CELL, BAR
    bars = [[(0, 0), (c, 0), (c - b, b), (b, b)], [(c, 0), (c, c), (c - b, c - b), (c - b, b)],
            [(c, c), (0, c), (b, c - b), (c - b, c - b)], [(0, c), (0, 0), (b, b), (b, c - b)]]
    for side, edge in enumerate(tile["edges"]):
        draw.polygon(bars[side], fill=EDGE_COLOURS[edge], outline=(0, 0, 0))
    label_points = [(CELL / 2, BAR + 12), (CELL - BAR - 14, CELL / 2), (CELL / 2, CELL - BAR - 12),
                    (BAR + 14, CELL / 2)]
    for side, edge in enumerate(tile["edges"]):
        text_centered(draw, label_points[side], f"{EDGE_LETTERS[edge]}{tile['links'][side]}", big,
                      EDGE_COLOURS[edge])

    # Marks, as written, next to the side they are written on.
    mark_points = [(CELL / 2, BAR + 40), (CELL - BAR - 46, CELL / 2), (CELL / 2, CELL - BAR - 40),
                   (BAR + 46, CELL / 2)]
    for side, marks in enumerate(tile["marks"]):
        names = [name for name, bit in MARK_BITS.items() if marks & bit]
        if names:
            text_centered(draw, mark_points[side], " ".join(names), small, (255, 230, 0))

    # Fields on half-edges, and the inner one.
    for e, field in enumerate(tile["fields"]):
        if field < 0:
            continue
        x, y = half_edge_point(e, BAR + 24)
        colour = FIELD_COLOURS[field % len(FIELD_COLOURS)]
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=colour, outline="white", width=2)
        text_centered(draw, (x, y), str(field), small, "white")
    if tile["inner_field"] >= 0:
        x = y = CELL / 2
        colour = FIELD_COLOURS[tile["inner_field"] % len(FIELD_COLOURS)]
        draw.ellipse((x - 18, y - 18, x + 18, y + 18), fill=colour, outline="white", width=3)
        text_centered(draw, (x, y), str(tile["inner_field"]), big, "white")

    # Caption.
    marks = "".join(f"  {name}" for name, bit in TILE_MARK_BITS.items() if tile["tile_marks"] & bit)
    title = f"type {tile['type']}  x{tile['count']}{marks}"
    draw.text((4, CELL + 4), title, font=big, fill="black")
    borders = []
    for field in range(tile["field_count"]):
        sides = "".join(SIDE_NAMES[s] for s in range(4) if tile["field_city_sides"][field] & (1 << s))
        borders.append(f"f{field}:{sides or '-'}")
    draw.text((4, CELL + 30), "  ".join(borders) if borders else "no fields", font=small, fill=(60, 60, 60))
    if tile["errors"]:
        draw.text((4, CELL + 50), f"{tile['errors']} errors, see dump_tiles", font=small, fill=(220, 0, 0))
    return canvas


def draw_sheet(tiles, tiles_dir, fonts, title):
    rows = (len(tiles) + COLUMNS - 1) // COLUMNS
    header = 40
    sheet = Image.new("RGB", (COLUMNS * (CELL + GAP) + GAP, header + rows * (CELL + CAPTION + GAP) + GAP),
                      (235, 235, 235))
    ImageDraw.Draw(sheet).text((GAP, 8), title, font=fonts[0], fill="black")
    for i, tile in enumerate(tiles):
        row, column = divmod(i, COLUMNS)
        sheet.paste(draw_tile(tile, tiles_dir, fonts),
                    (GAP + column * (CELL + GAP), header + GAP + row * (CELL + CAPTION + GAP)))
    return sheet


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("table", nargs="?", default=os.path.join(here, "results", "tile_table.json"))
    parser.add_argument("--tiles", default=os.path.join(here, "..", "tiles"))
    parser.add_argument("--out", default=os.path.join(here, "results"))
    args = parser.parse_args()

    with open(args.table, encoding="utf-8") as f:
        data = json.load(f)
    table = data["tile_types"]
    MARK_BITS.update(data["mark_bits"])
    TILE_MARK_BITS.update(data["tile_mark_bits"])
    os.makedirs(args.out, exist_ok=True)
    fonts = (load_font(20), load_font(15))

    expansions = []
    for tile in table:
        if tile["expansion"] not in expansions:
            expansions.append(tile["expansion"])
    for expansion in expansions:
        tiles = [t for t in table if t["expansion"] == expansion]
        count = sum(t["count"] for t in tiles)
        title = (f"{expansion}: {len(tiles)} types, {count} tiles   "
                 "bars: G grass, C city, R road, W river + link   dots: field per half-edge   "
                 "yellow: marks   fN: city sides field N borders")
        out = os.path.join(args.out, f"tile_sheet_{expansion}.png")
        draw_sheet(tiles, args.tiles, fonts, title).save(out)
        print(f"wrote {out} ({len(tiles)} types)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
