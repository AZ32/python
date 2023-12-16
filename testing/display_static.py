import adafruit_display_text.label
from adafruit_bitmap_font import bitmap_font
from displayio import Bitmap, ColorConverter, Palette

palette = Palette(3)
palette[0] = 0x000000 # Black
palette[1] = 0xFFFFFF # White
palette[2] = 0xFF0000 # Red

for x in range(320):
    for y in range(172):
        bitmap[x, y] = 1 if 50 <= x <= 270 and 50 <= y <= 122 else 0

# TODO: get BDF font to include here
font_file = "fonts/LeagueSpartan-Bold-12.bdf"
font = bitmap_font.load_font(font_file)

text_area = adafruit_display_text.label.Label(font, text="Hello!", color=0xFFFFFF)
text_area.x = 0
text_area.y = 0

group = displayio.Group()

group.append(displayio.TileGrid(bitmap, pixel_shader=palette))
group.append(text_area)

disp.show(group)