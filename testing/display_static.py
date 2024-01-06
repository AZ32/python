import board
from adafruit_st7789 import St7789
import adafruit_display_text.label
from adafruit_bitmap_font import bitmap_font
from displayio import Bitmap, ColorConverter, Palette, Group, TileGrid
import displayio
displayio.release_displays()
spi=board.SPI()
tft_cs = board.D5
tft_dc = board.D6
tft_rst = board.D9
display_bus = displayio.FourWire(spi, command=tft_dc, chip_select=tft_cs, reset=tft_rst)
display = ST7789(display_bus, width=320, height=172, colstart=34, rotation=270)

palette = Palette(3)
palette[0] = 0x000000 # Black
palette[1] = 0xFFFFFF # White
palette[2] = 0xFF0000 # Red
bitmap=Bitmap(320,172,2)
for x in range(320):
    for y in range(172):
        bitmap[x, y] = 1 if 50 <= x <= 270 and 50 <= y <= 122 else 0

# TODO: get BDF font to include here

font = bitmap_font.load_font("Minecraft-12.bdf")


text_area = adafruit_display_text.label.Label(font, text="Hello!", color=0xFFFFFF)
text_area.x = 0
text_area.y = 0

group = Group()

group.append(TileGrid(bitmap, pixel_shader=palette))
group.append(text_area)

disp.show(group)
