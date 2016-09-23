from PIL import Image
import sys

filename = sys.argv[1]
im = Image.open(filename)
newIm = im.rotate(180)
newIm.save("ans2.png")
