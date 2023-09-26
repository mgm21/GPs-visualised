import glob
from PIL import Image

# l1 = [f"00{i}" for i in range(1, 10)]
# l2 = [f"01{i}" for i in range(0, 3)]
#
# frames = [Image.open(f"images/counter_{i}.png") for i in l1]
# frames += [Image.open(f"images/counter_{i}.png") for i in l2]
# frame_one = frames[0]
# frame_one.save("gifs/xy_angled_gp.gif", format="GIF", append_images=frames,
#                save_all=True, duration=650, loop=0)

l1 = [f"00{i}" for i in range(1, 10)]
l2 = [f"01{i}" for i in range(0, 3)]

frames = [Image.open(f"images/counter_{i}.png") for i in l1]
frames += [Image.open(f"images/counter_{i}.png") for i in l2]
frame_one = frames[0]
frame_one.save("gifs/xy_angled_gp.gif", format="GIF", append_images=frames,
               save_all=True, duration=650, loop=0)