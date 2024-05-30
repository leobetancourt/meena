from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


parser = ArgumentParser(description="Create a movie from a .npy file.")

parser.add_argument('-f', '--file', type=str, required=True,
                    help='The path to the .npy file (required)')

parser.add_argument('-o', '--output', type=str, required=True,
                    choices=['density', 'vx', 'vy', 'pressure'],
                    help='The variable to plot: density, vx, vy, or pressure (required)')

args = parser.parse_args()

# validate the file argument
if not args.file.endswith('.npy'):
    parser.error("The file name must end with '.npy'")

file_path = args.file
var = args.output

hist = np.load(file_path)
print(hist)
# fig = plt.figure()
# # output video writer
# clear_frames = True
# fps = 24
# FFMpegWriter = animation.writers['ffmpeg']

# metadata = dict(title=file, comment='')
# writer = FFMpegWriter(fps=fps, metadata=metadata)
# PATH = f"./videos/{filename}"
# if not os.path.exists(PATH):
#     os.makedirs(PATH)
# cm = writer.saving(fig, f"{PATH}/{plot}.mp4", 100)

# for frame in hist:

#     if clear_frames:
#         fig.clear()
#     self.plot(plot)
#     writer.grab_frame()
