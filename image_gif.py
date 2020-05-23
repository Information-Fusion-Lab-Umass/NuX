import imageio
import os

experiment_folder = 'Experiments/CelebA128'

# Take an example folder that has the filenames we want
example_folder = 'Experiments/CelebA128/26000/plots'
file_names = list(os.walk(example_folder))[0][2]

# We will store the set of images together
image_dict = dict([(name, []) for name in file_names])

# Go through the images and put them together
for root, dirs, _ in os.walk(experiment_folder):
    for d in dirs:
        if(d != 'plots'):
            continue
        for f in file_names:
            image_path = os.path.join(root, d, f)
            image_dict[f].append(image_path)

# Sort the images
def sort_key(path):
    directory, _ = os.path.split(path)
    return int(directory.split('/')[-2])

for key, val in list(image_dict.items()):
    image_dict[key] = sorted(val, key=sort_key)

# Create each gif
# for key, filenames in image_dict.items():
#     gif_path = './test.gif'
#     images = []
#     for filename in filenames:
#         images.append(imageio.imread(filename))
#     imageio.mimsave(gif_path, images)
#     break

for key, filenames in image_dict.items():
    gif_path = '%s.gif'%(key.strip('.png'))
    with imageio.get_writer(gif_path, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
