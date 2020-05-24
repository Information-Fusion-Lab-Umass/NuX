import shutil

rm1 = 'CIFAR256'
rm2 = 'CIFAR512'
rm3 = 'CelebA128'
rm4 = 'CelebA256'
rm5 = 'CelebA512'


for i in range(500,100000, 1000):
    try:
        shutil.rmtree('Experiments/' + rm1 + '/' + str(i))
    except OSError as e:
        temp = 0

    try:
        shutil.rmtree('Experiments/' + rm2 + '/' + str(i))
    except OSError as e:
        temp = 0

    try:
        shutil.rmtree('Experiments/' + rm3 + '/' + str(i))
    except OSError as e:
        temp = 0

    try:
        shutil.rmtree('Experiments/' + rm4 + '/' + str(i))
    except OSError as e:
        temp = 0

    try:
        shutil.rmtree('Experiments/' + rm5 + '/' + str(i))
    except OSError as e:
        temp = 0
