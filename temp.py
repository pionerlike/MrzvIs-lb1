import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from IPython.display import clear_output

def to_blocks(array, color_number):
    blocks = []
    for i in range(height // block_height):
        for j in range(width // block_width):
            block = []
            for y in range(block_height):
                for x in range(block_width):
                    for color in range(color_number):
                        block.append(array[i * block_height + y, j * block_width + x, color])
            blocks.append(block)
    return np.array(blocks)

def to_array(blocks, color_number):
    array = []
    blocks_in_line = width // block_width
    for i in range(height // block_height):
        for y in range(block_height):
            line = []
            for j in range(blocks_in_line):
                for x in range(block_width):
                    pixel = []
                    for color in range(color_number):
                        pixel.append(blocks[i * blocks_in_line + j, (y * block_width * color_number) + (x * color_number) + color])
                    line.append(pixel)
            array.append(line)
    return np.array(array)

def show(array):
    array = 1 * (array + 1) / 2
    plt.axis('off')
    plt.imshow(array)
    plt.show()


image = imread("data/panda.png")

image = (2.0 * image / 1.0) - 1.0
array = np.array(image)
show(array)

# For example 3 in RGB, 4 in RGBA. More colors - works slower, but results are better
color_number = 4
# Max epoch number
max_number = 300

height = np.size(array, 0) # h
width = np.size(array, 1) # w

block_height = 4 # n
block_width = 4# m

number_of_blocks = int((height * width) / (block_height * block_width))
#4*4*4
input_layer_size = block_height * block_height * color_number # S * n * m
hidden_layer_size = 16 # P - number of neurons

blocks = to_blocks(array, color_number).reshape(number_of_blocks, 1, input_layer_size)


w1 = np.random.rand(input_layer_size, hidden_layer_size) * 2 - 1
z = np.copy(w1)
w2 = z.transpose()

error_max = 250.0
error_current = error_max + 1
alpha = 0.0007
alpha_trans = 0.0007
epoch = 0

while (error_current > error_max and epoch < max_number):
    error_current = 0
    epoch += 1
    
    for i in blocks:
        y = np.matmul(i, w1)#свернули в 16
        x1 = np.matmul(y, w2)#развернули в 64
        dx = x1 - i #reuslt-default
        alpha = 1 / (np.matmul(i, i.transpose()).take(0,0)*10)
        alpha_trans = 1 / (np.matmul(y, y.transpose()).take(0,0)*10)
        w1 -= alpha * np.matmul(np.matmul(i.transpose(), dx), w2.transpose())
        w2 -= alpha_trans * np.matmul(y.transpose(), dx)
    for i in blocks:
        Z = ((input_layer_size * number_of_blocks)  
            / ((input_layer_size + number_of_blocks) * hidden_layer_size + 2))
        dx = (np.matmul((np.matmul(i, w1)), w2)) - i
        error = (dx * dx).sum()
        error_current += error
    clear_output(wait=True)
    print('Epoch:', epoch)
    print('Error:', error_current)

print('Z:', Z)
    
result = []
for block in blocks:
    result.append(block.dot(w1).dot(w2))
result = np.array(result)

show(array)
show(to_array(result.reshape(number_of_blocks, input_layer_size), color_number))
