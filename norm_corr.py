import matplotlib.pyplot as plt
from scipy.signal import max_len_seq
import random
import hashlib
import numpy as np


# generate pn sequence
def generate_max_len_seq(nbits, state):
    # length of PN = 4096
    pn_row = max_len_seq(nbits=nbits, state=state, taps=[7,6,1], length=4096)[0]
    pn_row.tolist()
    return pn_row


# calculate normalized_correlation
def normalized_correlation(a, b):
    assert (a.shape == b.shape)

    numerator = np.dot(a, np.transpose(b))
    denominator = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))

    if denominator == 0:
        return 0
    else:
        norm_corr = numerator / denominator

    return norm_corr


# generate verification code, length is 10
def generate_verification_code(len):
    code_list = []
    for i in range(10):
        code_list.append(str(i))

    myslice = random.sample(code_list, len)
    verification_code = ''.join(myslice)
    return verification_code


# convert hash code to 512 bits binary
def hash_convert_binary(str):
    hash_code = hashlib.sha512(str.encode())
    he_hash_code = hash_code.hexdigest()

    scale = 16
    num_of_bits = 512
    code = bin(int(he_hash_code, scale))[2:].zfill(num_of_bits)

    # convert code to numpy type
    code = np.array(list(code), dtype=int)
    return code


# calculate mean
def calculate_mean(list):
    mean = np.mean(list)
    return mean


# calculate variance
def calculate_variance(list):
    variance = np.std(list)
    return variance


# step one
# generate 2 matrix, A and B
A = 1 - 2 * np.random.randint(2, size=(50, 100))
B = 1 - 2 * np.random.randint(2, size=(50, 100))

# generate matrix C
C = A * B

# for each row in matrix C, randomly generate PN sequence
pn = []
for row in C:
    pn.append(generate_max_len_seq(nbits=100, state=row))

# calculate normalized correlation
norm_corr = []
col = 1
matrix_length = len(C)
for row_index in range(matrix_length - 1):
    for col in range(1, matrix_length):
        if col > row_index:
            norm_corr.append(normalized_correlation(C[row_index], C[col]))

print("step one: ")
print("mean: ", calculate_mean(norm_corr))
print("variance: ", calculate_variance(norm_corr))
print()

# plot a image
plt.figure(0)
plt.ylim((-5, 5))
plt.scatter(np.arange(len(norm_corr)), np.array(norm_corr))
plt.show()

## step two
veri_code = []
binary_list = []
pn_images = []
num_image = 100 # number of image

# generate verification code
for i in range(num_image):
    veri_code.append(generate_verification_code(9))

# convert hash code to binary code
for i in range(len(veri_code)):
    binary_list.append(hash_convert_binary(veri_code[i]))

for i in range(len(binary_list)):
    for j in range(len(binary_list[0])):
        if binary_list[i][j] == 0:
            binary_list[i][j] = -1

# generate PN sequence
for item in binary_list:
    pn_images.append(generate_max_len_seq(nbits=512, state=item))

# calculate normalized correlation
norm_corr_images = []
col = 1
pn_length = len(pn_images)

for row_index in range(pn_length - 1):
    for col in range(1, pn_length):
        if col > row_index:
            norm_corr_images.append(normalized_correlation(pn_images[row_index], pn_images[col]))

print("step two: ")
print("mean: ", calculate_mean(norm_corr_images))
print("variance: ", calculate_variance(norm_corr_images))

# plot a image
plt.figure(1)
plt.ylim((-5, 5))
plt.scatter(np.arange(len(norm_corr_images)), np.array(norm_corr_images))
plt.show()
