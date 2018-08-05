import matplotlib.pyplot as plt
from scipy.signal import max_len_seq
import random
import hashlib
import numpy as np


def genrate_max_len_seq(nbits, state):
    pn_row = max_len_seq(nbits=nbits, state=state, taps=[7,6,1], length=4096)[0]
    pn_row.tolist()
    return pn_row


def normalized_correlation(a, b):
    assert (a.shape == b.shape)

    numerator = np.dot(a, np.transpose(b))
    denominator = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))

    if denominator == 0:
        return 0
    else:
        norm_corr = numerator / denominator

    return norm_corr


def generate_verification_code(len):
    code_list = []
    for i in range(10): # 0-9数字
        code_list.append(str(i))

    myslice = random.sample(code_list, len)
    verification_code = ''.join(myslice)
    return verification_code


def hash_convert_binary(str):
    hash_code = hashlib.sha512(str.encode())
    he_hash_code = hash_code.hexdigest()

    scale = 16

    num_of_bits = 512

    code = bin(int(he_hash_code, scale))[2:].zfill(num_of_bits)

    code = np.array(list(code), dtype=int)
    return code


# step one
A = 1 - 2 * np.random.randint(2, size=(50, 100))
B = 1 - 2 * np.random.randint(2, size=(50, 100))

C = np.dot(A, np.transpose(B))

pn = []
for row in C:
    pn.append(genrate_max_len_seq(nbits=50, state=row))

A_trans = np.transpose(A)
B_trans = np.transpose(B)

norm_corr = []
col = 1
matrix_length = len(A_trans)
for row_index in range(matrix_length - 1):
    for col in range(matrix_length):
        if col > row_index:
            norm_corr.append(normalized_correlation(A_trans[row_index], A_trans[col]))

plt.ylim((-5, 5))
plt.scatter(np.arange(len(norm_corr)), np.array(norm_corr))
plt.show()

## step two
veri_code = []
binary_list = []
pn_images = []
num_image = 100
for i in range(num_image):
    veri_code.append(generate_verification_code(9))

for i in range(len(veri_code)):
    binary_list.append(hash_convert_binary(veri_code[i]))

for item in binary_list:
    pn_images.append(genrate_max_len_seq(nbits=512, state=item))

pn_images_trans = np.transpose(pn_images)

norm_corr_images = []
col = 1
pn_length = len(pn_images)

for row_index in range(pn_length - 1):
    for col in range(pn_length):
        if col > row_index:
            norm_corr_images.append(normalized_correlation(pn_images_trans[row_index], pn_images_trans[col]))

print(norm_corr_images)
plt.ylim((-5, 5))
plt.scatter(np.arange(len(norm_corr_images)), np.array(norm_corr_images))
plt.show()