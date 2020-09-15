import numpy as np
import matplotlib.pyplot as plt
from skimage import io


# Minkowski distance
def dist_p(vec1, vec2, p):  
    L = len(vec1)
    s1 = 0
    for l in range(L):
        diff = np.abs(vec2[l] - vec1[l])
        s1 = s1 + diff ** p
    distance = s1 ** (1 / p)
    return distance


def init_mean(K, image, label_cluster):
    mean_list = []  ## List containing mean values of the clusters
    pixel_ls = [[] for k in range(K)]  ## Create list of empty lists to store pixels belonging to a certain cluster

    for i in range(label_cluster.shape[0]):
        for j in range(label_cluster.shape[1]):
            for k in range(K):
                if label_cluster[i, j] == k:  ## if the label of the pixel at location [i,j] is 'k'
                    pixel_ls[k].append(np.ravel(image[i, j, :]))  ## Fill the kth empty list with this pixel value

    for k in range(K):
        pixel_mat = np.asmatrix(pixel_ls[k])
        mean_k = np.mean(pixel_mat, axis=0)
        mean_list.append(np.ravel(mean_k))
    return mean_list


def label_update(prev_mean, image, label_cluster, p):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            dist_ls = []
            for k in range(len(prev_mean)):
                dist = dist_p(image[i, j, :], prev_mean[k],
                              p)  ## Calculate the distance of the pixel at [i,j] with the kth mean
                dist_ls.append(dist)  ## Put the distance values in a list
            dist_arr = np.array(dist_ls)  ## Convert it to a NumPy array
            new_label = np.argmin(
                dist_arr)  ##The new_label of the point is the one which is closest to the pixel at [i,j]
            label_cluster[i, j] = new_label  ## Set the new label
    return label_cluster


def mean_from_label(K, prev_mean, image, label_cluster):  ##label array is the current label array
    pixel_ls = [[] for k in range(K)]  ## Create list of empty lists to store pixels belonging to a certain cluster

    for i in range(label_cluster.shape[0]):
        for j in range(label_cluster.shape[1]):
            for k in range(K):
                if label_cluster[i, j] == k:  ## if the label of the pixel at location [i,j] is 'k'
                    pixel_ls[k].append(np.ravel(image[i, j, :]))  ## Fill the kth empty list with this pixel value

    for k in range(K):
        if len(pixel_ls[k]) != 0:  ## Only update the means of those clusters which has received at least one new point, else retain the old mean value
            pixel_mat = np.asmatrix(pixel_ls[k])
            mean_k = np.mean(pixel_mat, axis=0)
            prev_mean[k] = np.ravel(mean_k)  ##np.ravel to flatten the vector
    new_mean = prev_mean
    return new_mean


def KMeans(image, label_cluster, K, p, maxIter):  ##initial mean
    mean_old = init_mean(K, image, label_cluster)
    for t in range(maxIter):
        new_label_cluster = label_update(mean_old, image, label_cluster, p)  ##new mean
        mean_new = mean_from_label(K, mean_old, image, new_label_cluster)
        label_cluster = new_label_cluster  ## Update the label array
        mean_old = mean_new  ## Update the mean values
    return mean_new, label_cluster


def CartoonNizer_ID(images):
    # input image
    image_input = io.imread(images)
    # create and empty array
    label_cluster = np.zeros((image_input.shape[0], image_input.shape[1]))

    # define the K
    K = 8
    # make a random choice
    for i in range(label_cluster.shape[0]):
        for j in range(label_cluster.shape[1]):
            label_cluster[i, j] = np.random.choice(K)

    # apply the K-Means Cluster
    mean_new,label_arr = KMeans(image_input,label_cluster,K,1,3)

    # put the result into the empty array
    seg_image = np.zeros((image_input.shape[0], image_input.shape[1], image_input.shape[2]))
    for i in range(seg_image.shape[0]):
        for j in range(seg_image.shape[1]):
            k = label_cluster[i, j]
            seg_image[i, j, :] = mean_new[int(k)]
    # save the image and give the result
    seg_image = seg_image.astype(np.uint8)
    io.imsave("Result_task1.png", seg_image)
    plt.imshow(seg_image)
    plt.show()


CartoonNizer_ID("image-asset.jpg")