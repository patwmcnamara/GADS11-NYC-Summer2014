'''
We're going to do PCA the hard way, then the easy way
'''

### GENERATING 3-DIMENSIONAL SAMPLE DATA ###

import numpy as np
numpy.random.seed(1) #for consistency

mu_vec1 = numpy.array([10,0,0])
cov_mat1 = numpy.identity(3)
class1_sample = numpy.random.multivariate_normal(mu_vec1, cov_mat1, 20).T

mu_vec2 = numpy.array([1,1,1])
cov_mat2 = numpy.identity(3)
class2_sample = numpy.random.multivariate_normal(mu_vec2, cov_mat2, 20).T

# Plotting our data in 3-dimensions! #
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(class1_sample[0,:], class1_sample[1,:],\
    class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(class2_sample[0,:], class2_sample[1,:],\
    class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')

plt.title('Samples for classes 1 & 2')
ax.legend(loc='upper right')
plt.show()


### COMPUTING THE SCATTER MATRIX ###

# First we need the mean vector #
mean_vector = numpy.mean(all_samples, axis=1)
mean_vector

scatter_matrix = numpy.zeros((3,3))

for i in range(all_samples.shape[1]):
    scatter_matrix += (all_samples[:,i].reshape(3,1) - mean_vector).dot((all_samples[:,i].reshape(3,1) - mean_vector).T)

scatter_matrix

### COMPUTING THE COVARIANCE MATRIX ###

# Removing class labels, remember this is UNsupervised learning #

all_samples = numpy.concatenate((class1_sample, class2_sample), axis=1)
assert all_samples.shape == (3,40), "The matrix has not the dimensions 3x40"

cov_mat = numpy.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])

cov_mat

### COMPUTING EIGENVECTORS AND CORRESPONDING EIGENVALUES ###

# For the scatter matrix
eig_val_sc, eig_vec_sc = numpy.linalg.eig(scatter_matrix)

# For the covariance matrix
eig_val_cov, eig_vec_cov = numpy.linalg.eig(cov_mat)

'''
To show the eigenvectors are identical whether we derived them from the scatter or the 
covariance matrix, let's put an assert statement into the code. Also, we will see the 
eigenvalues were scaled by the factor 39 when we derived it from the scatter matrix.
'''
for i in range(len(eig_val_sc)):
    eigvec_sc = eig_vec_sc[:,i].reshape(1,3).T
    eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T
    assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'
    print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
    print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
    print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
    print('Scaling factor: ', eig_val_sc[i]/eig_val_cov[i])
    print(40 * '-')

# Visualizing the eigenvectors #
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(all_samples[0,:], all_samples[1,:], all_samples[2,:], 'o', markersize=8, color='green', alpha=0.2)
ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
for v in eig_vec_sc.T:
    a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)

ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')
plt.title('Eigenvectors')
plt.show()

### CHOOSING OUR EIGENVECTORS ###

# Make a list of (eigenvalue, eigenvector) tuples and sort them#
eig_pairs = [(numpy.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
eig_pairs.sort(reverse=True)
eig_pairs


### TRANSFORMING THE SAMPLES ONTO THE NEW SUBSPACE ###
transformed = matrix_w.T.dot(all_samples)
transformed

plt.plot(transformed[0,0:20], transformed[1,0:20],'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transformed[0,20:40], transformed[1,20:40],'^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')
plt.show()


'''
NOW THE EASY WAY, WITH SCI-KIT LEARN
'''
from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=2)
sklearn_transf = sklearn_pca.fit_transform(all_samples.T)

plt.plot(sklearn_transf[0:20,0],sklearn_transf[0:20,1],'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(sklearn_transf[20:40,0], sklearn_transf[20:40,1],'^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples with class labels from sklearnPCA')
plt.show()

# That looks like the opposite of our original graph, let's fix that #
sklearn_transf = sklearn_transf * (-1)
plt.plot(sklearn_transf[0:20,0],sklearn_transf[0:20,1],'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(sklearn_transf[20:40,0], sklearn_transf[20:40,1],'^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples via sklearn.decomposition.PCA')
plt.show()
