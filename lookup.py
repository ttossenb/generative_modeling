import numpy as np
import vis

# lables in latents_values idx: ('color':0, 'shape':1, 'scale':2, 'orientation':3, 'posX':4, 'posY':5)
# lables in latents_values qnty: ('color':1, 'shape':3, 'scale':6, 'orientation':40, 'posX':32, 'posY':32)
# color is not a parameter, because it's always '1'
def get_image(shape, scale, orientation, posX, posY):
    data = np.load("datasets/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    imgs = data['imgs']

    posY_const = 1
    posX_const = 32 * posY_const
    orientation_const = 32 * posX_const
    scale_const = 40 * orientation_const
    shape_const = 6 * scale_const

    index = shape * shape_const + scale * scale_const + orientation * orientation_const + posX * posX_const + posY * posY_const
    return np.array([ imgs[index] ])

#image = get_image(2, 5, 39, 31, 31)
#vis.plotImages(np.expand_dims(image, axis=3), len(image), 1, 'pictures/lookup/test')
