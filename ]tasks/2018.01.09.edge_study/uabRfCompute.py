# Compute input size that leads to a 1x1 output size, among other things
# [filter size, stride, padding]

convnet =[[3, 1, 0], [3, 1, 0], [2, 2, 0],
          [3, 1, 0], [3, 1, 0], [2, 2, 0],
          [3, 1, 0], [3, 1, 0], [2, 2, 0],
          [3, 1, 0], [3, 1, 0], [2, 2, 0],
          [3, 1, 0], [3, 1, 0]]
layer_name = ['conv1_1', 'conv1_2', 'pool1',
              'conv2_1', 'conv2_2', 'pool2',
              'conv3_1', 'conv3_2', 'pool3',
              'conv4_1', 'conv4_2', 'pool4',
              'conv5_1', 'conv5_2']
imsize = 572

def outFromIn(isz, layernum = 9, net = convnet):
    if layernum>len(net): layernum=len(net)

    totstride = 1
    insize = isz
    #for layerparams in net:
    outsize = 0
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2*pad) / stride + 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride

def inFromOut( layernum = 9, net = convnet):
    if layernum>len(net): layernum=len(net)
    outsize = 1
    #for layerparams in net:
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        outsize = ((outsize -1)* stride) + fsize
    RFsize = outsize
    return RFsize

if __name__ == '__main__':

    print('layer output sizes given image = %dx%d' % (imsize, imsize))
    for i in range(len(convnet)):
        p = outFromIn(imsize,i+1)
        rf = inFromOut(i+1)
        print("Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (layer_name[i], p[0], p[1], rf))
