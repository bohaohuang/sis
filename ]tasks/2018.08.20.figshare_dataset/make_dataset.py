import uab_collectionFunctions


blCol = uab_collectionFunctions.uabCollection('dc')
blCol.readMetadata()
img_mean = blCol.getChannelMeans([0, 1, 2])
