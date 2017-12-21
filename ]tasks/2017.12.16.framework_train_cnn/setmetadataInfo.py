import uab_collectionFunctions

blCol = uab_collectionFunctions.uabCollection('inria')
blCol.getMetaDataInfo([0, 1, 2])
print(blCol.getChannelMeans([0, 1, 2]))
print(type(blCol.getChannelMeans([0, 1, 2])))
