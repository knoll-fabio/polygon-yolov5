from utils.autoanchor import polygon_kmean_anchors

nl = 3 # number of anchor layers
na = 3 # number of anchors
img_size = 640 # image size for training and testing

datacfg = "data/polygon_signs.yaml"
modelcfg = "conf/polygon_yolov5s_signs.yaml"

# calculate anchors based on train set
anchors = polygon_kmean_anchors(datacfg, n=nl*na, gen=3000, img_size=img_size)
anchors = anchors.reshape(nl, na*2).astype(int)
print(anchors)

# read model config template
with open(modelcfg, "r") as file:
    filecontent = file.read()
    print(filecontent)

# replace placeholders with anchors
for i,l in enumerate(anchors):
    filecontent = filecontent.replace("$anchors{}".format(i+1), str(l.tolist()))

# write model config to file
with open(modelcfg, "w") as file:
    file.write(filecontent) 