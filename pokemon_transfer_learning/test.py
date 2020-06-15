import os,glob

# images = []
# images += glob.glob(os.path.join('data/pokeman/pikachu', '*.png'))
# images += glob.glob(os.path.join('data/pokeman/pikachu', '*.jpg'))
# images += glob.glob(os.path.join('data/pokeman/pikachu', '*.jpeg'))
#
# print(images)

effective_attachments = ['png','jpg','jpeg']
images = []
attachments = []
for name in os.listdir('data/pokeman/pikachu'):
    attachment = name.rsplit('.',-1)[-1]
    attachments.append(attachment)

    if name.rsplit('.',-1)[-1] in effective_attachments:
        images.append(os.path.join('data/pokeman/pikachu',name))

# print(set(attachments))
print(images)
with open(os.path.join('data','data.csv'),'w',newline='') as f:
    for img in images:
        lab_name = img.split(os.sep)[2]
        # print("%s,%s/n"%(img,lab_name))
        f.write("%s,%s\n"%(img,lab_name))