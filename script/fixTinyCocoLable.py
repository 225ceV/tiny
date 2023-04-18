import os
cls = ['9', '14', '32', '33', '67']
new_cls = ['0', '1', '2', '3', '4']
dic = dict(zip(cls, new_cls))
path = "../datasets/tiny-coco-test/labels"
filelist = os.listdir(os.path.join(path, "train"))
out_path = os.path.join(path, "train_new")
os.mkdir(out_path)
for file in filelist:
    result = list()
    with open(os.path.join(path, "train", file), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            obj = line.split(' ')
            if obj[0] in dic:
                obj[0] = dic[obj[0]]
                new_line = ' '.join(obj)+'\n'
                result.append(new_line)
    new_file = os.path.join(out_path, file)
    with open(new_file, 'w') as f:
        f.writelines(result)
