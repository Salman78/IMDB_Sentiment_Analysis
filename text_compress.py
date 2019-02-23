import csv
import os

def read_files(path):
    data = []
    for file_name in os.listdir(path):
        with open(os.path.join(path,file_name),'rb') as f:
            review = f.read().decode('utf-8').replace('\n','').strip().lower()
            data.append([review,int(file_name.split(".")[0])])
    return data



neg_path = "C:\\Users\\MSI\\Documents\\School things\\Comp 551\\Proj2_materials\\train\\neg"
training_text_neg = []
training_text_pos = []
test_text = []

# Reads each file in neg folder and appends to a list
training_text_neg = read_files(neg_path)
# Places all text in a new file
print("neg done")
with open('neg_text.txt', 'w',encoding='utf-8') as f:
    for item in training_text_neg:
        f.write("%s\n" % item[0])


# Reads each file in pos folder and appends to a list
pos_path = "C:\\Users\\MSI\\Documents\\School things\\Comp 551\\Proj2_materials\\train\\pos"
training_text_pos = read_files(pos_path)
# Places all text in a new file 
with open('pos_text.txt', 'w',encoding='utf-8') as f:
    for item in training_text_pos:
        f.write("%s\n" % item[0])


print("pos done")
# Reads each file in test folder and appends to a list
test_path = "C:\\Users\\MSI\\Documents\\School things\\Comp 551\\Proj2_materials\\test"
test_text = read_files(test_path)
# Places all text in a new file 
with open('test_text.txt', 'w',encoding='utf-8') as f:
    for item in test_text:
        f.write("%s\n" % item[0])
with open('test_index.txt', 'w',encoding='utf-8') as f:
    for item in test_text:
        f.write("%s\n" % item[1])
print("test done")



