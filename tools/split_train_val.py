import os
import shutil
import random
 
random.seed(0) # make sure that can redo the results
 
def mk_dir(file_path):
    if os.path.exists(file_path):
        return
        # shutil.rmtree(file_path)
    os.makedirs(file_path)
 
def split_data(file_path,new_file_path,train_rate,val_rate,test_rate):
    subfile = list()
    if test_rate != 0.0:
        subfile = ['train', 'val', 'test']
    else:
        subfile = ['train', 'val']
    # for name in subfile:
    #     mk_dir(new_file_path + '/' + name) 
    images_list = []
    for image in os.listdir(file_path+"/images_ours/"):
        images_list.append(image)
    total = len(images_list)
    print(total)
    random.shuffle(images_list)
    train_images = images_list[0:5000]
    val_images = images_list[5000:int((train_rate+val_rate)*total)]
    test_images = images_list[int((train_rate+val_rate)*total):]
    
    for image in train_images:
        old_path = file_path+'/images_ours/'+image
        new_path = new_file_path+'/'+'train'+'/images/'+image
        mk_dir(new_file_path+'/'+'train'+'/images/')
        shutil.copy(old_path,new_path)
        mask_old_path = file_path+'/masks_ours/'+image.split('.')[0]+'.png'
        mask_new_path = new_file_path+'/'+'train'+'/annotations/'+image
        mk_dir(new_file_path+'/'+'train'+'/annotations/')
        shutil.copy(mask_old_path,mask_new_path)

 
    for image in val_images:
        old_path = file_path+'/images_ours/'+image
        new_path = new_file_path+'/'+'test'+'/images/'+image
        mk_dir(new_file_path+'/'+'test'+'/images/')
        shutil.copy(old_path,new_path)
        mask_old_path = file_path+'/masks_ours/'+image.split('.')[0]+'.png'
        mask_new_path = new_file_path+'/'+'test'+'/annotations/'+image
        mk_dir(new_file_path+'/'+'test'+'/annotations/')
        shutil.copy(mask_old_path,mask_new_path)
 
    for image in test_images:
        old_path = file_path+'/'+image
        new_path = new_file_path+'/'+'test'+'/'+image
        shutil.copy(old_path,new_path)
 
if __name__ == '__main__':
    file_path = f"/root/paddlejob/workspace/env_run/output/haojing08/datasets_dental/2D-X-Ray/final_labelled"
    new_file_path = f"/root/paddlejob/workspace/env_run/output/haojing08/datasets_dental/2D-X-Ray/final_labelled/tooth-2d-x-ray-6k"
    split_data(file_path,new_file_path,train_rate=0.82,val_rate=0.18,test_rate=0.0)
