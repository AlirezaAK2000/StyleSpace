import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" #(or "1" or "2")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import os
import pickle
import numpy as np
from dnnlib import tflib  
import tensorflow as tf 

import argparse
from PIL import Image

def LoadModel(dataset_name):
    # Initialize TensorFlow.
    tflib.init_tf()
    model_path='./model/'
    model_name=dataset_name+'.pkl'
    
    tmp=os.path.join(model_path,model_name)
    with open(tmp, 'rb') as f:
        _, _, Gs = pickle.load(f)
    return Gs

def lerp(a,b,t):
     return a + (b - a) * t

#stylegan-ada
def SelectName(layer_name,suffix):
    if suffix==None:
        tmp1='add:0' in layer_name 
        tmp2='shape=(?,' in layer_name
        tmp4='G_synthesis_1' in layer_name
        tmp= tmp1 and tmp2 and tmp4  
    else:
        tmp1=('/Conv0_up'+suffix) in layer_name 
        tmp2=('/Conv1'+suffix) in layer_name 
        tmp3=('4x4/Conv'+suffix) in layer_name 
        tmp4='G_synthesis_1' in layer_name
        tmp5=('/ToRGB'+suffix) in layer_name
        tmp= (tmp1 or tmp2 or tmp3 or tmp5) and tmp4 
    return tmp


def GetSNames(suffix):
    #get style tensor name 
    with tf.Session() as sess:
        op = sess.graph.get_operations()
    layers=[m.values() for m in op]
    
    
    select_layers=[]
    for layer in layers:
        layer_name=str(layer)
        if SelectName(layer_name,suffix):
            select_layers.append(layer[0])
    return select_layers

def SelectName2(layer_name):
    tmp1='mod_bias' in layer_name 
    tmp2='mod_weight' in layer_name
    tmp3='ToRGB' in layer_name 
    
    tmp= (tmp1 or tmp2) and (not tmp3) 
    return tmp

def GetKName(Gs):
    
    layers=[var for name, var in Gs.components.synthesis.vars.items()]
    
    select_layers=[]
    for layer in layers:
        layer_name=str(layer)
        if SelectName2(layer_name):
            select_layers.append(layer)
    return select_layers

def GetCode(Gs,random_state,num_img,num_once,truncation=True):
    rnd = np.random.RandomState(random_state)  #5
    
    if truncation:
        truncation_psi=0.7
        truncation_cutoff=8
    
    dlatent_avg=Gs.get_var('dlatent_avg')
    
    dlatents=np.zeros((num_img,512),dtype='float32')
    for i in range(int(num_img/num_once)):
        src_latents =  rnd.randn(num_once, Gs.input_shape[1])
        src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
        
        # Apply truncation trick.
        if truncation:
            layer_idx = np.arange(src_dlatents.shape[1])[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_idx.shape, dtype=np.float32)
            coefs = np.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
            src_dlatents=lerp(dlatent_avg, src_dlatents, coefs)
            
        src_dlatents=src_dlatents[:,0,:].astype('float32')
        dlatents[(i*num_once):((i+1)*num_once),:]=src_dlatents
    print('get all z and w')
    print('truncation: '+str(truncation))
    
    return dlatents

    
def GetImg(Gs,start_index,num_img,num_once,output_path,resize=None):
    print('Generate Image')
    tmp=output_path+'/W.npy'
    dlatents=np.load(tmp) 
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    
    all_images=[]
    for i in range(int(num_img/num_once)):
        print(i)
        images=[]
        for k in range(num_once):
            tmp=dlatents[i*num_once+k]
            if len(tmp.shape) < 2:
                tmp=tmp[None,None,:]
                tmp=np.tile(tmp,(1,Gs.components.synthesis.input_shape[1],1))
            image2= Gs.components.synthesis.run(tmp, randomize_noise=False, output_transform=fmt)
            
            if resize is not None:
                img=Image.fromarray(image2[0]).resize((resize,resize),Image.LANCZOS)
                img=np.array(img)
                image2=img[None,:]
            
            images.append(image2)
            
        images=np.concatenate(images)
        
        all_images.append(images)
        
    all_images=np.concatenate(all_images)
    
    return all_images

def GetImgW(Gs,W,start_index,num_img,num_once,output_path,resize=None):
    dlatents=W[start_index:start_index + num_img] 
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    
    all_images=[]
    for i in range(int(num_img/num_once)):
        print(i)
        images=[]
        for k in range(num_once):
            tmp=dlatents[i*num_once+k]
            if len(tmp.shape) < 2:
                tmp=tmp[None,None,:]
                tmp=np.tile(tmp,(1,Gs.components.synthesis.input_shape[1],1))
            image2= Gs.components.synthesis.run(tmp, randomize_noise=False, output_transform=fmt)
            
            if resize is not None:
                img=Image.fromarray(image2[0]).resize((resize,resize),Image.LANCZOS)
                img=np.array(img)
                image2=img[None,:]
            
            images.append(image2)
            
        images=np.concatenate(images)
        
        all_images.append(images)
        
    all_images=np.concatenate(all_images)
    
    return all_images



def GetS(output_path,num_img):
    print('Generate S')
    tmp=output_path+'/W.npy'
    dlatents=np.load(tmp)[:num_img]
    print(f"dlatents shape : {dlatents.shape}")
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        Gs=LoadModel(dataset_name)
        Gs.print_layers()  #for ada
        select_layers1=GetSNames(suffix=None)  #None,'/mul_1:0','/mod_weight/read:0','/MatMul:0'
        if self.dataset_name != 'ffhq':    
            dlatents=dlatents[:,None,:]
            dlatents=np.tile(dlatents,(1,Gs.components.synthesis.input_shape[1],1))
        
        all_s = sess.run(
            select_layers1,
            feed_dict={'G_synthesis_1/dlatents_in:0': dlatents})
    
    layer_names=[layer.name for layer in select_layers1]
    save_tmp=[layer_names,all_s]
    print([s.shape for s in all_s])
    return save_tmp

    


def convert_images_to_uint8(images, drange=[-1,1], nchw_to_nhwc=False):
    """Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    if nchw_to_nhwc:
        images = np.transpose(images, [0, 2, 3, 1])
    
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    
    np.clip(images, 0, 255, out=images)
    images=images.astype('uint8')
    return images


def GetCodeMS(dlatents):
        m=[]
        std=[]
        for i in range(len(dlatents)):
            tmp= dlatents[i] 
            tmp_mean=tmp.mean(axis=0)
            tmp_std=tmp.std(axis=0)
            m.append(tmp_mean)
            std.append(tmp_std)
        return m,std



#%%
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--dataset_name',type=str,default='ffhq',
                    help='name of dataset, for example, ffhq')
    parser.add_argument('--code_type',choices=['w','s','s_mean_std','s_flat','images_1K','images_100k'],default='w')
    
    parser.add_argument('--no_truncation', action='store_false')
    
    parser.add_argument('--output_path',type=str,default=None,
                    help='root output path')
    
    parser.add_argument('--resize',type=int,default=None,
                    help='save image size')
    
    parser.add_argument('--num_img',type=int,default=None,
                    help='the number of images')
    
    parser.add_argument('--num_once',type=int,default=500,
                    help='the number of images')
    
    
    args = parser.parse_args()
    random_state=5
    num_img= args.num_img 
    num_once=args.num_once
    dataset_name=args.dataset_name
    truncation=args.no_truncation
    output_path=args.output_path
    
    if args.output_path is None:
        output_path='./npy/'+dataset_name
    print('output_path:',output_path)
    
    
    if not os.path.isfile('./model/'+dataset_name+'.pkl'):
        url='https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/'
        name='stylegan2-'+dataset_name+'-config-f.pkl'
        os.system('wget ' +url+name + '  -P  ./model/')
        os.system('mv ./model/'+name+' ./model/'+dataset_name+'.pkl')
    
    
    if not os.path.isdir(output_path):
        os.system('mkdir '+output_path)
    
    if args.code_type=='w':
        Gs=LoadModel(dataset_name=dataset_name)
        dlatents=GetCode(Gs,random_state,num_img,num_once,truncation)
        
        tmp=output_path+'/W'
        np.save(tmp,dlatents)
        
        
    elif args.code_type=='s':
        save_name='S'
        save_tmp=GetS(output_path,num_img=num_img)
        tmp=output_path+'/'+save_name
        with open(tmp, "wb") as fp:
            pickle.dump(save_tmp, fp)
        
    elif args.code_type=='s_mean_std':
        save_tmp=GetS(output_path,num_img=num_img)
        
        dlatents=save_tmp[1]
        m,std=GetCodeMS(dlatents)
        save_tmp=[m,std]
        print([s.shape for s in save_tmp[0]])   
        save_name='S_mean_std'
        tmp=output_path+'/'+save_name
        with open(tmp, "wb") as fp:
            pickle.dump(save_tmp, fp)
            
    elif args.code_type=='s_flat':
        save_tmp=GetS(output_path,num_img=num_img)
        dlatents=save_tmp[1]
        dlatents=np.concatenate(dlatents,axis=1)
        
        tmp=output_path+'/S_Flat'
        np.save(tmp,dlatents)
    
    elif args.code_type=='images':
        Gs=LoadModel(dataset_name=dataset_name)
        if dataset_name != "ffhq_real":
            all_images=GetImg(Gs,start_index=0,num_img=num_img,num_once=num_once,output_path=output_path,resize=args.resize)
            tmp=output_path+'/images'
            np.save(tmp,all_images)
        else:
            chunks = [0,20000 ,20000, 20000 ,10000]
            tmp=output_path+'/W.npy'
            dlatents=np.load(tmp)
            for i in range(len(chunks[1:])):
                print(f"chunk : {i} size : {chunks[i]}")
                all_images=GetImgW(Gs,W=dlatents, start_index=sum(chunks[:i-1]) ,num_img=chunks[i],num_once=num_once,output_path=output_path,resize=args.resize)
                tmp=output_path+f'/images{i}'
                np.save(tmp,all_images)

            
    
    
    
