

import os
import os.path
import pickle
from turtle import shape
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from dnnlib import tflib
from utils.visualizer import HtmlPageVisualizer
from PIL import Image
import matplotlib.pyplot as plt
import json
import argparse
import pickle
from functools import reduce
from concurrent.futures import ThreadPoolExecutor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def Vis(bname,suffix,out,rownames=None,colnames=None, logits=None , save=False):
    num_images=out.shape[0]
    step=out.shape[1]
    
    # if colnames is None:
    #     colnames=[f'Step {i:02d}' for i in range(1, step + 1)]
    # if rownames is None:
    #     rownames=[str(i) for i in range(num_images)]
    
    
    # visualizer = HtmlPageVisualizer(
    #   num_rows=num_images, num_cols=step + 1, viz_size=256)
    # visualizer.set_headers(
    #   ['Name'] +colnames)
    
    # for i in range(num_images):
    #     visualizer.set_cell(i, 0, text=rownames[i])
    
    
    row_size = step
    col_size = num_images
    resolution = 64
    plt.figure(figsize=(5 * step, 5 * num_images) , dpi=50)
    for i in range(num_images):
        for k in range(step):
            image=out[i,k,:,:,:]
            plt.subplot(num_images , row_size, i * row_size + k + 1)
            plt.axis("off")
            if logits != None:
                plt.title(round(logits[i][k],6))
            plt.imshow(Image.fromarray(image))
            if save:
                Image.fromarray(image).save(f'{i}_{k}.jpg')
            # visualizer.set_cell(i, 1+k, image=image)
    plt.savefig(f'./html/'+bname+'_'+suffix+'.png')
    # Save results.
    # visualizer.save(f'./html/'+bname+'_'+suffix+'.html')




def LoadData(img_path):
    tmp=img_path+'S'
    with open(tmp, "rb") as fp:   #Pickling
        s_names,all_s=pickle.load( fp)
    dlatents=all_s
    
    pindexs=[]
    mindexs=[]
    for i in range(len(s_names)):
        name=s_names[i]
        if not('ToRGB' in name):
            mindexs.append(i)
        else:
            pindexs.append(i)
    
    tmp=img_path+'S_mean_std'
    with open(tmp, "rb") as fp:   #Pickling
        m,std=pickle.load( fp)
    
    return dlatents,s_names,mindexs,pindexs,m,std


def LoadModel(model_path,model_name):
    # Initialize TensorFlow.
    tflib.init_tf()
    tmp=os.path.join(model_path,model_name)
    with open(tmp, 'rb') as f:
        _, _, Gs = pickle.load(f)
    Gs.print_layers()
    return Gs

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


def convert_images_from_uint8(images, drange=[-1,1], nhwc_to_nchw=False):
    """Convert a minibatch of images from uint8 to float32 with configurable dynamic range.
    Can be used as an input transformation for Network.run().
    """
    if nhwc_to_nchw:
        images=np.rollaxis(images, 3, 1)
    return images/ 255 *(drange[1] - drange[0])+ drange[0]




def check_manipulation(images , classifier):
    
    logits = []
    variances = []
    for i in range(images.shape[0]):
        tmp_imgs = images[i,:,:,:,:]
        tmp_imgs = convert_images_from_uint8(tmp_imgs, drange=[-1,1], nhwc_to_nchw=True)
        tmp = classifier.run(tmp_imgs, None)
        tmp1=tmp.reshape(-1)
        logits.append(tmp1)
        variances.append(np.var(tmp1))
    return logits , variances        

def resize_images(out,shp):
    resized_out = []
    for i in range(out.shape[0]):
        images = []
        for j in range(out.shape[1]):
            image = np.array(Image.fromarray(out[i,j,:,:,:]).resize(shp))
            image = image[None , :]
            images.append(image)
        images = np.concatenate(images)
        resized_out.append(images[None,:])
    resized_out = np.concatenate(resized_out)
    return resized_out


class Manipulator():
    def __init__(self,dataset_name, model_name , classifier = None):
        self.file_path='./'
        self.classifier = classifier
        self.img_path=self.file_path+'npy/'+dataset_name+'/'
        self.model_path=self.file_path+'model/'
        self.dataset_name=dataset_name
        self.model_name=model_name+'.pkl'
        
        self.alpha=[0] #manipulation strength 
        self.num_images=10
        self.img_index=0  #which image to start 
        self.viz_size=256
        self.manipulate_layers=None #which layer to manipulate, list
        
        self.dlatents,self.s_names,self.mindexs,self.pindexs,self.code_mean,self.code_std=LoadData(self.img_path)
        
        self.sess=tf.InteractiveSession()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.Gs=LoadModel(self.model_path,self.model_name)
        self.num_layers=len(self.dlatents)
        
        self.Vis=Vis
        self.noise_constant={}
        
        for i in range(len(self.s_names)):
            tmp1=self.s_names[i].split('/')
            if not 'ToRGB' in tmp1:
                tmp1[-1]='random_normal:0'
                size=int(tmp1[1].split('x')[0])
                tmp1='/'.join(tmp1)
                tmp=(1,1,size,size)
                self.noise_constant[tmp1]=np.random.random(tmp)
        
        tmp=self.Gs.components.synthesis.input_shape[1]
        d={}
        d['G_synthesis_1/dlatents_in:0']=np.zeros([1,tmp,512])
        names=list(self.noise_constant.keys())
        tmp=tflib.run(names,d)
        for i in range(len(names)):
            self.noise_constant[names[i]]=tmp[i]
        
        self.fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.img_size=self.Gs.output_shape[-1]
    
    def GenerateImg(self,codes):
        

        num_images,step=codes[0].shape[:2]

            
        out=np.zeros((num_images,step,self.img_size,self.img_size,3),dtype='uint8')
        for i in range(num_images):
            for k in range(step):
                d={}
                for m in range(len(self.s_names)):
                    d[self.s_names[m]]=codes[m][i,k][None,:]  #need to change
                d['G_synthesis_1/4x4/Const/Shape:0']=np.array([1,18,  512], dtype=np.int32)
                d.update(self.noise_constant)
                img=tflib.run('G_synthesis_1/images_out:0', d)
                image=convert_images_to_uint8(img, nchw_to_nhwc=True)
                out[i,k,:,:,:]=image[0]
        return out
    
    
    
    def MSCode(self,dlatent_tmp,boundary_tmp , candidates = []):
        
        step=len(self.alpha)
        dlatent_tmp1 = None
        if candidates != []:
            dlatent_tmp1 = [tmp.reshape((len(candidates),-1)) for tmp in dlatent_tmp]
        else:
            dlatent_tmp1 = [tmp.reshape((self.num_images,-1)) for tmp in dlatent_tmp]
        dlatent_tmp2 = [np.tile(tmp[:,None],(1,step,1)) for tmp in dlatent_tmp1] # (10, 7, 512)

        l=np.array(self.alpha)
        l=l.reshape(
                    [step if axis == 1 else 1 for axis in range(dlatent_tmp2[0].ndim)])
        
        if type(self.manipulate_layers)==int:
            tmp=[self.manipulate_layers]
        elif type(self.manipulate_layers)==list:
            tmp=self.manipulate_layers
        elif self.manipulate_layers is None:
            tmp=np.arange(len(boundary_tmp))
        else:
            raise ValueError('manipulate_layers is wrong')
            
        for i in tmp:
            dlatent_tmp2[i]+=l*boundary_tmp[i]
        
        codes=[]
        for i in range(len(dlatent_tmp2)):
            tmp=list(dlatent_tmp[i].shape)
            tmp.insert(1,step)
            codes.append(dlatent_tmp2[i].reshape(tmp))
        return codes
    
    
    def EditOne(self,bname,dlatent_tmp=None):
        if dlatent_tmp==None:
            dlatent_tmp=[tmp[self.img_index:(self.img_index+self.num_images)] for tmp in self.dlatents]
        
        boundary_tmp=[]
        for i in range(len(self.boundary)):
            tmp=self.boundary[i]
            if len(tmp)<=bname:
                boundary_tmp.append([])
            else:
                boundary_tmp.append(tmp[bname])
        
        codes=self.MSCode(dlatent_tmp,boundary_tmp)
            
        out=self.GenerateImg(codes)
        return codes,out
    
    def EditOneC(self,cindex,dlatent_tmp=None , candidates= []): 
        if dlatent_tmp==None:
            if candidates != []:
                dlatent_tmp=[tmp[candidates] for tmp in self.dlatents]
            else:
                dlatent_tmp=[tmp[self.img_index:(self.img_index+self.num_images)] for tmp in self.dlatents]

                
        
        boundary_tmp=[[] for i in range(len(self.dlatents))]
        
        #'only manipulate 1 layer and one channel'
        assert len(self.manipulate_layers)==1 
        
        ml=self.manipulate_layers[0]
        tmp=dlatent_tmp[ml].shape[1] #ada
        tmp1=np.zeros(tmp)
        tmp1[cindex]=self.code_std[ml][cindex]  #1
        boundary_tmp[ml]=tmp1
        
        codes=self.MSCode(dlatent_tmp,boundary_tmp, candidates=candidates)
        out=self.GenerateImg(codes)
        return codes,out
    
        
    def W2S(self,dlatent_tmp):
        
        all_s = self.sess.run(
            self.s_names,
            feed_dict={'G_synthesis_1/dlatents_in:0': dlatent_tmp})
        return all_s
        
    
    
    def find_related_attributes(self, S ,codes , image_attributes , cand):
        # codes must be flattened before assignment 
        classes = []
        close_img = []
        for i in tqdm(range(codes.shape[0]) , desc="image distance"):
            img_index = cand[i]
            image_att = set(image_attributes[str(img_index)])
            all_att = set()
            cm = []
            for j in range(codes.shape[1]):
                img_vec = codes[i,j,:]
                distances = (np.sum((S * img_vec) ** 2 , axis=1)) ** 1/2
                sorted_distances = np.argsort(distances)
                ch_vecs = sorted_distances[-5:]
                cm.append(ch_vecs)
                for k in ch_vecs:
                    all_att = all_att.union(set(image_attributes[str(k)]).difference(image_att))
            close_img.append(cm)
            classes.append(list(all_att))    
        
        return close_img, classes    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    
    parser.add_argument('--bname',type=str,default=None,
                help='attribute')
    
    args = parser.parse_args()
    
    
    features = {'male':'00-male', 'smiling':'01-smiling', 'attractive':'02-attractive', 'wavy-hair':'03-wavy-hair', 'young':'04-young',
       '5-o-clock-shadow':'05-5-o-clock-shadow', 'arched-eyebrows':'06-arched-eyebrows', 'bags-under-eyes':'07-bags-under-eyes',
       'bald':'08-bald', 'bangs':'09-bangs', 'big-lips':'10-big-lips', 'big-nose':'11-big-nose', 'black-hair':'12-black-hair',
       'blond-hair':'13-blond-hair', 'blurry':'14-blurry', 'brown-hair':'15-brown-hair', 'bushy-eyebrows':'16-bushy-eyebrows',
       'chubby':'17-chubby', 'double-chin':'18-double-chin', 'eyeglasses':'19-eyeglasses', 'goatee':'20-goatee',
       'gray-hair':'21-gray-hair', 'heavy-makeup':'22-heavy-makeup', 'high-cheekbones':'23-high-cheekbones',
       'mouth-slightly-open':'24-mouth-slightly-open', 'mustache':'25-mustache', 'narrow-eyes':'26-narrow-eyes',
       'no-beard':'27-no-beard', 'oval-face':'28-oval-face', 'pale-skin':'29-pale-skin', 'pointy-nose':'30-pointy-nose',
       'receding-hairline':'31-receding-hairline', 'rosy-cheeks':'32-rosy-cheeks', 'sideburns':'33-sideburns',
       'straight-hair':'34-straight-hair', 'wearing-earrings':'35-wearing-earrings', 'wearing-hat':'36-wearing-hat',
       'wearing-lipstick':'37-wearing-lipstick', 'wearing-necklace':'38-wearing-necklace', 'wearing-necktie':'39-wearing-necktie'}
    
    
    
    bname = features[args.bname]
    
    tflib.init_tf()

    
    with open(f"metrics_checkpoint/celebahq-classifier-{bname}.pkl" , 'rb') as f:
        classifier = pickle.load(f)
    
    
    
    
    with open("npy/ffhq/candidates.json", 'r') as f:
        ffhq_candidates = json.load(f)
    
    ffhq_cand = np.array(reduce(lambda x,y : x + y , ffhq_candidates.values()))[:5]
    
        
    with open("npy/celeba-hq/candidates.json", 'r') as f:
        celeba_candidates = json.load(f)
    
    with open('npy/ffhq/img_cls.json' , 'r') as f:
        image_attributes = json.load(f)
    
    
    M=Manipulator(dataset_name='ffhq' , model_name='ffhq')
    
    #%%
    M.alpha=[-20,-10,-5,0,5,10,20]
    # M.num_images=5
    # M.img_index = 0
    lindex,cindex=15,45
    
    M.manipulate_layers=[lindex]
    codes,out=M.EditOneC(cindex , candidates=ffhq_cand) #dlatent_tmp
    
    a = M.find_related_attributes(np.concatenate(M.dlatents, axis=1) , np.concatenate(codes,axis=2) , image_attributes=image_attributes,cand=ffhq_cand)
    print(a)
    resized_out = resize_images(out,(256,256))
    

    
    # logits, variances = check_manipulation(resized_out,classifier)
    

    # tmp=str(M.manipulate_layers)+'_'+str(cindex)
    # M.Vis(tmp,'c',out , save=False, logits=logits)
    
    # plt.plot(list(variances) , [i for i in range(len(variances))])
    # plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    




