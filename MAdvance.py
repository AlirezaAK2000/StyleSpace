import os
import time

from numpy.lib.function_base import select
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



from manipulate import Manipulator
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json
import shutil
from tqdm import tqdm
import argparse

def lp2istr(x):
    return str(x[0])+'_'+str(x[1])

class MAdvance(Manipulator):
    
    def __init__(self,dataset_name , model_name , positive_bank , num_pos , exempelar_set,save_selected):
        super().__init__(dataset_name, model_name)
        self.positive_bank=positive_bank
        self.dataset_name = dataset_name
        self.num_pos=num_pos #example
        self.exempelar_set = exempelar_set
        self.img_path_exm=self.file_path+'npy/'+self.exempelar_set+'/'
        self.save_selected = save_selected

        self.num_m=10 #number of output
        self.threshold1=0.5 #pass this ratio
        self.threshold2=0.25 #gap between first and second
        self.w=np.load(f'{self.img_path_exm}W.npy')
        
        print(f'shape of W: {self.w.shape}')
        
        self.code_mean2=np.concatenate(self.code_mean)
        self.code_std2=np.concatenate(self.code_std)
        
        fmaps=[512, 512, 512, 512, 512, 256, 128,  64, 32]
        self.fmaps=np.repeat(fmaps,3)
        
        try:
            self.LoadSemantic()
        except FileNotFoundError:
            print('semantic_top_32 not exist')
        
        try:
            self.results=pd.read_csv(self.img_path_exm+'attribute')
            print(f'attribute file loaded from {self.img_path_exm}')
        except FileNotFoundError:
            print('attribute not exist')
    
    def LoadSemantic(self):
        name='semantic_top_32'
        with open(self.img_path+name, 'rb') as handle:
            all_semantic_top = pickle.load(handle)

            
        self.all_semantic_top2=np.concatenate(all_semantic_top)
        self.num_semantic=self.all_semantic_top2.shape[1] #ignore low frequency area, bed 10
        
        tmp=pd.read_csv(self.img_path+'label')
        self.label=tmp['names']
    
    def RemovePG(self,l_p,findex=None): #l_p or indexs2
        for i in self.pindexs:
            select=l_p[:,0]==i
            l_p=l_p[~select]
            if not findex is None:
                findex=findex[~select]
        
        if findex is None:
            return l_p
        else:
            return l_p,findex
    
    def GetRank(self,target_index):
        top_sum=self.all_semantic_top2[:,target_index].sum(axis=1)
        
        tmp=list(np.arange(self.num_semantic))
        for i in target_index:
            tmp.remove(i)
        tmp=self.all_semantic_top2[:,tmp] #all the rest semantic 
        second_max=tmp.max(axis=1)
        
        select1=top_sum>self.threshold1
        select2=top_sum-second_max>self.threshold2
        
        select=np.logical_and(select1,select2)
        findex=np.arange(len(select))[select]
        l_p=self.GetLCIndex(findex)
        
        index2=np.zeros([len(l_p),3])
        index2[:,2]=top_sum[findex]
        index2[:,(0,1)]=l_p
        
        select_index=np.argsort(index2[:,2])[::-1]
        index2=index2[select_index]
        findex=findex[select_index]

        index2,findex2=self.RemovePG(index2,findex)
        return index2,findex2
    
    def AllCheck(self,positive=True):
        
        tmp_save=self.num_pos
        # self.num_pos=self.positive_bank
        
        positive_train,_=self.SimulateInput(positive)
        index2,_=self.GetComponent(positive_train)
        
        self.num_pos=tmp_save
        lp_sort=pd.DataFrame(index2[:,-1])
        lp_sort.index=list(map(lp2istr, index2[:,:-1].astype(int)))
        
        return index2,lp_sort
    
    def SimulateInput(self,positive=True):
        print('bname: '+str(self.bname))
        tmp_indexs=self.results[self.bname].argsort()
        if positive:
            tmp=tmp_indexs[:self.positive_bank]
        else:
            tmp=tmp_indexs[-self.positive_bank:]
        scores = np.array([self.results[self.bname][i] * -1 for i in tmp])
        scores_normed = (scores - np.min(scores))/(np.max(scores) - np.min(scores))
        probes = scores_normed / np.sum(scores_normed)
        positive_indexs=np.random.choice(tmp,size=self.num_pos,replace=False,p=probes)
        
        print(self.save_selected)
        if self.save_selected:
            images = np.load(f'{self.img_path_exm}images.npy')
            
            dirrr = f'selected/{dataset_name}_{args.exempelar_set}/{M.bname}'
            # os.mkdir(dirrr)
            logits = {}
            for i in tqdm(positive_indexs):
                img = Image.fromarray(images[i])
                img.save(f"{dirrr}/{i}.png")
                logits[str(i)] = self.results[self.bname][i]
            
            with open(f'{dirrr}/logits.json' , 'w') as f:
                json.dump(logits,f)
            
        tmp=self.w[positive_indexs] #only use 50 images
        tmp_dlatents = None
        if self.exempelar_set == 'ffhq':
            tmp=tmp[:,None,:]
            w_plus=np.tile(tmp,(1,self.Gs.components.synthesis.input_shape[1],1))
            tmp_dlatents=self.W2S(w_plus)
        else:
            tmp_dlatents=self.W2S(tmp)
            
        
        positive_train=[tmp for tmp in tmp_dlatents]
        return positive_train,positive_indexs
    
    def GetComponent(self,positive_train): #sort s2n, remove pg, 
        
        feature_s2n=self.S2N(positive_train)
        
        feature_index=feature_s2n.argsort()
        findex=feature_index[::-1] #index in concatenate form 
        
        l_p=self.GetLCIndex(findex)
        
        index2=np.zeros([len(l_p),3])
        index2[:,2]=feature_s2n[findex]
        index2[:,(0,1)]=l_p
        
        index2,findex2=self.RemovePG(index2,findex)
        return index2,findex
    
    def S2N(self,positive_train):
        positive_train2=np.concatenate(positive_train,axis=1)
        normalize_positive=(positive_train2-self.code_mean2)/self.code_std2
        
        feature_mean=np.abs(normalize_positive.mean(axis=0))
        feature_std=normalize_positive.std(axis=0)
        
        feature_s2n=feature_mean/feature_std
        return feature_s2n
    
    def GetLCIndex(self,findex):
        l_p=[]
        cfmaps=np.cumsum(self.fmaps)
        for i in range(len(findex)):
            tmp_index=findex[i]
            tmp=tmp_index-cfmaps
            tmp=tmp[tmp>0]
            lindex=len(tmp)
            if lindex==0:
                cindex=tmp_index
            else:
                cindex=tmp[-1]
            
            if cindex ==self.fmaps[lindex]:
                cindex=0
                lindex+=1
            l_p.append([lindex,cindex])
        l_p=np.array(l_p)
        return l_p
    
    #%%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    
    
    parser.add_argument('--dataset_name',type=str,default=None,
                    help='dataset name')
    
    parser.add_argument('--model_name',type=str,default=None,
                    help='generator name')
    
    parser.add_argument('--bname',type=str,default=None,
                    help='attribute')
    
    parser.add_argument('--positive_bank',type=int,default=None,
                    help='positive_bank')
    
    parser.add_argument('--num_pos',type=int,default=None,
                    help='attribute')
    
    parser.add_argument('--start_index',type=int,default=None,
                    help='start index')
    
    parser.add_argument('--exempelar_set',type=str,default=None,
                    help='attribute')
    
    parser.add_argument('--save_selected',type=bool,default=None,
                    help='save selected images')
    
    args = parser.parse_args()
    
    start = time.time()
    dataset_name= args.dataset_name
    model_name = args.model_name
    save_selected = args.save_selected
    
    
    M=MAdvance(dataset_name=dataset_name 
               ,model_name=model_name
               ,positive_bank=args.positive_bank
               ,num_pos=args.num_pos
               ,exempelar_set=args.exempelar_set
               ,save_selected=save_selected)
    np.set_printoptions(suppress=True)
    #%%
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
    
    M.bname= features[args.bname] #01-smiling, 37-wearing-lipstick,13-blond-hair
    dirrr = f'selected/{dataset_name}_{args.exempelar_set}/{M.bname}'
    if os.path.exists(dirrr):
        shutil.rmtree(dirrr)
    os.system(f'mkdir {dirrr} --p')
    
#    lp_sort=M.ConsistenceCheck(num_run=1000)
    
    lp_candidate,lp_sort= M.AllCheck(positive=True)
    # lp_sort = lp_sort[lp_sort.index.isin(filter(lambda x : int(x.split('_')[0]) <= 18 , lp_sort.index))]
    plt.figure()
    plt.title(f"{M.bname} ({dataset_name}_{args.exempelar_set})")
    plt.plot(lp_sort[:10],'*')
    plt.ylabel('signal2noise')
    plt.xlabel('(layer_index, channel_index)')
    plt.savefig(dirrr)
    print('fig saved')
    # plt.show()
    #%%
    
    M.alpha=[-15,-10,-5,0,5,10,15]
    M.img_index= args.start_index
    M.num_images=10
    start=0

    os.system(f'mkdir html/{dataset_name}_{args.exempelar_set} --p')
    for i in tqdm(range(10),desc='visual chanel generation'):
        tmp=lp_sort.index[start+i]
        lindex,bname=np.array(tmp.split('_')) 
        lindex,bname=int(lindex),int(bname)
        
        M.manipulate_layers=[lindex]
        codes,out=M.EditOneC(bname) 
        tmp=f'{dataset_name}_{args.exempelar_set}/'+str(M.manipulate_layers)+'_'+str(bname)
        M.Vis(tmp,'c',out)
    
    print(f"finish time : {round((time.time() - start)/60 , 2)} mins")
    #%%
    
    # num_view=5
    # target_index=(10,)
    # lp_candidate,_=M.GetRank(target_index)
    # print(lp_candidate.shape)
    # #%%
    
    # M.alpha=[-20,-10,-5,0,5,10,20]
    # M.img_index=0
    # M.num_images=20
    # start=0
    
    # for i in range(num_view):
        
    #     lindex,bname,_=lp_candidate[start+i].astype(int)
    #     lindex=int(lindex)
    #     M.manipulate_layers=[lindex]
    #     codes,out=M.EditOneC(bname)
    #     tmp=str(M.manipulate_layers)+'_'+str(bname)
    #     M.Vis(tmp,'c',out)
    # #%%
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
