import numpy as np
import pandas as pd 
import os 
import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data 
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from nyc_taxicabdata import attribute_index
import time 
import matplotlib.pyplot as plt 
from collections import defaultdict
from sklearn.metrics import mean_squared_error

from tqdm import tqdm, tqdm_notebook, tnrange
tqdm.pandas(desc='Progress')

FULL_DATA_PATH = '/localdisk3/nyc_2021-09_updated.csv'
device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TabularModel(nn.Module):
    
    def __init__(self, emb_sizes, n_cont, out_szs, layers, p=0.4):
        super().__init__()
        
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_sizes])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layer_list = []
        n_emb = sum([nf for ni,nf in emb_sizes])
        n_in = n_emb + n_cont
        
        for i in layers:
            layer_list.append(nn.Linear(n_in, i))
            layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.BatchNorm1d(i))
            layer_list.append(nn.Dropout(p))
            n_in = i
            
        layer_list.append(nn.Linear(layers[-1], out_szs))
        
        self.layers = nn.Sequential(*layer_list)
        
    def forward(self, x_cat, x_cont):
        embeddings = []
        
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
            
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x,x_cont], 1)
        x = self.layers(x)
        
        return x

def emb_init(x):
    x = x.weight.data
    sc = 2/(x.size(1)+1)
    x.uniform_(-sc,sc)

class RegressionColumnarDataset(data.Dataset):
    def __init__(self, df, cats, y):
        self.dfcats = df[cats]
        self.dfconts = df.drop(cats, axis=1)

        self.cats = np.stack([c.values for n,c in self.dfcats.items()], axis=1).astype(np.int64)
        self.conts = np.stack([c.values for n,c in self.dfconts.items()], axis=1).astype(np.float32)
        self.y = y.values.astype(np.float32)
    
    def __len__(self): return len(self.y)

    def __getitem__(self, index):
        return [self.cats[index], self.conts[index], self.y[index]]

class MixedInputModel(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops, y_range, use_bn=True):
        super().__init__()
        for i,(c,s) in enumerate(emb_szs): assert c > 1, f"cardinality must be >=2, got emb_szs[{i}]: ({c},{s})"
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c,s in emb_szs])
        for emb in self.embs: emb_init(emb)
        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont=n_emb, n_cont
        
        szs = [n_emb+n_cont] + szs
        self.lins = nn.ModuleList([nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(sz) for sz in szs[1:]])
        for o in self.lins: nn.init.kaiming_normal_(o.weight.data)
        self.outp = nn.Linear(szs[-1], out_sz)
        nn.init.kaiming_normal_(self.outp.weight.data)

        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn,self.y_range = use_bn,y_range

    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embs)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x2 = self.bn(x_cont)
            x = torch.cat([x, x2], 1) if self.n_emb != 0 else x2
        for l,d,b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            if self.use_bn: x = b(x)
            x = d(x)
        x = self.outp(x)
        if self.y_range:
            x = torch.sigmoid(x)
            x = x*(self.y_range[1] - self.y_range[0])
            x = x+self.y_range[0]
        return x.squeeze()

def prepare_dataset(df):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['PU_Hour'] = df['tpep_pickup_datetime'].dt.hour
    df['PU_am_or_pm'] = np.where(df['PU_Hour']<12, 'am', 'pm')
    df['PU_weekday'] = df['tpep_pickup_datetime'].dt.strftime("%a")

    df['DO_Hour'] = df['tpep_dropoff_datetime'].dt.hour
    df['DO_am_or_pm'] = np.where(df['DO_Hour']<12, 'am', 'pm')
    df['DO_weekday'] = df['tpep_dropoff_datetime'].dt.strftime("%a")
    y = np.log(df['total_amount'])
    df = df.drop(['total_amount', 'tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)
    return df, y 



def split_features(df):
    categorical_columns = ['PU_Hour', 'DO_Hour', 'PU_am_or_pm', 'DO_am_or_pm', 'PU_weekday', 'DO_weekday']
    continuous_columns = ['trip_distance', 'PULong', 'PULat', 'DOLong', 'DOLat', 'passenger_count']
    for c in categorical_columns:
        df[c] = df[c].astype('category').cat.as_ordered()
        df[c] = df[c].cat.codes+1
    
    return categorical_columns, continuous_columns


def scale_contf(df, contf):
    scaler = preprocessing.StandardScaler().fit(df)
    cols = contf
    index = df.index
    scaled = scaler.transform(df[contf])
    scaled = pd.DataFrame(scaled, columns=cols, index=index)
    return pd.concat([scaled, df.drop(contf, axis=1)], axis=1)

def inv_y(y): return np.exp(y)

def rmse(targ, y_pred):
    return np.sqrt(mean_squared_error(inv_y(y_pred), inv_y(targ))) #.detach().numpy()

def train(model, train_dl, val_dl, loss_fn, opt, scheduler, epochs=3):
    num_batch = len(train_dl)
    lr = defaultdict(list)
    tloss = defaultdict(list)
    vloss = defaultdict(list)
    for epoch in tnrange(epochs):      
        y_true_train = list()
        y_pred_train = list()
        total_loss_train = 0          
        
        t = tqdm_notebook(iter(train_dl), leave=False, total=num_batch)
        for cat, cont, y in t:
            cat = cat.cuda()
            cont = cont.cuda()
            y = y.cuda()
            
            t.set_description(f'Epoch {epoch}')
            
            opt.zero_grad()
            pred = model(cat, cont)
            loss = loss_fn(pred, y)
            loss.backward()
            lr[epoch].append(opt.param_groups[0]['lr'])
            tloss[epoch].append(loss.item())
            opt.step()
            scheduler.step()
            
            t.set_postfix(loss=loss.item())
            
            y_true_train += list(y.cpu().data.numpy())
            y_pred_train += list(pred.cpu().data.numpy())
            total_loss_train += loss.item()
            
        train_acc = rmse(y_true_train, y_pred_train)
        train_loss = total_loss_train/len(train_dl)

        if val_dl:
            y_true_val = list()
            y_pred_val = list()
            total_loss_val = 0
            for cat, cont, y in tqdm_notebook(val_dl, leave=False):
                cat = cat.cuda()
                cont = cont.cuda()
                y = y.cuda()
                pred = model(cat, cont)
                loss = loss_fn(pred, y)
                
                y_true_val += list(y.cpu().data.numpy())
                y_pred_val += list(pred.cpu().data.numpy())
                total_loss_val += loss.item()
                vloss[epoch].append(loss.item())
            valacc = rmse(y_true_val, y_pred_val)
            valloss = total_loss_val/len(val_dl)
            print(f'Epoch {epoch}: train_loss: {train_loss:.4f} train_rmse: {train_acc:.4f} | val_loss: {valloss:.4f} val_rmse: {valacc:.4f}')
        else:
            print(f'Epoch {epoch}: train_loss: {train_loss:.4f} train_rmse: {train_acc:.4f}')
    
    return lr, tloss, vloss

def load_data(path):
    df = pd.read_csv(path)
    df.columns = attribute_index.keys()
    # print(df.head())
    # print(df.keys())
    df = df.drop(['VendorID', 'RatecodeID', 'store_and_fwd_flag',
       'PULocationID', 'DOLocationID', 'payment_type', 'extra',
       'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
       'fare_amount', 'congestion_surcharge', 'airport_fee', 'ID'], axis=1)
    # print(df.head())
    df, y = prepare_dataset(df)
    catf, contf = split_features(df)
    print(catf)
    print(contf)
    # X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0)
    df =df.reset_index()
    X_train = df
    y_train = y
    print(X_train.shape)
    print(y_train.shape)
    # X_train_sc = scale_contf(X_train, contf)
    trainds = RegressionColumnarDataset(X_train, catf, y_train)
    params = {
        'batch_size' : 128,
        'shuffle' : True,
        'num_workers' : 8
    }
    traindl = data.DataLoader(trainds, **params, drop_last=True)
    y_range = (0, y_train.max()*1.2)
    cat_sz = [(c, df[c].max()+1) for c in catf]
    emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]


    model = MixedInputModel(emb_szs=emb_szs, 
                    n_cont=len(df.columns)-len(catf), 
                    emb_drop=0.04, 
                    out_sz=1, 
                    szs=[1000,500,250], 
                    drops=[0.001,0.01,0.01], 
                    y_range=y_range).to(device)

    opt = optim.Adam(model.parameters(), 1e-2)
    lr_cosine = lr_scheduler.CosineAnnealingLR(opt, 1000)

    

    lr, tloss, _ = train(model=model, train_dl=traindl, val_dl=None, loss_fn=F.mse_loss, opt=opt, scheduler=lr_cosine, epochs=100)

    # print(X_test.shape)
    # print(df.head())
    # print(df.info())

    # categorical_columns = ['PU_Hour', 'DO_Hour', 'PU_am_or_pm', 'DO_am_or_pm', 'PU_weekday', 'DO_weekday']
    # continuous_columns = ['trip_distance', 'PULong', 'PULat', 'DOLong', 'DOLat', 'passenger_count']
    # y_column = ['total_amount']

    # for cat in categorical_columns:
    #     df[cat] = df[cat].astype('category')
    
    # print(df.dtypes)

    # PU_hr = df['PU_Hour'].cat.codes.values
    # DO_hr = df['DO_Hour'].cat.codes.values
    # PU_am_pm = df['PU_am_or_pm'].cat.codes.values
    # DO_am_pm = df['DO_am_or_pm'].cat.codes.values
    # PU_wkdy = df['PU_weekday'].cat.codes.values
    # DO_wkdy = df['DO_weekday'].cat.codes.values 

    # cats = np.stack([PU_hr, DO_hr, PU_am_pm, DO_am_pm, PU_wkdy, DO_wkdy], axis=1)
    # cats = torch.tensor(cats, dtype=torch.int64)

    # # stacking continuous columns 
    # conts = np.stack([df[col].values for col in continuous_columns], axis=1)
    # conts = torch.tensor(conts, dtype=torch.float)

    # # converting the total amount column
    # y = torch.tensor(df[y_column].values).flatten()
    # y = y.type(torch.LongTensor)


    # ### Defining embedding sizes

    # cat_sizes = [len(df[col].cat.categories) for col in categorical_columns]
    # embedding_sizes = [(size, min(50, (size+1)//2)) for size in cat_sizes]
    # self_embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in embedding_sizes])
    # print(self_embeds)

    # # model training
    # torch.manual_seed(33)
    # model = TabularModel(embedding_sizes, conts.shape[1], 1, [512,256,128,64,32,16,8,4], p=0.4)
    # # model = model.to(0)
    # print(model)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # start_time = time.time()
    # final_losses = []
    # for epochs in range(500):
    #     optimizer.zero_grad()
    #     # cats, conts = cats.to(0), conts.to(0)
    #     # y = y.to(0)
    #     y_pred = model(cats, conts)
    #     loss = criterion(y_pred, y)
    #     final_losses.append(loss)
    #     loss.backward()
    #     optimizer.step()
    #     print(f"Epoch {epochs+1}, loss: {loss.item()}")
    
    # duration = time.time() - start_time
    # print(f"Training took {duration/60} minutes")

    # plt.plot(range(epochs), final_losses)
    # plt.ylabel('Cross Entropy Loss')
    # plt.xlabel('epoch')
    # plt.savefig('./figures/nyc_full_data_loss.png')


    # Evaluate test set 
    # get the test data, make it from the entire dataset of the rows that are not selected 
    # with torch.no_grad():
    #     y_val = model(cat_test, con_test)
    #     loss = criterion(y_val, y_test)
    # print(f'CE Loss: {loss:.8f}')



if __name__ == '__main__':
    load_data(FULL_DATA_PATH)
