import torch
import torch.nn as nn
from tqdm import tqdm
import transformers
from utils import AverageMeter

def loss_function(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader,model,optimizer,device,scheduler):
    model.train()
    losses = AverageMeter()
    tk0=tqdm(data_loader,total=len(data_loader))
  
    for row in tk0:
        ids = row["ids"]
        token_type_ids = row["token_type_ids"]
        mask = row["mask"]
        targets = row["targets"]
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        optimizer.zero_grad()
        outputs = model(ids=ids,mask=mask,token_type_ids=token_type_ids,)
        loss = loss_function(outputs,targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(),ids.size(0))
        tk0.set_postfix(loss=losses.avg)

def eval_fn(data_loader, model, device):
    model.eval()
    final_targets = []
    final_output = []
    
    with torch.no_grad():
        for idx,row in tqdm(enumerate(data_loader),total=len(data_loader)):
            ids = row["ids"]
            token_type_ids = row["token_type_ids"]
            mask = row["mask"]
            targets = row["targets"]
            
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device,dtype=torch.long)
            mask = mask.to(device,dtype=torch.long)
            targets = targets.to(device,dtype=torch.float)
            
            outputs = model(ids=ids,mask=mask,token_type_ids=token_type_ids)
            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_output.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return final_output,final_targets


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased",return_dict=False)
        self.bert_drop = nn.Dropout(0.3)
        self.relu =  nn.ReLU()
        self.linear1 = nn.Linear(768,512)
        self.linear2 = nn.Linear(512,128)
        self.linear3 = nn.Linear(128,1)
        
    def forward(self,ids,mask,token_type_ids):
        sequence_output,pooled_output = self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids) 
        ## pooled_output shape(batch_size,768)
        
        bert_output = self.bert_drop(pooled_output)
        relu_output=self.relu(bert_output)
        
        linear_output1=self.linear1(relu_output) ###takes 768 dim vectors
        ## linear_output1 shape (batch_size,512)
        linear_output2 = self.linear2(linear_output1)
        ## linear_output2 shape (batch_size,128)
        final_output = self.linear3(linear_output2)
        ## linear_output2 shape (batch_size,1)        
        return final_output



