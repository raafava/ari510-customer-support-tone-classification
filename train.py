"""Compare tone classifiers using supplied train/validation splits; seed 42."""
import argparse,json,hashlib,random
from pathlib import Path
import pandas as pd,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args();d=Path(a.data);o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
random.seed(42);np.random.seed(42)
frames={s:pd.read_csv(d/(s+'.csv')).fillna('') for s in ['train','validation','test']};sample=pd.read_csv(d/'sample_submission.csv')
for s,df in frames.items():assert df.id.is_unique
for s,t in [('train','validation'),('train','test'),('validation','test')]:assert set(frames[s].id).isdisjoint(frames[t].id)
y=frames['train'].label.astype(int);yv=frames['validation'].label.astype(int);labels=sorted(y.unique().tolist());assert labels==[1,2,3,4,5]
configs=[]
for source in ['last_customer_message','context']:
 for kind in ['word_logistic','char_logistic','word_neural']:
  configs.append((source,kind))
results=[]
for i,(source,kind) in enumerate(configs,1):
 run='run'+str(i)
 def texts(df):
  return df.last_customer_message if source=='last_customer_message' else df.conversation+' FINAL CUSTOMER: '+df.last_customer_message+' '+df.last_customer_message
 if kind=='char_logistic':v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,max_features=10000)
 else:v=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=5000)
 if kind=='word_neural':clf=MLPClassifier(hidden_layer_sizes=(32,),solver='lbfgs',alpha=1,max_iter=1000,random_state=42)
 else:clf=LogisticRegression(C=2,class_weight='balanced',max_iter=2000,random_state=42)
 model=make_pipeline(v,clf)
 print('Training',run,source,kind,flush=True);model.fit(texts(frames['train']),y)
 pred=model.predict(texts(frames['validation']));score=f1_score(yv,pred,average='macro',labels=labels,zero_division=0)
 # Keep the train-only model used for validation; no extra refit on validation.
 test=model.predict(texts(frames['test']));lookup=dict(zip(frames['test'].id,test));sub=pd.DataFrame({'id':sample.id,'label':[int(lookup[k]) for k in sample.id]})
 assert len(sub)==101 and sub.id.is_unique and set(sub.label)<=set(labels)
 sub.to_csv(o/(run+'_tone_submission.csv'),index=False)
 r={'run_id':run,'text_source':source,'approach':kind,'validation_macro_f1':score,'validation_accuracy':accuracy_score(yv,pred),'public_macro_f1':None,'seed':42,'model_settings':clf.get_params(),'vectorizer_settings':{k:str(val) for k,val in v.get_params().items()},'classification_report':classification_report(yv,pred,labels=labels,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(yv,pred,labels=labels).tolist()};results.append(r)
 print(run,'macro F1',score,flush=True)
(o/'results.json').write_text(json.dumps(results,indent=2))
best=max(results,key=lambda r:r['validation_macro_f1'])
import shutil
name='customer_tone_'+best['run_id']+'_submission.csv';shutil.copyfile(o/(best['run_id']+'_tone_submission.csv'),o/name)
print('BEST',best['run_id'],best['validation_macro_f1'],name,flush=True)
(o/'data_hashes.json').write_text(json.dumps({f:hashlib.sha256((d/f).read_bytes()).hexdigest() for f in ['train.csv','validation.csv','test.csv','sample_submission.csv']},indent=2))
