#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.
cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='ZIi4JSlFPUh15Iv7OK6KEhu-6YRGcAXD7wVajsKDZtnk',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.private.us.cloud-object-storage.appdomain.cloud')

bucket = 'liverdisease-donotdelete-pr-w9txx9dxbqrxld'
object_key = 'indian_liver.csv'

body = cos_client.get_object(Bucket=bucket,Key=object_key)['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()


# In[2]:


import pandas as pd
import numpy as np


# ## Read Dataset

# ## Data Cleaning

# In[3]:


dataset['Albumin_and_Globulin_Ratio'] = dataset['Albumin_and_Globulin_Ratio'].fillna(dataset['Albumin_and_Globulin_Ratio'].median())


# In[4]:


dataset['Gender'] = np.where(dataset['Gender']=='Male', 1,0)


# In[5]:


dataset = dataset.drop('Direct_Bilirubin', axis=1)


# In[6]:


X=dataset.drop(['Dataset'],axis='columns')
y=dataset['Dataset']


# In[7]:


X.head()


# In[8]:


#y = dataset.iloc[:, -1]
y.head()


# In[9]:


dataset.isnull().sum()


# ## Training and Testing

# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=50)


# ## Build Model

# In[11]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=40)
model.fit(x_train, y_train)


# In[12]:


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
accuracy_score(y_test, y_predict)


# In[13]:


pd.crosstab(y_test, y_predict)


# In[14]:


get_ipython().system('pip install ibm_watson_machine_learning')


# In[15]:


from ibm_watson_machine_learning import APIClient
import json


# In[16]:


wml_credentials = {
    "apikey":"hM7Z2Iv367dTufm5LGq4w-1l2CI7bfutV_mtXDiD2B7e",
    "url":"https://us-south.ml.cloud.ibm.com"
}


# In[17]:


wml_client = APIClient(wml_credentials)
wml_client.spaces.list()


# In[18]:


Space_Id="4f824129-7262-43eb-aa84-dd1b43aa342f"


# In[19]:


wml_client.set.default_space(Space_Id)


# In[20]:


wml_client.software_specifications.list(500)


# In[21]:


import sklearn
sklearn.__version__


# In[22]:


pip install scikit-learn==0.24


# In[23]:


MODEL_NAME = 'liver_disease'
DEPLOYMENT_NAME = 'liver_disease'
DEMO_MODEL = model


# In[24]:


software_spec_uid = wml_client.software_specifications.get_id_by_name('runtime-22.1-py3.9')


# In[25]:


# Setup model meta
model_props = {
    wml_client.repository.ModelMetaNames.NAME: MODEL_NAME, 
    wml_client.repository.ModelMetaNames.TYPE: 'scikit-learn_1.0', 
    wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid 
}


# In[26]:


#Save model
import joblib
model_details = wml_client.repository.store_model(
    model=DEMO_MODEL, 
    meta_props=model_props, 
    training_data=x_train, 
    training_target=y_train
)


# In[27]:


model_details


# In[28]:


model_id = wml_client.repository.get_model_id(model_details)
model_id


# In[29]:


# Set meta
deployment_props = {
    wml_client.deployments.ConfigurationMetaNames.NAME:DEPLOYMENT_NAME, 
    wml_client.deployments.ConfigurationMetaNames.ONLINE: {}
}


# In[30]:


# Deploy
deployment = wml_client.deployments.create(
    artifact_uid=model_id, 
    meta_props=deployment_props 
)


# In[ ]:




