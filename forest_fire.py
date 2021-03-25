import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
data = pd.read_csv("Data.csv")
data.head()
data.info()


# In[11]:


data = np.array(data)


# In[12]:


data


# In[13]:


x = data[:,1:-1]


# In[14]:


x


# In[15]:


y = data[:,-1]


# In[16]:


y


# In[17]:


y = y.astype('int')


# In[18]:


y


# In[19]:


x = x.astype('int')


# In[20]:


x


# In[21]:


X_train, X_test, y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)


# In[22]:


X_train,X_test


# In[23]:


log_reg = LogisticRegression()


# In[24]:


log_reg.fit(X_train, y_train)


# In[25]:


inputt = [int(X) for X in "45 32 60".split(' ')]


# In[26]:


inputt


# In[27]:


final = [np.array(inputt)]


# In[28]:


final


# In[31]:


b = log_reg.predict_proba(final)


# In[32]:


b


# In[33]:


pickle.dump(log_reg,open('model.pkl','wb'))


# In[34]:


model=pickle.load(open('model.pkl','rb'))


# In[ ]:





# In[ ]:




