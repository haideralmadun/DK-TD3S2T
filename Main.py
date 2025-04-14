import pandas as pd
from numpy import array
from DK_TD3S2T_model import DK_TD3S2T_model 
from summarize_average_performance import summarize_average_performance



# load the dataset
#dataset = pd.read_csv(r'changhua_Water_Rainfall.csv')

dataset = pd.read_csv(r'tunxi 1981-2016_interpolated.csv')



# remove date colume 
dataset_new= dataset.iloc[:,1:]

dataset_new= dataset_new.dropna()

# Using DataFrame.insert() to add a column
#dataset_new.insert(2, "output", dataset.iloc[:,7:], True)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
dataset_new_no = scaler.fit_transform(dataset_new)


# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = MinMaxScaler()
df_Close = pd.DataFrame(dataset_new['streamflow'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)




# convert data to dataframe to enable split it for three part

df = pd.DataFrame(dataset_new_no)


# split data to x.train y.train and  x.test y.test

dataset_train = df.iloc[:37622,:].values
dataset_val   = df.iloc[37622:42997,:].values
dataset_test  = df.iloc[42997:,:].values




   


# 1 , 2 and 3 hours prediction in future 
from keras import Model
from keras.layers import Layer
import keras.backend as K
from keras.layers import Input, Dense, LSTM , Reshape




# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)






# choose a number of time steps
n_steps_in, n_steps_out = 12, 6



# convert into input/output
X_train, y_train = split_sequences(dataset_train, n_steps_in, n_steps_out)
print(X_train.shape, y_train.shape)
X_val, y_val = split_sequences(dataset_val , n_steps_in, n_steps_out)
print(X_val.shape, y_val.shape)
X_test, y_test = split_sequences(dataset_test, n_steps_in, n_steps_out)
print(X_test.shape, y_test.shape)


n_features = X_train.shape[2]


n_seq = 2
n_steps = 6

# Reshape the input data
X_train1 = X_train.reshape((X_train.shape[0], n_seq, n_steps, n_features))
X_val1 = X_val.reshape((X_val.shape[0], n_seq, n_steps, n_features))
X_test1 = X_test.reshape((X_test.shape[0], n_seq, n_steps, n_features))





# Initialize the DK-TD3S2T model
model = DK_TD3S2T_model(
    n_steps_in=n_steps_in,
    n_steps_out=n_steps_out,
    n_features=n_features,
    n_seq=n_seq,
    n_steps=n_steps
)

# Show model summary
model.summary()


from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
										mode ="min", patience = 5,
										restore_best_weights = True)


# Fit the model
history = model.fit([X_train, X_train1], 
                    y_train, 
                    batch_size=200, 
                    epochs=100, 
                    callbacks=[earlystopping], 
                    validation_data=([X_val, X_val1], y_val))






    

    
# demonstrate prediction

y_pred = model.predict([X_test, X_test1] ,batch_size=200 , verbose=1)

# Unscale the predicted values
y_pred = scaler_pred.inverse_transform(y_pred)
# Unscale the actual values
y_test_unscaled = scaler_pred.inverse_transform(y_test)



# summarize average_performance
summarize_average_performance('DK-TD3S2T', y_test_unscaled, y_pred)




##### Load the model

with open('model.json','r') as f:
    json = f.read()
model.load_weights("DK-TD3STCN_tunxi.h5")

# summarize model.
model.summary()


