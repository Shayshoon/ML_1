
def prepare_data(training_data, new_data):
	import numpy as np
	import pandas as pd
	from sklearn.preprocessing import MinMaxScaler, StandardScaler

	# fill null values
	fillna_values = {"household_income": training_data.household_income.median(),
	 				 "PCR_02": training_data.PCR_02.median()}
	prepared_data = new_data.fillna(value=fillna_values)

	# scale using minmax

	minmax_scaler = MinMaxScaler(feature_range=(-1,1))
	minmax_features = ['PCR_01','PCR_03','PCR_04','PCR_06','PCR_07','PCR_08', 'PCR_09']

	minmax_scaler.fit(training_data[minmax_features])
	prepared_data[minmax_features] = 
		minmax_scaler.transform(new_data[minmax_features])

	# scale using standard

	standard_scaler = StandardScaler()
	standard_features = ['PCR_02','PCR_05','PCR_10']
	
	standard_scaler.fit(training_data[standard_features])
	prepared_data[standard_features] = 
		standard_scaler.transform(new_data[standard_features])

	# generate new feature `blood_type_group`

	prepared_data['blood_type_group'] = new_data['blood_type'].isin(["O+", "B+"])
	prepared_data = new_data.drop(columns=['blood_type'])

	return prepare_data