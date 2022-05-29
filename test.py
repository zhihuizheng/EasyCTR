import numpy as np
import pandas as pd


# data_path = 'data/criteo_x1/train_sample.csv'
# ddf = pd.read_csv(data_path, memory_map=True)
#
# a = set(list(ddf['C1']))
# print(len(a))
# print(a)
#
# print(ddf.describe())
# print(ddf.info())


# from sklearn.preprocessing import OrdinalEncoder
#
# # Create encoder
# ordinal_encoder = OrdinalEncoder()
#
# # Fit on training data
# ordinal_encoder.fit(np.array([1,2,3,4,5]).reshape(-1, 1))
#
# # Transform, notice that 0 and 6 are values that were never seen before
# ordinal_encoder.transform(np.array([0,1,2,3,4,5,6]).reshape(-1, 1))


from easyctr import preprocess

country_list = ['Argentina', 'Australia', 'Canada', 'France', 'Italy', 'Spain', 'US', 'Canada', 'Argentina', 'US']

label_encoder = preprocess.LabelEncoderExt()

label_encoder.fit(country_list)
print(label_encoder.classes_) # you can see new class called Unknown
print(label_encoder.transform(country_list))

new_country_list = ['Canada', 'France', 'Italy', 'Spain', 'US', 'India', 'Pakistan', 'South Africa']
print(label_encoder.transform(new_country_list))
