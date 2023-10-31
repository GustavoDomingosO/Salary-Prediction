import pandas as pd

# Importing the datasets
train_data = pd.read_csv("adult.data")
test_data = pd.read_csv("adult.test")

columns = ["age", "workclass", "fnlwgt", "education", "education-num",
                      "marital-status", "occupation", "relationship", "race", "sex", 
                      "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]



# Adding column names to the datasets
train_data.columns = ["age", "workclass", "fnlwgt", "education", "education-num",
                      "marital-status", "occupation", "relationship", "race", "sex", 
                      "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

test_data.columns = ["age", "workclass", "fnlwgt", "education", "education-num",
                      "marital-status", "occupation", "relationship", "race", "sex", 
                      "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

#Count the amount of data with "?" in every column
'''for column in columns:
    if train_data[column].dtype == 'O':  # 'O' represents object/string dtype
        print(column)
        print(train_data[column].str.strip().eq("?").sum())
        print(test_data[column].str.strip().eq("?").sum())
        print("\n")
    else:
        print(f"{column} is not a string column.")'''


#workclass, occupation, native-country have "?" in the data. So we will remove every line that has "?" in these columns
train_data = train_data[(train_data["workclass"].str.strip() != "?") & 
                        (train_data["occupation"].str.strip() != "?") & 
                        (train_data["native-country"].str.strip() != "?")]

test_data = test_data[(test_data["workclass"].str.strip() != "?") & 
                      (test_data["occupation"].str.strip() != "?") & 
                      (test_data["native-country"].str.strip() != "?")]

#Check if there are still "?" in the data
'''
for column in columns:
    if train_data[column].dtype == 'O':  # 'O' represents object/string dtype
        print(column)
        print(train_data[column].str.strip().eq("?").sum())
        print(test_data[column].str.strip().eq("?").sum())
        print("\n")
'''
#Check how many diffect work_class there are
#print(train_data["workclass"].value_counts())

#Change each work_class to a number
workclass_mapping = {
    "Private": 0,
    "Self-emp-not-inc": 1,
    "Local-gov": 2,
    "State-gov": 3,
    "Self-emp-inc": 4,
    "Federal-gov": 5,
    "Without-pay": 6
}

train_data["workclass"] = train_data["workclass"].map(workclass_mapping)
test_data["workclass"] = test_data["workclass"].map(workclass_mapping)

#Mapping the education column to a number
education_mapping = {
    "HS-grad": 0,
    "Some-college": 1,
    "Bachelors": 2,
    "Masters": 3,
    "Assoc-voc": 4,
    "11th": 5,
    "Assoc-acdm": 6,
    "10th": 7,
    "7th-8th": 8,
    "Prof-school": 9,
    "9th": 10,
    "12th": 11,
    "Doctorate": 12,
    "5th-6th": 13,
    "1st-4th": 14,
    "Preschool": 15
}
train_data["education"] = train_data["education"].map(education_mapping)
test_data["education"] = test_data["education"].map(education_mapping)

#Mapping the marital-status column to a number
marital_mapping = {
    "Married-civ-spouse": 0,
    "Never-married": 1,
    "Divorced": 2,
    "Separated": 3,
    "Widowed": 4,
    "Married-spouse-absent": 5,
    "Married-AF-spouse": 6
}
train_data["marital-status"] = train_data["marital-status"].map(marital_mapping)
test_data["marital-status"] = test_data["marital-status"].map(marital_mapping)

#Mapping for occupation
occupation_mapping = {
    "Prof-specialty": 0,
    "Craft-repair": 1,
    "Exec-managerial": 2,
    "Adm-clerical": 3,
    "Sales": 4,
    "Other-service": 5,
    "Machine-op-inspct": 6,
    "Transport-moving": 7,
    "Handlers-cleaners": 8,
    "Farming-fishing": 9,
    "Tech-support": 10,
    "Protective-serv": 11,
    "Priv-house-serv": 12,
    "Armed-Forces": 13
}
train_data["occupation"] = train_data["occupation"].map(occupation_mapping)
test_data["occupation"] = test_data["occupation"].map(occupation_mapping)

#Mapping for relationship
relationship_mapping = {
    "Husband": 0,
    "Not-in-family": 1,
    "Own-child": 2,
    "Unmarried": 3,
    "Wife": 4,
    "Other-relative": 5
}
train_data["relationship"] = train_data["relationship"].map(relationship_mapping)
test_data["relationship"] = test_data["relationship"].map(relationship_mapping)

#Mapping for race
race_mapping = {
    "White": 0,
    "Black": 1,
    "Asian-Pac-Islander": 2,
    "Amer-Indian-Eskimo": 3,
    "Other": 4,
}
train_data["race"] = train_data["race"].map(race_mapping)
test_data["race"] = test_data["race"].map(race_mapping)

#Mapping for sex
sex_mapping = {
    "Male": 0,
    "Female": 1,
}
train_data["sex"] = train_data["sex"].map(sex_mapping)
test_data["sex"] = test_data["sex"].map(sex_mapping)

#Mapping for native-country
#For native country we will only map the country to 0 or 1. 0 if the person is from the US and 1 if the person is from another country
native_mapping = {
    "United-States": 0,
    "Mexico": 1,
    "Philippines": 1,
    "Germany": 1,
    "Puerto-Rico": 1,
    "Canada": 1,
    "El-Salvador": 1,
    "India": 1,
    "Cuba": 1,
    "England": 1,
    "Jamaica": 1,
    "South": 1,
    "China": 1,
    "Italy": 1,
    "Dominican-Republic": 1,
    "Vietnam": 1,
    "Guatemala": 1,
    "Japan": 1,
    "Poland": 1,
    "Columbia": 1,
    "Taiwan": 1,
    "Haiti": 1,
    "Iran": 1,
    "Portugal": 1,
    "Nicaragua": 1,
    "Peru": 1,
    "Greece": 1,
    "France": 1,
    "Ecuador": 1,
    "Ireland": 1,
    "Hong": 1,
    "Cambodia": 1,
    "Trinadad&Tobago": 1,
    "Laos": 1,
    "Thailand": 1,
    "Yugoslavia": 1,
    "Outlying-US(Guam-USVI-etc)": 1,
    "Hungary": 1,
    "Honduras": 1,
    "Scotland": 1,
    "Holand-Netherlands": 1
}
train_data["native-country"] = train_data["native-country"].map(native_mapping)
test_data["native-country"] = test_data["native-country"].map(native_mapping)

#Change the income to 0 or 1, in which 0 is <=50K and 1 is >50K
income_mapping = {
    "<=50K": 0,
    ">50K": 1
}
train_data["income"] = train_data["income"].map(income_mapping)
test_data["income"] = test_data["income"].map(income_mapping)

#Next we will normalize the data