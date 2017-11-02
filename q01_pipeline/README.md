# Let begin to solve Machine Learning Challenges

* We have give you the leverage to use any Machine learning algorithm, any preprocessing technique you like to apply.
* We have just loaded the data and split in train and test.


## Write a function `pipeline` that:

* Will take test, train and model to create a function that will execute and return a AUC score and model.
* This model will be used to validate on our dataset and your score should be equal or more than threshold score value of AUC which is 0.72.
* Use random _state to 9.


### Parameters:

| Parameter | dtype | argument type | default value | description |
| --- | --- | --- | --- | --- |
| X_train | DataFrame | compulsory | | Dataframe containing feature variables for training|
| X_test | DataFrame | compulsory | | Dataframe containing feature variables for testing|
| y_train | Series/DataFrame | compulsory | | Training dataset target Variable |
| y_test | Series/DataFrame | compulsory | | Testing dataset target Variable |
| model | list | compulsory | | Which model needs to be build |

### Return :

| Return | dtype | description |
| --- | --- | --- |
| AUC | float | AUC score of the model |
| Model | model | Final model that is trained |
