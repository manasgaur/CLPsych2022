**CLPsych 2022 Shared Task Structure**

* Organize the files _(Next 2 weeks)_
* Task : Predict the moments of change in the posts made by the user. Following are some functions needed to create baseline models 
    * data_reader.py : Show take as input certain path to a training dataset containing all the timelines 
    * evaluator.py : Make custom functions for Precision, Recall, F1-Score, and other relevant metrics
    * utils.py : store the results inside utils. 
    * model_building.py : Deep language models specific for the task. Numpy and Torch are acceptable
* Each data point would be an array of:
    * Timeline ID
    * Post ID
    * User ID
    * Date
    * Label : ['IS', 'IE', 'O']
    * text : Post made a user at particular instance of time.
* While assessing the baseline models, we are specifically interested in 'IS' and 'IE' labels.