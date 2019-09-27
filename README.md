# Google_Toxicity

---

**This repository contains a project built off of my submission to Jigsaw's Kaggle competition. The Python model itself is hosted on a Flask server with a "/toxicity" endpoint. If you want to find the toxicity score for a given input string, all that's necessary is to query this endpoint with the proper syntax for a POST request. I also built a simple web application to expedite the process in the style of Google's homepage.**

---

### Installation: 

In order to run this project, you'll need to install the latest version of NodeJS/npm (https://nodejs.org/en/download/) as well as Python 3.7.1 and its respective "pip" installer. I included a "package.json" file to indicate the required JS packages and a "requirements.txt" file for the necessary Python ones. After downloading both NodeJS and Python, run the following commands to start the downloads:

<pre>
cd google_toxicity
sudo npm install
sudo pip install -r requirements.txt
</pre>

### Models:

In order to save space, I hosted the trained models on Google Drive. Use the following links to download both and place them in the "service" folder as indicated by ".gitkeep"

- Link 1 - bert_pytorch_model.bin - [Google Drive Link (bert_pytorch_model.bin)](https://drive.google.com/a/virginia.edu/uc?id=1TBflpUhF02InlANzLWMBYd6NoAvV4ldc&export=download)
- Link 2 - uncased_L-12_H-768_A-12 - [Google Drive Link (uncased-L-12_H-768_A-12)](https://drive.google.com/a/virginia.edu/uc?id=1LI38JwWIMB7vZoU8UFuLjvWcU0Ae6FFt&export=download)

### Using the Service: 

Once everything is installed, I included a "toxicity.py" file to run all necessary commands for which to start the python/react servers. It should ensure that both are running on different ports--3000/3001 respectively--and take care of compiling and hosting the web application on localhost.

<pre>
sudo python toxicity.py
</pre>

In order to calculate the toxicity score for a given input string, enter it into the search-bar shown and either press "Enter" or click the "Calculate Toxicity" button.
