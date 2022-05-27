Install venv and requirements:

python3 -m venv venvName			//Creates virtual environment with name venvName
source venvName/bin/activate			//Activates virtual environment
python3 -m pip install -r requirements.txt	//Installs program dependences

python3 -m spacy download en_core_web_trf	//Download English pipeline for spacy methods 

python3 conceptual_maps.py -d ./data/input.txt	//Executes the program

deactivate					//Leaves virtual environment

