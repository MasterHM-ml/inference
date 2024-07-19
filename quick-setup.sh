cd /workspace
rm -rf inference
git clone https://github.com/MasterHM-ml/inference.git
cd inference
pip3 install -r requirements.txt
cd models
gdown --fuzzy https://drive.google.com/file/d/1RQXf0-Rp8ve25e2gd1gTW0wpfXyezWVh/view?usp=drive_link
cd ..
python3 app.py