sudo apt update
sudo apt install -y python3-pip
git clone https://github.com/Kamlesh364/Object-Detection-Web-App.git
cd Object-Detection-Web-App
export PATH="$PATH:~/.local/bin"
pip install -r requirements.txt
sudo apt install -y python3-opencv
