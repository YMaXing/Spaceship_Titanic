Spaceship Titanic is my first ever data science project on Kaggle.


To avoid github ssh access denid in container in Windows:

Always run before running the container:
$ cp -r C:/Users/garru/.ssh .

Always run in the container:
$ eval "$(ssh-agent -s)"
$ ssh-add .ssh/id_ed25519