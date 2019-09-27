import os
import subprocess
from service.flask_service import main as FlaskApp

def main():

    # Start Web Application (NodeJS)"
    subprocess.call("sudo npm start &", shell=True)
    # Start Flask Server (Python)
    flask_server = FlaskApp()

if __name__ == "__main__":
    main()
