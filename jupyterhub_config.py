# Configuration file for jupyterhub.

c = get_config()  # noqa


c.JupyterHub.authenticator_class = "dummyauthenticator.DummyAuthenticator"
import os

c.DummyAuthenticator.password = os.environ.get("JUPYTER_DEMO_PASSWORD", "")

c.Authenticator.allowed_users = {"thom", "demo"}


# Force the user's notebook directory
c.Spawner.default_url = "/lab"

# Change working directory when spawning
c.Spawner.notebook_dir = "/home/thom/Desktop/CIPM/FruitFlyPheno/fflies"

# Ensure user is chrooted to that directory (no ".." access)
c.Spawner.args = ["--NotebookApp.allow_origin=*"]

# hosting
c.JupyterHub.bind_url = "http://127.0.0.1:8000"
