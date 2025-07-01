#!/bin/bash
(.venv/bin/jupyterhub -f jupyterhub_config.py &) && sleep 5 && ngrok http --domain=correctly-precise-impala.ngrok-free.app 8000

