#!/bin/bash

apt-get update
apt-get upgrade -y
apt-get install -y git
apt-get install -y npm
npm install pm2 -g

pip install -r requirements.txt