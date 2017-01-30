#!/bin/bash

wget -nc https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip
wget -nc https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip

unzip non-vehicles.zip non-vehicles/*
unzip vehicles.zip vehicles/*
